"""Phase 4 — Location–Location Spatial Relation Extraction (OVERHAUL).

⚠️  SCHEMA OVERHAUL NOTICE
    This phase has been completely re-scoped. The previous Phase 4 emitted
    a 15-type geocoding-oriented `SpatialRelation` taxonomy (north_of,
    on_coast, within, distance_approx, …). It has been **replaced** by a
    location–location spatial-role-labeling task whose unit of output is a
    sentence-grounded `SentenceLocationRelations` record carrying one or
    more `LocationRelation` triples (location_1, location_2,
    spatial_indicator, semantic_type ∈ {REGION, DIRECTION, DISTANCE}).
    Records are only written when ≥1 valid relation is found.

Annotation scheme reference:
    Kordjamshidi, P., van Otterlo, M., & Moens, M.-F. (2017).
    "Spatial Role Labeling Annotation Scheme."
    In N. Ide & J. Pustejovsky (Eds.), *Handbook of Linguistic Annotation*.
    Springer.

Backend (unchanged): local Mistral 7B served by Ollama.

Run:
    python -m src.pipeline --config config.yaml --phase 4 --force

Output:
    data/location_relations.jsonl   — one JSON record per sentence that
                                      contains ≥1 location–location relation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from src.utils.io import data_dir, iter_jsonl, read_json, write_jsonl
from src.utils.schemas import (
    LocationRelation,
    SentenceLocationRelations,
    SentenceRecord,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spatial-indicator → semantic-type lexicon
# ---------------------------------------------------------------------------

VALID_SEMANTIC_TYPES: Set[str] = {"REGION", "DIRECTION", "DISTANCE"}

# Default mapping used to validate / fall back when the model omits the type.
# Keys are lowercased indicator strings.
INDICATOR_LEXICON: Dict[str, str] = {
    # REGION — containment / position
    "in": "REGION",
    "on": "REGION",
    "inside": "REGION",
    "outside": "REGION",
    "within": "REGION",
    "behind": "REGION",
    "in front of": "REGION",
    "between": "REGION",
    "among": "REGION",
    "above": "REGION",
    "below": "REGION",
    "beneath": "REGION",
    "under": "REGION",
    "underneath": "REGION",
    "on top of": "REGION",
    "to the left of": "REGION",
    "to the right of": "REGION",
    "at": "REGION",
    # DIRECTION — orientation / path
    "toward": "DIRECTION",
    "towards": "DIRECTION",
    "into": "DIRECTION",
    "onto": "DIRECTION",
    "through": "DIRECTION",
    "across": "DIRECTION",
    "away from": "DIRECTION",
    "out of": "DIRECTION",
    "up": "DIRECTION",
    "down": "DIRECTION",
    "along": "DIRECTION",
    # DISTANCE — proximity
    "near": "DISTANCE",
    "close to": "DISTANCE",
    "far from": "DISTANCE",
    "next to": "DISTANCE",
    "beside": "DISTANCE",
    "adjacent to": "DISTANCE",
    "by": "DISTANCE",
}

# Sentences containing only these surface forms are almost always metaphorical
# rather than spatial. They aren't a hard reject — the LLM is the primary
# filter — but they trigger an extra sanity check on the proposed relation.
_METAPHORICAL_HINTS: Set[str] = {
    "in love", "in trouble", "in charge", "in mind", "in time",
    "on time", "on edge", "in danger", "in luck",
}


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Thin, stateless wrapper around the Ollama /api/chat endpoint."""

    def __init__(self, host: str, model: str, timeout: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._chat_url = f"{self.host}/api/chat"
        self._tags_url = f"{self.host}/api/tags"

    def is_available(self) -> bool:
        try:
            return requests.get(self._tags_url, timeout=5).status_code == 200
        except Exception:
            return False

    def is_model_pulled(self) -> bool:
        try:
            r = requests.get(self._tags_url, timeout=5)
            if r.status_code != 200:
                return False
            target = self.model.split(":")[0].lower()
            return any(
                m.get("name", "").split(":")[0].lower() == target
                for m in r.json().get("models", [])
            )
        except Exception:
            return False

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }
        resp = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a spatial-role-labeling system. Your job is to identify LOCATION spans
in a sentence and extract spatial relationships **between locations only**.

━━ WHAT COUNTS AS A LOCATION ━━
  • Physical places: "the table", "the room", "the park"
  • Geographic entities: "Paris", "the river", "the mountain"
  • Spatial regions: "inside the house", "the left side of the building"

DO NOT treat as locations:
  • People, animals, or objects (e.g. "the cat", "the car", "John")
  • Abstract or metaphorical uses (e.g. "in trouble", "in love", "in charge")

━━ SPATIAL INDICATORS ━━
  • Prepositions: in, on, under, behind, near, inside, outside
  • Multi-word: in front of, to the left of, next to, on top of
  • Directional: toward, away from, into, through, across

━━ SEMANTIC TYPES ━━
  • REGION    — containment or position ("in the room", "on the table")
  • DIRECTION — orientation or path ("toward the city", "into the garden")
  • DISTANCE  — proximity ("near the park", "far from the river")
A relation may carry one or more types; pick the most precise.

━━ RULES ━━
  1. Only emit a relation when BOTH location_1 AND location_2 are locations.
  2. If either side is a person/animal/object/abstract idea → emit nothing
     for that pair.
  3. Use exact spans copied verbatim from the sentence (preserve articles,
     e.g. "the park", not "park").
  4. spatial_indicator must be the exact cue word/phrase from the sentence.
  5. A sentence may yield multiple relations.
  6. If no valid location–location relation exists, return an empty list.

━━ OUTPUT ━━
Return ONLY a valid JSON object of the form:
{
  "location_relations": [
    {
      "location_1": "...",
      "location_2": "...",
      "spatial_indicator": "...",
      "semantic_type": ["REGION" | "DIRECTION" | "DISTANCE", ...]
    }
  ]
}

━━ EXAMPLE ━━
Sentence: "The park is near the river."
Output: {"location_relations":[{"location_1":"the park","location_2":"the river","spatial_indicator":"near","semantic_type":["DISTANCE"]}]}

━━ NON-EXAMPLE ━━
Sentence: "The cat is on the table."
Output: {"location_relations":[]}
Reason: "the cat" is not a location.
"""

_FALLBACK_SUFFIX = (
    "\n\nCRITICAL: Return ONLY a JSON object with key 'location_relations'. "
    "No markdown, no backticks, no explanation."
)


def _build_user_prompt(sentence: str) -> str:
    return f"SENTENCE:\n{sentence.strip()}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _span_in_sentence(span: str, sentence: str) -> bool:
    """Verify that `span` appears (case-insensitively) in `sentence`."""
    if not span:
        return False
    return _norm(span) in _norm(sentence)


def _infer_semantic_types(indicator: str) -> List[str]:
    """Fallback: derive semantic type from the indicator lexicon."""
    key = _norm(indicator)
    if key in INDICATOR_LEXICON:
        return [INDICATOR_LEXICON[key]]
    # Try multi-word match against known phrases (longest first)
    for phrase in sorted(INDICATOR_LEXICON, key=len, reverse=True):
        if phrase in key or key in phrase:
            return [INDICATOR_LEXICON[phrase]]
    return []


def _is_likely_metaphorical(sentence: str, indicator: str, loc1: str, loc2: str) -> bool:
    """Catch obvious metaphorical idioms that slipped past the model."""
    s_norm = _norm(sentence)
    pair = f"{_norm(indicator)} {_norm(loc2)}"
    for hint in _METAPHORICAL_HINTS:
        if hint in s_norm and hint in pair:
            return True
    return False


def _validate_relation(raw: dict, sentence: str) -> Optional[LocationRelation]:
    if not isinstance(raw, dict):
        return None

    loc1 = str(raw.get("location_1", "")).strip()
    loc2 = str(raw.get("location_2", "")).strip()
    indicator = str(raw.get("spatial_indicator", "")).strip()

    if not loc1 or not loc2 or not indicator:
        return None
    if _norm(loc1) == _norm(loc2):
        return None

    # Spans must come from the sentence.
    if not (_span_in_sentence(loc1, sentence)
            and _span_in_sentence(loc2, sentence)
            and _span_in_sentence(indicator, sentence)):
        log.debug("Discarding: span(s) not found verbatim in sentence")
        return None

    if _is_likely_metaphorical(sentence, indicator, loc1, loc2):
        log.debug("Discarding likely-metaphorical relation: %r", raw)
        return None

    # Semantic types: keep only the valid set; fall back to lexicon.
    raw_types = raw.get("semantic_type") or []
    if isinstance(raw_types, str):
        raw_types = [raw_types]
    types = [t.upper() for t in raw_types if isinstance(t, str) and t.upper() in VALID_SEMANTIC_TYPES]
    if not types:
        types = _infer_semantic_types(indicator)
    if not types:
        log.debug("Discarding: no valid semantic_type for indicator %r", indicator)
        return None

    return LocationRelation(
        location_1=loc1,
        location_2=loc2,
        spatial_indicator=indicator,
        semantic_type=sorted(set(types)),
    )


def _parse_response(raw_text: str, sentence: str) -> List[LocationRelation]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("location_relations") or data.get("relations") or []
    else:
        return []
    if not isinstance(items, list):
        return []

    out: List[LocationRelation] = []
    seen: Set[Tuple[str, str, str]] = set()
    for item in items:
        rel = _validate_relation(item, sentence)
        if rel is None:
            continue
        key = (_norm(rel.location_1), _norm(rel.spatial_indicator), _norm(rel.location_2))
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


# ---------------------------------------------------------------------------
# Per-sentence extraction
# ---------------------------------------------------------------------------

def _extract_sentence(
    client: OllamaClient,
    sentence_text: str,
    max_retries: int = 3,
    temperature: float = 0.1,
) -> Tuple[List[LocationRelation], Optional[str]]:
    user_prompt = _build_user_prompt(sentence_text)
    for attempt in range(max_retries):
        system = _SYSTEM_PROMPT + (_FALLBACK_SUFFIX if attempt > 0 else "")
        retry_temp = min(temperature + attempt * 0.05, 0.3)
        try:
            raw = client.chat(system, user_prompt, temperature=retry_temp)
            return _parse_response(raw, sentence_text), None
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as exc:
            log.error("Connection error: %s", exc)
            if attempt < max_retries - 1:
                time.sleep(5)
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP error: %s", exc)
            break
        except Exception as exc:
            log.exception("Unexpected error: %s", exc)
            if attempt < max_retries - 1:
                time.sleep(1)
    return [], f"failed after {max_retries} attempts"


# ---------------------------------------------------------------------------
# Checkpoint (resume on interrupt)
# ---------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.processed_ids: Set[str] = set()
        self.records: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
            self.processed_ids = set(data.get("processed_sentence_ids", []))
            self.records = data.get("records", [])
            log.info(
                "Checkpoint loaded — %d sentences done, %d records emitted",
                len(self.processed_ids), len(self.records),
            )
        except Exception as exc:
            log.warning("Checkpoint load failed (%s) — starting fresh", exc)

    def save(self, sentence_id: str, record: Optional[dict]) -> None:
        self.processed_ids.add(sentence_id)
        if record is not None:
            self.records.append(record)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "processed_sentence_ids": list(self.processed_ids),
                    "records": self.records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def is_done(self, sentence_id: str) -> bool:
        return sentence_id in self.processed_ids


# ---------------------------------------------------------------------------
# Progress tracker (feeds the dashboard)
# ---------------------------------------------------------------------------

class ProgressTracker:
    def __init__(self, path: Path):
        self.path = path
        self._state: dict = {}

    def initialize(
        self,
        doc_id: str,
        model: str,
        total_sentences: int,
        resumed_done: int = 0,
        resumed_relations: int = 0,
    ) -> None:
        self._state = {
            "status": "running",
            "task": "location_location_spatial_relations",
            "doc_id": doc_id,
            "model": model,
            "total_sentences": total_sentences,
            "processed_sentences": resumed_done,
            "error_sentences": 0,
            "relations_extracted": resumed_relations,
            "semantic_type_counts": {"REGION": 0, "DIRECTION": 0, "DISTANCE": 0},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "current_sentence_preview": "",
            "recent_relations": [],
            "error_log": [],
        }
        self._write()

    def sentence_started(self, preview: str) -> None:
        self._state["current_sentence_preview"] = preview[:300]
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def sentence_done(self, relations: List[LocationRelation], error: Optional[str] = None) -> None:
        if error:
            self._state["error_sentences"] += 1
            self._state["error_log"].append(error)
            self._state["error_log"] = self._state["error_log"][-20:]
        else:
            self._state["processed_sentences"] += 1

        self._state["relations_extracted"] += len(relations)
        for r in relations:
            for t in r.semantic_type:
                self._state["semantic_type_counts"][t] = (
                    self._state["semantic_type_counts"].get(t, 0) + 1
                )
            self._state["recent_relations"].append({
                "location_1": r.location_1,
                "spatial_indicator": r.spatial_indicator,
                "location_2": r.location_2,
                "semantic_type": r.semantic_type,
            })
        self._state["recent_relations"] = self._state["recent_relations"][-8:]
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def finalize(self, status: str = "complete") -> None:
        self._state["status"] = status
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        tmp.replace(self.path)


# ---------------------------------------------------------------------------
# Phase entry point
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 4: Location–Location Spatial Relation Extraction ===")
    log.info("Schema: SentenceLocationRelations (REGION / DIRECTION / DISTANCE)")

    dd = data_dir(cfg)
    output_path = dd / "location_relations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        log.info("location_relations.jsonl exists — skipping (use --force to re-run).")
        return

    rel_cfg = cfg.get("relations", {})
    m_cfg = rel_cfg.get("mistral", {})

    host        = m_cfg.get("host", "http://localhost:11434")
    model       = m_cfg.get("model", "mistral")
    temperature = float(m_cfg.get("temperature", 0.1))
    timeout     = int(m_cfg.get("timeout_seconds", 120))
    max_retries = int(m_cfg.get("max_retries", 3))

    progress_path   = dd / rel_cfg.get("progress_file", "phase4_progress.json")
    checkpoint_path = dd / rel_cfg.get("checkpoint_file", "phase4_checkpoint.json")

    # ── Preflight ─────────────────────────────────────────────────────────
    client = OllamaClient(host=host, model=model, timeout=timeout)
    if not client.is_available():
        raise RuntimeError(
            f"Ollama is not running at {host}.\n"
            "  → Start it:  ollama serve\n"
            f"  → Pull model: ollama pull {model}"
        )
    if not client.is_model_pulled():
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n  → Run: ollama pull {model}"
        )
    log.info("Ollama ready  model=%s  host=%s", model, host)

    # ── Load corpus sentences ─────────────────────────────────────────────
    cleaned_dir   = Path(cfg["corpus"]["cleaned_dir"])
    metadata_file = Path(cfg["corpus"]["metadata_file"])
    doc_filter    = set(cfg["corpus"].get("doc_filter", []))

    if metadata_file.exists():
        metadata = read_json(metadata_file)
        doc_ids = [
            m["doc_id"] for m in metadata
            if not doc_filter or m["doc_id"] in doc_filter
        ]
    else:
        doc_ids = list(doc_filter) if doc_filter else []

    all_sentences: List[SentenceRecord] = []
    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found: %s", cleaned_path)
            continue
        all_sentences.extend(iter_jsonl(cleaned_path, model=SentenceRecord))

    log.info("Total sentences to process: %d", len(all_sentences))

    # ── Checkpoint + progress ─────────────────────────────────────────────
    checkpoint = ProgressTracker  # silence linter
    cp = Checkpoint(checkpoint_path)
    progress = ProgressTracker(progress_path)
    progress.initialize(
        doc_id=doc_ids[0] if doc_ids else "unknown",
        model=model,
        total_sentences=len(all_sentences),
        resumed_done=len(cp.processed_ids),
        resumed_relations=sum(len(r.get("location_relations", [])) for r in cp.records),
    )

    # ── Main loop ─────────────────────────────────────────────────────────
    for sent in all_sentences:
        if cp.is_done(sent.sentence_id):
            continue

        progress.sentence_started(sent.text[:250].replace("\n", " "))

        t0 = time.monotonic()
        relations, error = _extract_sentence(
            client=client,
            sentence_text=sent.text,
            max_retries=max_retries,
            temperature=temperature,
        )
        duration = time.monotonic() - t0

        # Only persist sentences that have ≥1 valid relation.
        if relations:
            record = SentenceLocationRelations(
                doc_id=sent.doc_id,
                sentence_id=sent.sentence_id,
                sentence=sent.text,
                location_relations=relations,
            )
            cp.save(sent.sentence_id, record.model_dump())
            log.info(
                "%-40s  +%d relation(s)  (%.1fs)",
                sent.sentence_id, len(relations), duration,
            )
        else:
            cp.save(sent.sentence_id, None)

        progress.sentence_done(relations=relations, error=error)

    # ── Write final output ────────────────────────────────────────────────
    final_records: List[SentenceLocationRelations] = [
        SentenceLocationRelations(**r) for r in cp.records
    ]
    write_jsonl(output_path, final_records, overwrite=True)
    progress.finalize("complete")

    # Summary
    total_rel = sum(len(r.location_relations) for r in final_records)
    type_counts = {"REGION": 0, "DIRECTION": 0, "DISTANCE": 0}
    for r in final_records:
        for rel in r.location_relations:
            for t in rel.semantic_type:
                type_counts[t] = type_counts.get(t, 0) + 1

    log.info(
        "Phase 4 complete — %d sentences with relations, %d total relations → %s",
        len(final_records), total_rel, output_path,
    )
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info("  %-10s %d", t, c)
