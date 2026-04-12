"""Stage 2 (Strategy B) — Joint Mistral NER + Spatial Relation Extraction.

Single Ollama call per chunk returns structured {entities, relations}.
CoReNer NER is bypassed; entity discovery is model-driven with no static
blocklist.  Outputs:

  - data/entities.jsonl   (Entity objects for downstream coref + grounding)
  - data/phase4_checkpoint.json  (relation candidates, consumed later by
    the graph-building stage after coref and grounding are complete)

Reuses the OllamaClient, Checkpoint, and ProgressTracker patterns from the
original phase4_relations module.

Run:
    python -m src.pipeline --config config.yaml --phase 2 --force
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from src.utils.io import data_dir, iter_jsonl, read_json, write_jsonl
from src.utils.schemas import Entity, EntityMention, SentenceRecord

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RELATION_TYPES: Set[str] = {
    "near", "far",
    "north_of", "south_of", "east_of", "west_of",
    "across", "on_coast",
    "within", "contains", "part_of", "borders",
    "on_shore_of", "connected_via", "distance_approx",
}

NULLABLE_ENTITY2_TYPES: Set[str] = {"on_coast"}

_EVIDENCE_NULL_VALUES: Set[str] = {
    "", "none", "null", "n/a", "not stated", "not mentioned",
    "not found", "not specified", "not applicable",
}

# Known fictional Fitzgerald places — used for type classification only,
# NOT as a blocklist.
KNOWN_FICTIONAL: Set[str] = {
    "East Egg",
    "West Egg",
    "Valley of Ashes",
    "Eggs",
}


# ---------------------------------------------------------------------------
# Ollama HTTP client (same as phase4_relations)
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
            r = requests.get(self._tags_url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def is_model_pulled(self) -> bool:
        try:
            r = requests.get(self._tags_url, timeout=5)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            target = self.model.split(":")[0].lower()
            return any(m.get("name", "").split(":")[0].lower() == target for m in models)
        except Exception:
            return False

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> str:
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
                "num_predict": 4096,
            },
        }
        resp = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Checkpoint (persist processed chunks + entities + relations to disk)
# ---------------------------------------------------------------------------

class Checkpoint:
    """Persist processed chunk IDs, accumulated entities, and relation
    candidates for resume support."""

    def __init__(self, path: Path):
        self.path = path
        self.processed_ids: Set[str] = set()
        self.entities: Dict[str, dict] = {}   # canonical_name -> entity info
        self.relations: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
            self.processed_ids = set(data.get("processed_chunk_ids", []))
            self.entities = data.get("entities", {})
            self.relations = data.get("relations", [])
            log.info(
                "Checkpoint loaded: %d chunks done, %d entities, %d relations",
                len(self.processed_ids), len(self.entities), len(self.relations),
            )
        except Exception as exc:
            log.warning("Checkpoint load failed (%s) — starting fresh", exc)

    def save(
        self,
        chunk_id: str,
        new_entities: Dict[str, dict],
        new_relations: List[dict],
    ) -> None:
        self.processed_ids.add(chunk_id)
        # Merge entity info (accumulate mentions)
        for name, info in new_entities.items():
            if name in self.entities:
                self.entities[name]["mentions"].extend(info["mentions"])
                self.entities[name]["doc_ids"] = list(
                    set(self.entities[name]["doc_ids"]) | set(info["doc_ids"])
                )
            else:
                self.entities[name] = info
        self.relations.extend(new_relations)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "processed_chunk_ids": list(self.processed_ids),
                    "entities": self.entities,
                    "relations": self.relations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def is_done(self, chunk_id: str) -> bool:
        return chunk_id in self.processed_ids


# ---------------------------------------------------------------------------
# Progress tracker (feeds the live dashboard)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Writes a progress JSON file after every chunk for the dashboard."""

    def __init__(self, path: Path):
        self.path = path
        self._state: dict = {}

    def initialize(
        self,
        doc_id: str,
        model: str,
        total_chunks: int,
        resumed_done: int = 0,
        resumed_relations: int = 0,
        resumed_entities: int = 0,
        resumed_type_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self._state = {
            "status": "running",
            "stage": "mistral_joint_extraction",
            "doc_id": doc_id,
            "model": model,
            "total_chunks": total_chunks,
            "processed_chunks": resumed_done,
            "error_chunks": 0,
            "relations_extracted": resumed_relations,
            "entities_extracted": resumed_entities,
            "relation_type_counts": dict(resumed_type_counts or {}),
            "chunk_durations_s": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "current_chunk_preview": "",
            "recent_relations": [],
            "error_log": [],
        }
        self._write()

    def chunk_started(self, preview: str) -> None:
        self._state["current_chunk_preview"] = preview[:300]
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        self._write()

    def chunk_done(
        self,
        new_relations: List[dict],
        new_entity_count: int,
        duration_s: float,
        error: Optional[str] = None,
    ) -> None:
        if error:
            self._state["error_chunks"] += 1
            self._state["error_log"].append(error)
            self._state["error_log"] = self._state["error_log"][-20:]
        else:
            self._state["processed_chunks"] += 1

        self._state["relations_extracted"] += len(new_relations)
        self._state["entities_extracted"] += new_entity_count
        counts = self._state["relation_type_counts"]
        for r in new_relations:
            rt = r.get("relation_type", "unknown")
            counts[rt] = counts.get(rt, 0) + 1

        self._state["chunk_durations_s"].append(round(duration_s, 2))
        self._state["chunk_durations_s"] = self._state["chunk_durations_s"][-100:]

        for r in new_relations:
            self._state["recent_relations"].append({
                "entity_1": r["entity_1"],
                "relation_type": r["relation_type"],
                "entity_2": r.get("entity_2"),
                "confidence": r.get("confidence", 0.0),
                "evidence": r.get("evidence", "")[:120],
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
# Prompts — joint extraction (entities + relations in one call)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise geographic entity and spatial relation extractor for literary texts.
Read the provided passage and:
1. Identify all geographic entities (places, regions, bodies of water, landmarks, streets, buildings).
2. Extract spatial relationships between those entities.

== ENTITY RULES ==
- Include ANY named place, region, body of water, landmark, street, or building mentioned in the passage.
- Classify each entity as "GPE" (geopolitical: city, state, country), "LOC" (natural: river, mountain, sea), or "FAC" (facility: building, bridge, road, station).
- Include fictional/invented places (e.g. "East Egg") — classify them with the most appropriate NER label.
- Do NOT include person names, character names, or non-geographic nouns.
- Use the entity name exactly as it appears in the text. Normalise abbreviations (e.g. "N.Y." -> "New York").

== RELATION RULES ==
1. Only extract relations between entities you listed.
2. Only extract relations explicitly stated or unambiguously implied — never speculate.
3. Quality over quantity: when uncertain, omit.
4. The "evidence" field must be a verbatim or near-verbatim quote from the passage.
5. entity_2 may be null ONLY for "on_coast" when no body of water is named.

== RELATION TYPES ==
  near           — close proximity (~10 km)
  far            — distant (>50 km)
  north_of       — entity_1 is north of entity_2
  south_of       — entity_1 is south of entity_2
  east_of        — entity_1 is east of entity_2
  west_of        — entity_1 is west of entity_2
  across         — entity_1 is across a body of water from entity_2
  on_coast       — entity_1 is on a shoreline (entity_2 = body of water or null)
  within         — entity_1 is inside entity_2
  contains       — entity_1 contains entity_2
  part_of        — entity_1 is administratively part of entity_2
  borders        — entity_1 and entity_2 share a boundary
  on_shore_of    — entity_1 is on the bank/shore of entity_2
  connected_via  — entity_1 and entity_2 are connected by a road/bridge/waterway
  distance_approx — entity_1 is approximately N units from entity_2

== OUTPUT FORMAT ==
Return ONLY a valid JSON object. No markdown, no explanation.

{
  "entities": [
    {
      "name": "<entity name as in text>",
      "ner_label": "GPE" | "LOC" | "FAC",
      "classification": "real" | "fictional" | "uncertain"
    }
  ],
  "relations": [
    {
      "entity_1": "<exact entity name>",
      "relation_type": "<type from list above>",
      "entity_2": "<exact entity name or null>",
      "confidence": <0.5-1.0>,
      "distance_value": <number or null>,
      "distance_unit": "miles" | "km" | null,
      "evidence": "<verbatim quote from passage>",
      "reasoning": "<one sentence>"
    }
  ]
}

If no entities found, return: {"entities": [], "relations": []}
If entities but no relations, return: {"entities": [...], "relations": []}

Confidence guide:
  0.90-1.00 -> explicitly stated
  0.70-0.89 -> clearly implied
  0.50-0.69 -> weakly implied (use sparingly)
"""

_FALLBACK_PROMPT_SUFFIX = (
    "\n\nCRITICAL: Return ONLY a valid JSON object with this exact structure: "
    '{"entities": [...], "relations": [...]}. No markdown, no backticks, no explanation.'
)


def _build_user_prompt(chunk_text: str) -> str:
    return f"PASSAGE:\n{chunk_text}"


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

def _build_chunks(
    sentences: List[SentenceRecord],
    chunk_size: int = 6,
    overlap: int = 2,
) -> List[Tuple[str, List[SentenceRecord]]]:
    if not sentences:
        return []
    stride = max(1, chunk_size - overlap)
    chunks: List[Tuple[str, List[SentenceRecord]]] = []
    doc_id = sentences[0].doc_id
    idx = 0
    for i in range(0, len(sentences), stride):
        batch = sentences[i : i + chunk_size]
        chunk_id = f"{doc_id}_chunk_{idx:04d}"
        chunks.append((chunk_id, batch))
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Entity name normalisation
# ---------------------------------------------------------------------------

# Surface normalisation map (abbreviations found in Fitzgerald)
SURFACE_NORM: Dict[str, str] = {
    "N.Y.": "New York",
    "N.Y": "New York",
    "NYC": "New York City",
    "N.Y.C.": "New York City",
    "L.I.": "Long Island",
    "L.I": "Long Island",
}


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _canonical_name(raw_name: str) -> str:
    """Produce a stable canonical name from a raw entity name."""
    stripped = raw_name.strip()
    # Apply surface normalisation
    normed = SURFACE_NORM.get(stripped, stripped)
    # Title-case for consistency, but preserve known forms
    if normed == normed.lower() and len(normed) > 2:
        normed = normed.title()
    return normed


def _classify_entity(name: str, classification_hint: str, fictional_overrides: Set[str]) -> str:
    """Determine entity type: real, fictional, or uncertain."""
    if name in KNOWN_FICTIONAL or name in fictional_overrides:
        return "fictional"
    hint = classification_hint.lower().strip()
    if hint == "fictional":
        return "fictional"
    if hint == "real":
        return "real"
    # Heuristic fallback: check against known real indicators
    real_indicators = {
        "New York", "Long Island", "Manhattan", "Brooklyn", "Queens",
        "Bronx", "Connecticut", "New Jersey", "Chicago", "London",
        "Paris", "Europe", "America", "United States", "Oxford",
        "Louisville", "Minnesota", "Santa Barbara", "San Francisco",
        "Venice", "Rome", "Ohio", "Indiana", "Missouri",
        "Normandy", "Marseilles", "Versailles", "Castile",
        "France", "England", "Germany", "Spain", "Italy",
        "Brazil", "Rio de Janeiro", "Manaus", "Amazon",
        "Madeira", "Bolivia", "Peru", "Buenos Aires",
        "Montenegro", "Wisconsin", "Lake Superior", "Lake Michigan",
        "Atlantic", "Pacific", "Mediterranean", "Fifth Avenue",
        "Broadway", "Wall Street", "Central Park",
    }
    if name in real_indicators:
        return "real"
    real_suffixes = (
        "City", " Island", " River", " Lake", " Street",
        " Avenue", " Road", " Bay", " County", " State",
        " Ocean", " Sea", " Mountain", " Mountains", " Valley",
    )
    if any(name.endswith(sfx) for sfx in real_suffixes):
        return "real"
    return "uncertain"


# ---------------------------------------------------------------------------
# Response parsing — joint entities + relations
# ---------------------------------------------------------------------------

def _parse_model_response(
    raw_text: str,
    chunk_text: str,
    sentences: List[SentenceRecord],
    doc_id: str,
) -> Tuple[Dict[str, dict], List[dict]]:
    """Parse Mistral's joint JSON response.

    Returns:
        (entity_registry, validated_relations)
        entity_registry: canonical_name -> {name, canonical_name, ner_label,
                         classification, mentions, doc_ids}
        validated_relations: list of cleaned relation dicts
    """
    # Parse JSON
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            log.warning("No JSON object found in model response")
            return {}, []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            log.warning("JSON parse failed after salvage: %s", exc)
            return {}, []

    if not isinstance(data, dict):
        log.warning("Unexpected top-level JSON type: %s", type(data))
        return {}, []

    # --- Parse entities ---
    raw_entities = data.get("entities", [])
    if not isinstance(raw_entities, list):
        raw_entities = []

    entity_registry: Dict[str, dict] = {}
    entity_name_lookup: Dict[str, str] = {}  # normalized -> canonical

    for raw_ent in raw_entities:
        if not isinstance(raw_ent, dict):
            continue
        raw_name = str(raw_ent.get("name", "")).strip()
        if not raw_name or len(raw_name) < 2:
            continue
        ner_label = str(raw_ent.get("ner_label", "LOC")).strip().upper()
        if ner_label not in ("GPE", "LOC", "FAC"):
            ner_label = "LOC"
        classification = str(raw_ent.get("classification", "uncertain")).strip()

        canon = _canonical_name(raw_name)
        norm_key = _normalize(canon)

        if norm_key not in entity_name_lookup:
            entity_name_lookup[norm_key] = canon
            # Find mention spans in the chunk text
            mentions = _find_mentions(raw_name, chunk_text, sentences)
            entity_registry[canon] = {
                "name": canon,
                "canonical_name": canon,
                "ner_label": ner_label,
                "classification": classification,
                "mentions": [m.model_dump() for m in mentions],
                "doc_ids": [doc_id],
            }
        else:
            # Entity already seen — add any new mentions
            existing_canon = entity_name_lookup[norm_key]
            new_mentions = _find_mentions(raw_name, chunk_text, sentences)
            for m in new_mentions:
                entity_registry[existing_canon]["mentions"].append(m.model_dump())

    # --- Parse relations ---
    raw_relations = data.get("relations", [])
    if not isinstance(raw_relations, list):
        raw_relations = []

    validated: List[dict] = []
    for raw_rel in raw_relations:
        cleaned = _validate_relation(raw_rel, entity_name_lookup, chunk_text)
        if cleaned:
            validated.append(cleaned)

    return entity_registry, validated


def _find_mentions(
    entity_name: str,
    chunk_text: str,
    sentences: List[SentenceRecord],
) -> List[EntityMention]:
    """Find occurrences of entity_name in the chunk's sentences."""
    mentions: List[EntityMention] = []
    name_lower = entity_name.lower()
    seen: Set[Tuple[str, int, int]] = set()

    for sent in sentences:
        text = sent.text
        text_lower = text.lower()
        start = 0
        while True:
            idx = text_lower.find(name_lower, start)
            if idx < 0:
                break
            char_end = idx + len(entity_name)
            key = (sent.sentence_id, idx, char_end)
            if key not in seen:
                seen.add(key)
                context_start = max(0, idx - 20)
                context_end = min(len(text), char_end + 20)
                mentions.append(EntityMention(
                    sentence_id=sent.sentence_id,
                    char_start=idx,
                    char_end=char_end,
                    source_text=f"...{text[context_start:context_end]}...",
                ))
            start = idx + 1
    return mentions


def _validate_relation(
    raw: dict,
    entity_name_lookup: Dict[str, str],
    chunk_text: str,
) -> Optional[dict]:
    """Validate and normalise a single relation dict from Mistral."""
    if not isinstance(raw, dict):
        return None

    rel_type = str(raw.get("relation_type", "")).strip()
    if rel_type not in VALID_RELATION_TYPES:
        log.debug("Discarding unknown relation type: %r", rel_type)
        return None

    e1_raw = str(raw.get("entity_1", "")).strip()
    e1 = _resolve_entity(e1_raw, entity_name_lookup)
    if not e1:
        log.debug("Discarding: entity_1 %r not in entity set", e1_raw)
        return None

    raw_e2 = raw.get("entity_2")
    if raw_e2 and str(raw_e2).strip().lower() not in ("null", "none", ""):
        e2 = _resolve_entity(str(raw_e2).strip(), entity_name_lookup)
        if not e2:
            log.debug("Discarding: entity_2 %r not in entity set", raw_e2)
            return None
    else:
        e2 = None

    if e2 is None and rel_type not in NULLABLE_ENTITY2_TYPES:
        log.debug("Discarding: entity_2 is null for non-unary type %r", rel_type)
        return None

    if e1 == e2:
        log.debug("Discarding self-relation for %r", e1)
        return None

    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    evidence = str(raw.get("evidence", "")).strip()
    if len(evidence) > 15:
        snippet = evidence[:60].lower()
        if snippet not in chunk_text.lower():
            confidence *= 0.85

    dist_value: Optional[float] = None
    dist_unit: Optional[str] = None
    try:
        dv = raw.get("distance_value")
        if dv is not None:
            dist_value = float(dv)
    except (TypeError, ValueError):
        pass
    du = raw.get("distance_unit")
    if du in ("miles", "km"):
        dist_unit = du

    return {
        "entity_1": e1,
        "relation_type": rel_type,
        "entity_2": e2,
        "confidence": round(confidence, 3),
        "distance_value": dist_value,
        "distance_unit": dist_unit,
        "evidence": evidence[:500],
        "reasoning": str(raw.get("reasoning", "")).strip()[:500],
    }


def _resolve_entity(raw: str, lookup: Dict[str, str]) -> Optional[str]:
    """Map a raw entity name to a canonical name via the lookup."""
    if not raw:
        return None
    key = _normalize(raw)
    if key in lookup:
        return lookup[key]
    # Also try canonical form
    canon_key = _normalize(_canonical_name(raw))
    if canon_key in lookup:
        return lookup[canon_key]
    # Substring fallback
    for norm_key, canonical in lookup.items():
        if key in norm_key or norm_key in key:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Core extraction with retry
# ---------------------------------------------------------------------------

def _extract_chunk(
    client: OllamaClient,
    chunk_id: str,
    chunk_text: str,
    sentences: List[SentenceRecord],
    doc_id: str,
    max_retries: int = 3,
    temperature: float = 0.1,
) -> Tuple[Dict[str, dict], List[dict], Optional[str]]:
    """Extract entities + relations from a single chunk.

    Returns: (entity_registry, relations, error_message_or_None)
    """
    user_prompt = _build_user_prompt(chunk_text)

    for attempt in range(max_retries):
        system = _SYSTEM_PROMPT + (_FALLBACK_PROMPT_SUFFIX if attempt > 0 else "")
        retry_temperature = min(temperature + attempt * 0.05, 0.3)

        try:
            t0 = time.monotonic()
            raw_text = client.chat(system, user_prompt, temperature=retry_temperature)
            elapsed = time.monotonic() - t0

            entities, relations = _parse_model_response(
                raw_text, chunk_text, sentences, doc_id,
            )
            log.debug(
                "%s: %d entities, %d relations (%.1fs, attempt %d/%d)",
                chunk_id, len(entities), len(relations),
                elapsed, attempt + 1, max_retries,
            )
            return entities, relations, None

        except requests.exceptions.Timeout:
            msg = f"{chunk_id}: timeout (attempt {attempt + 1}/{max_retries})"
            log.warning(msg)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

        except requests.exceptions.ConnectionError as exc:
            msg = f"{chunk_id}: connection error — {exc}"
            log.error(msg)
            if attempt < max_retries - 1:
                time.sleep(5)

        except requests.exceptions.HTTPError as exc:
            msg = f"{chunk_id}: HTTP error — {exc}"
            log.error(msg)
            break

        except Exception as exc:
            msg = f"{chunk_id}: unexpected error — {exc}"
            log.exception(msg)
            if attempt < max_retries - 1:
                time.sleep(1)

    return {}, [], f"{chunk_id} failed after {max_retries} attempts"


# ---------------------------------------------------------------------------
# Phase entry point
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Stage 2: Joint Mistral NER + Relation Extraction ===")

    dd = data_dir(cfg)
    entities_path = dd / "entities.jsonl"
    entities_path.parent.mkdir(parents=True, exist_ok=True)

    rel_cfg = cfg.get("relations", {})
    m_cfg = rel_cfg.get("mistral", {})

    host         = m_cfg.get("host", "http://localhost:11434")
    model        = m_cfg.get("model", "mistral")
    temperature  = float(m_cfg.get("temperature", 0.1))
    timeout      = int(m_cfg.get("timeout_seconds", 120))
    max_retries  = int(m_cfg.get("max_retries", 3))
    chunk_size   = int(m_cfg.get("sentences_per_chunk", 6))
    overlap      = int(m_cfg.get("chunk_overlap", 2))
    min_conf     = float(m_cfg.get("min_confidence", 0.55))

    progress_path    = dd / rel_cfg.get("progress_file", "phase4_progress.json")
    checkpoint_path  = dd / rel_cfg.get("checkpoint_file", "phase4_checkpoint.json")

    if entities_path.exists() and not force:
        log.info("entities.jsonl exists — skipping (use --force to re-run).")
        return

    # ── Preflight: Ollama ─────────────────────────────────────────────────
    client = OllamaClient(host=host, model=model, timeout=timeout)

    if not client.is_available():
        raise RuntimeError(
            f"Ollama is not running at {host}.\n"
            "  -> Start it:  ollama serve\n"
            f"  -> Pull model: ollama pull {model}"
        )
    if not client.is_model_pulled():
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n"
            f"  -> Run: ollama pull {model}"
        )
    log.info("Ollama ready  model=%s  host=%s", model, host)

    # ── Load corpus + build chunks ────────────────────────────────────────
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

    all_chunks: List[Tuple[str, List[SentenceRecord], str]] = []  # (chunk_id, sentences, doc_id)
    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found: %s", cleaned_path)
            continue
        sentences = list(iter_jsonl(cleaned_path, model=SentenceRecord))
        for chunk_id, chunk_sents in _build_chunks(sentences, chunk_size=chunk_size, overlap=overlap):
            all_chunks.append((chunk_id, chunk_sents, doc_id))

    log.info("Total chunks to process: %d", len(all_chunks))

    # ── Checkpoint + progress setup ───────────────────────────────────────
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Cleared existing checkpoint (--force)")
    checkpoint = Checkpoint(checkpoint_path)

    resumed_counts: Dict[str, int] = defaultdict(int)
    for r in checkpoint.relations:
        resumed_counts[r.get("relation_type", "unknown")] += 1

    progress = ProgressTracker(progress_path)
    progress.initialize(
        doc_id=doc_ids[0] if doc_ids else "unknown",
        model=model,
        total_chunks=len(all_chunks),
        resumed_done=len(checkpoint.processed_ids),
        resumed_relations=len(checkpoint.relations),
        resumed_entities=len(checkpoint.entities),
        resumed_type_counts=dict(resumed_counts),
    )

    # ── Configuration ─────────────────────────────────────────────────────
    fictional_overrides = set(cfg.get("ner", {}).get("fictional_overrides", []))

    # ── Main extraction loop ──────────────────────────────────────────────
    for chunk_id, sentences, doc_id in all_chunks:
        if checkpoint.is_done(chunk_id):
            continue

        chunk_text = " ".join(s.text for s in sentences)
        preview = chunk_text[:250].replace("\n", " ")
        progress.chunk_started(preview)

        log.info("%-30s  sentences=%d", chunk_id, len(sentences))

        t0 = time.monotonic()
        new_entities, relations, error = _extract_chunk(
            client=client,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            sentences=sentences,
            doc_id=doc_id,
            max_retries=max_retries,
            temperature=temperature,
        )
        duration = time.monotonic() - t0

        # Apply confidence threshold to relations
        relations = [r for r in relations if r["confidence"] >= min_conf]

        # Tag relations with source chunk
        for r in relations:
            r["source_chunk_id"] = chunk_id

        checkpoint.save(chunk_id, new_entities, relations)
        progress.chunk_done(
            new_relations=relations,
            new_entity_count=len(new_entities),
            duration_s=duration,
            error=error,
        )

        if new_entities or relations:
            log.info(
                "  -> %d entities, %d relations  (%.1fs)",
                len(new_entities), len(relations), duration,
            )
        elif error:
            log.warning("  -> %s", error)

    # ── Materialise entities.jsonl ────────────────────────────────────────
    log.info("Materialising %d entities to entities.jsonl", len(checkpoint.entities))

    entities: List[Entity] = []
    for idx, (canon, info) in enumerate(sorted(checkpoint.entities.items())):
        mention_dicts = info.get("mentions", [])
        mentions = []
        seen_keys: Set[Tuple[str, int, int]] = set()
        for md in mention_dicts:
            key = (md["sentence_id"], md["char_start"], md["char_end"])
            if key not in seen_keys:
                seen_keys.add(key)
                mentions.append(EntityMention.model_validate(md))

        classification = _classify_entity(
            canon,
            info.get("classification", "uncertain"),
            fictional_overrides,
        )

        if not mentions:
            log.debug("Skipping entity '%s' — no mentions found in text", canon)
            continue

        entities.append(Entity(
            entity_id=f"e_{idx:04d}",
            name=info.get("name", canon),
            canonical_name=canon,
            type=classification,
            ner_label=info.get("ner_label", "LOC"),
            mentions=mentions,
            mention_count=len(mentions),
            doc_ids=sorted(set(info.get("doc_ids", []))),
        ))

    write_jsonl(entities_path, entities, overwrite=True)
    progress.finalize("complete")

    log.info(
        "Stage 2 complete: %d entities, %d relation candidates -> %s",
        len(entities), len(checkpoint.relations), entities_path,
    )
    log.info(
        "  Classification: %d real, %d fictional, %d uncertain",
        sum(1 for e in entities if e.type == "real"),
        sum(1 for e in entities if e.type == "fictional"),
        sum(1 for e in entities if e.type == "uncertain"),
    )
