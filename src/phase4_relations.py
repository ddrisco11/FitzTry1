"""Phase 4 — Spatial Relation Extraction via Mistral 7B (Ollama).

High-accuracy spatial relation extraction using a local Mistral 7B model
served by Ollama. Features:
  - Structured JSON extraction with mandatory evidence grounding
  - Checkpoint/resume support (safe to interrupt mid-run)
  - Real-time progress file written after every chunk (feeds the dashboard)
  - Comprehensive entity validation and relation deduplication
  - Exponential-backoff retry with a tighter fallback prompt
  - Expanded relation taxonomy: within, contains, part_of, borders,
    on_shore_of, connected_via, distance_approx (plus all original types)

Prerequisites (one-time setup):
    brew install ollama
    ollama pull mistral
    ollama serve          # starts automatically on macOS after install

Run:
    python -m src.pipeline --config config.yaml --phase 4 --force

Dashboard (separate terminal):
    python -m src.dashboard
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

from src.utils.io import data_dir, iter_jsonl, read_json, read_jsonl, write_jsonl
from src.utils.schemas import GroundedEntity, SentenceRecord, SpatialRelation

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

# Relation types where entity_2 may legitimately be None
NULLABLE_ENTITY2_TYPES: Set[str] = {"on_coast"}

# Phase 2 NER routinely extracts these as LOC/GPE but they are directions,
# adjectives, or continents — useless for local spatial reasoning.
_ENTITY_BLOCKLIST: Set[str] = {
    "east", "west", "north", "south", "middle",
    "the east", "the west", "the north", "the south",
    "middle western", "middle west", "the middle west",
    "north america", "south america", "central america",
    "europe", "asia", "africa", "oceania", "antarctica",
    "the old world", "the new world",
}

_EVIDENCE_NULL_VALUES: Set[str] = {
    "", "none", "null", "n/a", "not stated", "not mentioned",
    "not found", "not specified", "not applicable",
}

_RELATION_ID_COUNTER = 0


def _next_rid() -> str:
    global _RELATION_ID_COUNTER
    _RELATION_ID_COUNTER += 1
    return f"r_{_RELATION_ID_COUNTER:04d}"


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
        """Send a chat request. Returns assistant response text. Raises on HTTP error."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",          # grammar-constrained JSON output
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 2048,
            },
        }
        resp = requests.post(self._chat_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise geographic relation extractor for literary texts.
Read the provided passage and identify spatial relationships between the listed geographic entities.

━━ RULES ━━
1. Only extract relations where BOTH entities are in the ENTITY LIST.
2. Only extract relations explicitly stated or unambiguously implied — never speculate.
3. Quality over quantity: when uncertain, omit.
4. Use entity names EXACTLY as they appear in the ENTITY LIST (same spelling, capitalisation).
5. The "evidence" field must be a verbatim or near-verbatim quote from the passage.
6. entity_2 may be null ONLY for the "on_coast" type when no body of water is named.

━━ RELATION TYPES ━━
  near           — close proximity (~10 km): "next to", "close to", "nearby", "just across"
  far            — distant (>50 km): "far from", "distant from"
  north_of       — entity_1 is geographically north of entity_2
  south_of       — entity_1 is geographically south of entity_2
  east_of        — entity_1 is geographically east of entity_2
  west_of        — entity_1 is geographically west of entity_2
  across         — entity_1 is directly across a body of water from entity_2
  on_coast       — entity_1 is on a shoreline (entity_2 = body of water, or null)
  within         — entity_1 is geographically inside entity_2 (village within a county)
  contains       — entity_1 geographically contains entity_2
  part_of        — entity_1 is administratively part of entity_2
  borders        — entity_1 and entity_2 share a geographic boundary
  on_shore_of    — entity_1 is on the bank/shore of a named body of water (entity_2)
  connected_via  — entity_1 and entity_2 are connected by a named road, bridge, or waterway
  distance_approx — entity_1 is approximately N units from entity_2

━━ OUTPUT FORMAT ━━
Return ONLY a valid JSON object. No markdown, no explanation, no extra text.

{
  "relations": [
    {
      "entity_1": "<exact name from entity list>",
      "relation_type": "<type from list above>",
      "entity_2": "<exact name from entity list, or null>",
      "confidence": <0.5–1.0>,
      "distance_value": <number or null>,
      "distance_unit": "miles" | "km" | null,
      "evidence": "<verbatim or near-verbatim quote from the passage>",
      "reasoning": "<one sentence: why this relation holds>"
    }
  ]
}

Confidence guide:
  0.90–1.00 → explicitly stated in the text
  0.70–0.89 → clearly and unambiguously implied
  0.50–0.69 → weakly implied (use sparingly)

If no spatial relations are found, return: {"relations": []}
"""

_FALLBACK_PROMPT_SUFFIX = (
    "\n\nCRITICAL: Return ONLY a valid JSON object with this exact structure: "
    '{"relations": [...]}. No markdown, no backticks, no explanation.'
)


def _build_user_prompt(chunk_text: str, entity_list: List[str]) -> str:
    entities_block = "\n".join(f"  • {e}" for e in sorted(entity_list))
    return f"ENTITY LIST:\n{entities_block}\n\nPASSAGE:\n{chunk_text}"


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------

def _build_chunks(
    sentences: List[SentenceRecord],
    chunk_size: int = 6,
    overlap: int = 2,
) -> List[Tuple[str, List[SentenceRecord]]]:
    """
    Group sentences into overlapping chunks of `chunk_size` with `overlap` sentences
    of context carried over between adjacent chunks.

    Returns list of (chunk_id, sentence_list) tuples.
    """
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
# Entity name resolution
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _resolve_entity(raw: str | None, lookup: Dict[str, str]) -> Optional[str]:
    """Map a raw entity name (from Mistral) to a canonical name. Returns None if unresolved."""
    if not raw:
        return None
    key = _normalize(raw)
    if key in lookup:
        return lookup[key]
    # Substring match as fallback
    for norm_key, canonical in lookup.items():
        if key in norm_key or norm_key in key:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

def _validate_relation(
    raw: dict,
    entity_lookup: Dict[str, str],
    chunk_text: str,
) -> Optional[dict]:
    """
    Validate and normalise a single relation dict from Mistral.
    Returns a cleaned dict ready for the checkpoint, or None to discard.
    """
    if not isinstance(raw, dict):
        return None

    rel_type = str(raw.get("relation_type", "")).strip()
    if rel_type not in VALID_RELATION_TYPES:
        log.debug("Discarding unknown relation type: %r", rel_type)
        return None

    e1 = _resolve_entity(raw.get("entity_1"), entity_lookup)
    if not e1:
        log.debug("Discarding: entity_1 %r not in entity set", raw.get("entity_1"))
        return None

    raw_e2 = raw.get("entity_2")
    if raw_e2 and str(raw_e2).strip().lower() not in ("null", "none", ""):
        e2 = _resolve_entity(raw_e2, entity_lookup)
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

    # Confidence
    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Penalise if evidence cannot be found in the chunk
    evidence = str(raw.get("evidence", "")).strip()
    if len(evidence) > 15:
        snippet = evidence[:60].lower()
        if snippet not in chunk_text.lower():
            confidence *= 0.85
            log.debug("Evidence not found verbatim — confidence penalised (%.2f)", confidence)

    # Distance fields
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


def _parse_model_response(
    raw_text: str,
    entity_lookup: Dict[str, str],
    chunk_text: str,
) -> List[dict]:
    """Parse Mistral's JSON response and return validated relations."""
    # Attempt direct parse
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to salvage a JSON object from the text
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            log.warning("No JSON object found in model response")
            return []
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            log.warning("JSON parse failed after salvage attempt: %s", exc)
            return []

    # Accept {"relations": [...]} or a bare list
    if isinstance(data, list):
        raw_relations = data
    elif isinstance(data, dict):
        raw_relations = data.get("relations", [])
    else:
        log.warning("Unexpected top-level JSON type: %s", type(data))
        return []

    if not isinstance(raw_relations, list):
        log.warning("'relations' key is not a list")
        return []

    validated: List[dict] = []
    for item in raw_relations:
        cleaned = _validate_relation(item, entity_lookup, chunk_text)
        if cleaned:
            validated.append(cleaned)

    return validated


# ---------------------------------------------------------------------------
# Checkpoint (persist processed chunks + relations to disk)
# ---------------------------------------------------------------------------

class Checkpoint:
    """Persist processed chunk IDs and accumulated relations for resume support."""

    def __init__(self, path: Path):
        self.path = path
        self.processed_ids: Set[str] = set()
        self.relations: List[dict] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
            self.processed_ids = set(data.get("processed_chunk_ids", []))
            self.relations = data.get("relations", [])
            log.info(
                "Checkpoint loaded — %d chunks already done, %d relations accumulated",
                len(self.processed_ids), len(self.relations),
            )
        except Exception as exc:
            log.warning("Checkpoint load failed (%s) — starting fresh", exc)

    def save(self, chunk_id: str, new_relations: List[dict]) -> None:
        """Atomically persist the checkpoint after processing one chunk."""
        self.processed_ids.add(chunk_id)
        self.relations.extend(new_relations)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "processed_chunk_ids": list(self.processed_ids),
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
    """Writes a progress JSON file after every chunk for the dashboard to read."""

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
        resumed_type_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        self._state = {
            "status": "running",
            "doc_id": doc_id,
            "model": model,
            "total_chunks": total_chunks,
            "processed_chunks": resumed_done,
            "error_chunks": 0,
            "relations_extracted": resumed_relations,
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
        counts = self._state["relation_type_counts"]
        for r in new_relations:
            rt = r.get("relation_type", "unknown")
            counts[rt] = counts.get(rt, 0) + 1

        self._state["chunk_durations_s"].append(round(duration_s, 2))
        self._state["chunk_durations_s"] = self._state["chunk_durations_s"][-100:]

        # Live feed: last 8 relations (most recent first in display)
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
# Core extraction with retry
# ---------------------------------------------------------------------------

def _extract_chunk(
    client: OllamaClient,
    chunk_id: str,
    chunk_text: str,
    entity_lookup: Dict[str, str],
    entity_names: List[str],
    max_retries: int = 3,
    temperature: float = 0.1,
) -> Tuple[List[dict], Optional[str]]:
    """
    Extract relations from a single chunk with exponential-backoff retries.
    Returns (relations, error_message_or_None).
    """
    user_prompt = _build_user_prompt(chunk_text, entity_names)

    for attempt in range(max_retries):
        system = _SYSTEM_PROMPT + (_FALLBACK_PROMPT_SUFFIX if attempt > 0 else "")
        retry_temperature = min(temperature + attempt * 0.05, 0.3)

        try:
            t0 = time.monotonic()
            raw_text = client.chat(system, user_prompt, temperature=retry_temperature)
            elapsed = time.monotonic() - t0

            relations = _parse_model_response(raw_text, entity_lookup, chunk_text)
            log.debug(
                "%s: %d relations extracted (%.1fs, attempt %d/%d)",
                chunk_id, len(relations), elapsed, attempt + 1, max_retries,
            )
            return relations, None

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
            break  # not retryable

        except Exception as exc:
            msg = f"{chunk_id}: unexpected error — {exc}"
            log.exception(msg)
            if attempt < max_retries - 1:
                time.sleep(1)

    return [], f"{chunk_id} failed after {max_retries} attempts"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(relations: List[dict]) -> List[dict]:
    """
    Keep at most one relation per (entity_1, relation_type, entity_2) triple.
    When duplicates exist, keep the one with the highest confidence.
    """
    best: Dict[Tuple[str, str, str], dict] = {}
    for r in relations:
        key = (
            _normalize(r["entity_1"]),
            r["relation_type"],
            _normalize(r.get("entity_2") or ""),
        )
        if key not in best or r["confidence"] > best[key]["confidence"]:
            best[key] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# Entity quality filter
# ---------------------------------------------------------------------------

def _is_quality_entity(entity: GroundedEntity, fictional_overrides: Set[str]) -> bool:
    """
    Return True if this entity is worth passing to Mistral.

    Rejects:
      - Cardinal directions and geographic adjectives ("East", "Middle Western")
      - Continental/overly-broad terms ("North America", "Europe")
      - Names shorter than 4 characters
      - Real entities that failed geocoding (lat/lon is None)
      - Any entity with only 1 mention (too ambiguous to reason about)
    """
    name = entity.canonical_name
    name_lower = name.lower().strip()

    if name_lower in _ENTITY_BLOCKLIST:
        return False

    if len(name.strip()) < 4:
        return False

    # Fictional overrides (config list) always pass with ≥1 mention
    if name in fictional_overrides or entity.type == "fictional":
        return entity.mention_count >= 1

    # Real entities must have been geocoded AND have ≥2 mentions
    if entity.latitude is None or entity.longitude is None:
        return False

    return entity.mention_count >= 2


# ---------------------------------------------------------------------------
# Phase entry point
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 4: Spatial Relation Extraction (Mistral/Ollama) ===")

    dd = data_dir(cfg)
    relations_path = dd / "relations.jsonl"
    relations_path.parent.mkdir(parents=True, exist_ok=True)

    if relations_path.exists() and not force:
        log.info("relations.jsonl exists — skipping (use --force to re-run).")
        return

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

    # ── Preflight: Ollama ─────────────────────────────────────────────────
    client = OllamaClient(host=host, model=model, timeout=timeout)

    if not client.is_available():
        raise RuntimeError(
            f"Ollama is not running at {host}.\n"
            "  → Start it:  ollama serve\n"
            f"  → Pull model: ollama pull {model}"
        )

    if not client.is_model_pulled():
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n"
            f"  → Run: ollama pull {model}"
        )

    log.info("Ollama ready  model=%s  host=%s", model, host)

    # ── Load entities ─────────────────────────────────────────────────────
    grounded_path = dd / "grounded_entities.jsonl"
    grounded: List[GroundedEntity] = read_jsonl(grounded_path, model=GroundedEntity)
    entity_lookup: Dict[str, str] = {_normalize(e.canonical_name): e.canonical_name for e in grounded}
    all_entity_names: List[str] = [e.canonical_name for e in grounded]
    log.info("Loaded %d grounded entities", len(grounded))

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

    # Build per-sentence entity map and all chunks
    sent_to_entities: Dict[str, List[str]] = defaultdict(list)
    all_chunks: List[Tuple[str, List[SentenceRecord]]] = []

    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found: %s", cleaned_path)
            continue

        sentences = list(iter_jsonl(cleaned_path, model=SentenceRecord))
        all_chunks.extend(_build_chunks(sentences, chunk_size=chunk_size, overlap=overlap))

        for entity in grounded:
            for mention in entity.mentions:
                if mention.sentence_id.startswith(doc_id):
                    sent_to_entities[mention.sentence_id].append(entity.canonical_name)

    log.info("Total chunks to process: %d", len(all_chunks))

    # ── Checkpoint + progress setup ───────────────────────────────────────
    checkpoint = Checkpoint(checkpoint_path)

    # Rebuild type counts from checkpointed relations for dashboard accuracy
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
        resumed_type_counts=dict(resumed_counts),
    )

    # ── Main extraction loop ──────────────────────────────────────────────
    for chunk_id, sentences in all_chunks:
        if checkpoint.is_done(chunk_id):
            continue

        # Collect entities appearing in this chunk
        entities_in_chunk: Set[str] = set()
        for sent in sentences:
            for ent in sent_to_entities.get(sent.sentence_id, []):
                entities_in_chunk.add(ent)

        chunk_text = " ".join(s.text for s in sentences)
        preview    = chunk_text[:250].replace("\n", " ")

        progress.chunk_started(preview)

        # Skip chunks with no entities (nothing to extract)
        if not entities_in_chunk:
            checkpoint.save(chunk_id, [])
            progress.chunk_done([], duration_s=0.0)
            continue

        log.info(
            "%-30s  entities=%d  sentences=%d",
            chunk_id, len(entities_in_chunk), len(sentences),
        )

        t0 = time.monotonic()
        relations, error = _extract_chunk(
            client=client,
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            entity_lookup=entity_lookup,
            entity_names=sorted(entities_in_chunk),
            max_retries=max_retries,
            temperature=temperature,
        )
        duration = time.monotonic() - t0

        # Apply confidence threshold
        relations = [r for r in relations if r["confidence"] >= min_conf]

        # Tag with source chunk for traceability
        for r in relations:
            r["source_chunk_id"] = chunk_id

        checkpoint.save(chunk_id, relations)
        progress.chunk_done(new_relations=relations, duration_s=duration, error=error)

        if relations:
            log.info("  → %d relations  (%.1fs)", len(relations), duration)
        elif error:
            log.warning("  → %s", error)

    # ── Finalise: deduplicate and write output ────────────────────────────
    all_raw = _deduplicate(checkpoint.relations)
    log.info(
        "Deduplication: %d → %d relations",
        len(checkpoint.relations), len(all_raw),
    )

    spatial_relations: List[SpatialRelation] = []
    for r in all_raw:
        spatial_relations.append(
            SpatialRelation(
                relation_id=_next_rid(),
                type=r["relation_type"],
                entity_1=r["entity_1"],
                entity_2=r.get("entity_2"),
                direction=None,
                distance_value=r.get("distance_value"),
                distance_unit=r.get("distance_unit"),
                weight=round(r.get("confidence", 0.7), 3),
                uncertainty=round(1.0 - r.get("confidence", 0.7), 3),
                source_sentence_id=r.get("source_chunk_id", ""),
                source_text=r.get("evidence", "")[:200],
                extraction_method="mistral",
            )
        )

    write_jsonl(relations_path, spatial_relations, overwrite=True)
    progress.finalize("complete")

    # Summary breakdown
    type_counts: Dict[str, int] = defaultdict(int)
    for r in spatial_relations:
        type_counts[r.type] += 1
    log.info("Phase 4 complete — %d relations → %s", len(spatial_relations), relations_path)
    for rt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info("  %-20s %d", rt, cnt)
