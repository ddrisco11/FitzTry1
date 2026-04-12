"""Stage 5 (Strategy B) — Graph Building.

Runs AFTER coref (stage 3) and geographic grounding (stage 4) to produce
the final relations.jsonl.  Reads the relation *candidates* checkpointed
by the joint Mistral extraction stage, applies coref-resolved canonical
names, uses grounding information to filter/weight edges, deduplicates,
and writes the committed spatial-relation graph consumed by all downstream
stages (geographic edges, constraints, inference, visualisation).

Inputs:
  - data/grounded_entities.jsonl   (coreffed + grounded entities)
  - data/phase4_checkpoint.json    (relation candidates from Mistral)

Output:
  - data/relations.jsonl           (final SpatialRelation list)
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.utils.io import data_dir, read_jsonl, write_jsonl
from src.utils.schemas import GroundedEntity, SpatialRelation

log = logging.getLogger(__name__)

_RELATION_ID_COUNTER = 0


def _next_rid() -> str:
    global _RELATION_ID_COUNTER
    _RELATION_ID_COUNTER += 1
    return f"r_{_RELATION_ID_COUNTER:04d}"


def _normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


# ---------------------------------------------------------------------------
# Entity quality filter (uses grounded data, no static blocklist)
# ---------------------------------------------------------------------------

def _is_quality_entity(entity: GroundedEntity, fictional_overrides: Set[str]) -> bool:
    """Return True if this entity should participate in the relation graph.

    Uses grounding status and mention count — no static blocklist.
    """
    name = entity.canonical_name

    # Very short names are likely noise
    if len(name.strip()) < 3:
        return False

    # Fictional overrides and fictional entities always pass with >= 1 mention
    if name in fictional_overrides or entity.type == "fictional":
        return entity.mention_count >= 1

    # Real entities must have been geocoded AND have >= 2 mentions
    if entity.latitude is None or entity.longitude is None:
        return False

    return entity.mention_count >= 2


# ---------------------------------------------------------------------------
# Coref-aware name resolution
# ---------------------------------------------------------------------------

def _build_grounded_lookup(
    entities: List[GroundedEntity],
) -> Dict[str, str]:
    """Build normalized-name -> canonical_name lookup from grounded entities."""
    lookup: Dict[str, str] = {}
    for e in entities:
        lookup[_normalize(e.canonical_name)] = e.canonical_name
        # Also index by raw name if different
        if e.name != e.canonical_name:
            lookup[_normalize(e.name)] = e.canonical_name
    return lookup


def _resolve_to_grounded(
    raw_name: str,
    lookup: Dict[str, str],
) -> Optional[str]:
    """Resolve a relation endpoint name to a grounded canonical name."""
    if not raw_name:
        return None
    key = _normalize(raw_name)
    if key in lookup:
        return lookup[key]
    # Substring fallback
    for norm_key, canonical in lookup.items():
        if key in norm_key or norm_key in key:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(relations: List[dict]) -> List[dict]:
    """Keep at most one relation per (entity_1, type, entity_2) triple,
    preferring the highest confidence."""
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
# Phase entry point
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Stage 5: Graph Building ===")

    dd = data_dir(cfg)
    relations_path = dd / "relations.jsonl"
    relations_path.parent.mkdir(parents=True, exist_ok=True)

    if relations_path.exists() and not force:
        log.info("relations.jsonl exists — skipping (use --force to re-run).")
        return

    # ── Load grounded entities ────────────────────────────────────────────
    grounded_path = dd / "grounded_entities.jsonl"
    grounded: List[GroundedEntity] = read_jsonl(grounded_path, model=GroundedEntity)
    log.info("Loaded %d grounded entities", len(grounded))

    name_lookup = _build_grounded_lookup(grounded)
    name_to_entity: Dict[str, GroundedEntity] = {
        e.canonical_name: e for e in grounded
    }

    # Build quality entity set
    fictional_overrides = set(cfg.get("ner", {}).get("fictional_overrides", []))
    quality_names: Set[str] = {
        e.canonical_name
        for e in grounded
        if _is_quality_entity(e, fictional_overrides)
    }
    log.info("Quality entities: %d / %d", len(quality_names), len(grounded))

    # ── Load relation candidates from Mistral checkpoint ──────────────────
    rel_cfg = cfg.get("relations", {})
    checkpoint_path = dd / rel_cfg.get("checkpoint_file", "phase4_checkpoint.json")

    if not checkpoint_path.exists():
        log.error(
            "Checkpoint file not found: %s — run Mistral extraction first.",
            checkpoint_path,
        )
        return

    checkpoint_data = json.loads(checkpoint_path.read_text("utf-8"))
    raw_relations: List[dict] = checkpoint_data.get("relations", [])
    log.info("Loaded %d relation candidates from checkpoint", len(raw_relations))

    # ── Resolve endpoints to grounded canonical names ─────────────────────
    resolved: List[dict] = []
    dropped_unresolved = 0
    dropped_quality = 0

    for r in raw_relations:
        e1 = _resolve_to_grounded(r.get("entity_1", ""), name_lookup)
        if not e1:
            dropped_unresolved += 1
            continue

        raw_e2 = r.get("entity_2")
        if raw_e2:
            e2 = _resolve_to_grounded(raw_e2, name_lookup)
            if not e2:
                dropped_unresolved += 1
                continue
        else:
            e2 = None

        # Quality filter: both endpoints must be quality entities
        if e1 not in quality_names:
            dropped_quality += 1
            continue
        if e2 and e2 not in quality_names:
            dropped_quality += 1
            continue

        # Skip self-relations (can arise from coref merging)
        if e1 == e2:
            continue

        resolved.append({
            "entity_1": e1,
            "relation_type": r["relation_type"],
            "entity_2": e2,
            "confidence": r.get("confidence", 0.7),
            "distance_value": r.get("distance_value"),
            "distance_unit": r.get("distance_unit"),
            "evidence": r.get("evidence", ""),
            "reasoning": r.get("reasoning", ""),
            "source_chunk_id": r.get("source_chunk_id", ""),
        })

    log.info(
        "Endpoint resolution: %d resolved, %d dropped (unresolved), %d dropped (quality)",
        len(resolved), dropped_unresolved, dropped_quality,
    )

    # ── Deduplicate ───────────────────────────────────────────────────────
    deduped = _deduplicate(resolved)
    log.info("Deduplication: %d -> %d relations", len(resolved), len(deduped))

    # ── Write final relations.jsonl ───────────────────────────────────────
    global _RELATION_ID_COUNTER
    _RELATION_ID_COUNTER = 0

    spatial_relations: List[SpatialRelation] = []
    for r in deduped:
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

    # Summary
    type_counts: Dict[str, int] = defaultdict(int)
    for r in spatial_relations:
        type_counts[r.type] += 1
    log.info(
        "Stage 5 complete — %d relations written to %s",
        len(spatial_relations), relations_path,
    )
    for rt, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        log.info("  %-20s %d", rt, cnt)
