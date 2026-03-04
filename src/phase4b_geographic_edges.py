"""Phase 4b — Geographic Knowledge Edges.

For all pairs of real (grounded) entities in the target dataset that have
known coordinates, compute cardinal direction (N/S, E/W) and great-circle
distance.  Then create simple anchor-radius constraints linking fictional
entities to co-occurring real entities.

New relations are appended to the existing relations.jsonl with
extraction_method = "geographic_knowledge" (real↔real) or
"anchor_geographic" (fictional→real anchor constraint).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.utils.io import read_jsonl, write_jsonl, data_dir
from src.utils.schemas import GroundedEntity, SpatialRelation
from src.utils.geo import haversine_km

log = logging.getLogger(__name__)

_RID_COUNTER = 0


def _next_rid(prefix: str = "rg") -> str:
    global _RID_COUNTER
    _RID_COUNTER += 1
    return f"{prefix}_{_RID_COUNTER:05d}"


def _cardinal_directions(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> Tuple[str, str]:
    """Return (ns_type, ew_type) describing entity-1 relative to entity-2."""
    ns = "north_of" if lat1 > lat2 else "south_of"
    ew = "east_of" if lon1 > lon2 else "west_of"
    return ns, ew


def _amazon_mention_count(entity: GroundedEntity, doc_id: str) -> int:
    return sum(1 for m in entity.mentions if m.sentence_id.startswith(doc_id))


def _make_relation(
    rid_prefix: str,
    rel_type: str,
    e1_name: str,
    e2_name: str,
    dist_km: float,
    weight: float,
    source_id: str,
    source_text: str,
    method: str,
) -> SpatialRelation:
    direction = rel_type.replace("_of", "") if rel_type in (
        "north_of", "south_of", "east_of", "west_of"
    ) else None
    return SpatialRelation(
        relation_id=_next_rid(rid_prefix),
        type=rel_type,
        entity_1=e1_name,
        entity_2=e2_name,
        direction=direction,
        distance_value=round(dist_km, 1),
        distance_unit="km",
        weight=round(weight, 3),
        uncertainty=round(1.0 - weight, 3),
        source_sentence_id=source_id,
        source_text=source_text[:200],
        extraction_method=method,
    )


# ---------------------------------------------------------------------------
# Core: real↔real geographic edges
# ---------------------------------------------------------------------------

def _build_real_edges(
    real_entities: List[GroundedEntity],
    weight: float,
) -> Tuple[List[SpatialRelation], Dict[str, List[Tuple[str, str, str, float]]]]:
    """Compute direction + distance for every pair of real entities.

    Returns the list of new SpatialRelation objects **and** a lookup dict
    mapping each real entity name to its geographic relationships with all
    others (used later for transitive propagation).
    """
    relations: List[SpatialRelation] = []
    geo_lookup: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)

    for e1, e2 in combinations(real_entities, 2):
        dist = haversine_km(e1.latitude, e1.longitude, e2.latitude, e2.longitude)
        ns, ew = _cardinal_directions(e1.latitude, e1.longitude, e2.latitude, e2.longitude)

        ns_rev = "south_of" if ns == "north_of" else "north_of"
        ew_rev = "west_of" if ew == "east_of" else "east_of"

        geo_lookup[e1.canonical_name].append((e2.canonical_name, ns, ew, dist))
        geo_lookup[e2.canonical_name].append((e1.canonical_name, ns_rev, ew_rev, dist))

        desc = f"{e1.canonical_name} → {e2.canonical_name}: {dist:.0f} km"

        relations.append(_make_relation(
            "rg", ns, e1.canonical_name, e2.canonical_name, dist,
            weight, "geographic_knowledge", f"{ns.replace('_', ' ')}: {desc}",
            "geographic_knowledge",
        ))
        relations.append(_make_relation(
            "rg", ew, e1.canonical_name, e2.canonical_name, dist,
            weight, "geographic_knowledge", f"{ew.replace('_', ' ')}: {desc}",
            "geographic_knowledge",
        ))
        relations.append(_make_relation(
            "rg", "distance_approx", e1.canonical_name, e2.canonical_name, dist,
            weight, "geographic_knowledge", f"distance: {desc}",
            "geographic_knowledge",
        ))

    return relations, geo_lookup


# ---------------------------------------------------------------------------
# Core: simple anchor constraints for fictional entities
# ---------------------------------------------------------------------------

def _build_anchor_edges(
    existing_relations: List[SpatialRelation],
    name_to_entity: Dict[str, GroundedEntity],
    real_name_set: Set[str],
    anchor_radius_km: float,
) -> List[SpatialRelation]:
    """For each fictional entity, create a single 'near' constraint to every
    real entity it co-occurs with or has any textual relation to, and a 'near'
    constraint between co-occurring fictional entity pairs."""

    fict_real_pairs: Set[Tuple[str, str]] = set()
    fict_fict_pairs: Set[Tuple[str, str]] = set()

    for rel in existing_relations:
        if rel.extraction_method in ("geographic_knowledge", "anchor_geographic"):
            continue
        e1 = name_to_entity.get(rel.entity_1)
        e2 = name_to_entity.get(rel.entity_2) if rel.entity_2 else None
        if not (e1 and e2):
            continue

        if e1.type == "fictional" and e2.type == "real" and e2.canonical_name in real_name_set:
            fict_real_pairs.add((e1.canonical_name, e2.canonical_name))
        if e2.type == "fictional" and e1.type == "real" and e1.canonical_name in real_name_set:
            fict_real_pairs.add((e2.canonical_name, e1.canonical_name))
        if e1.type == "fictional" and e2.type == "fictional":
            pair = tuple(sorted([e1.canonical_name, e2.canonical_name]))
            fict_fict_pairs.add(pair)

    relations: List[SpatialRelation] = []

    for fict_name, real_name in sorted(fict_real_pairs):
        relations.append(_make_relation(
            "ra", "near", fict_name, real_name, anchor_radius_km, 0.8,
            "anchor_geographic",
            f"{fict_name} near {real_name} (anchor radius {anchor_radius_km} km)",
            "anchor_geographic",
        ))

    for name_a, name_b in sorted(fict_fict_pairs):
        relations.append(_make_relation(
            "ra", "near", name_a, name_b, anchor_radius_km, 0.7,
            "anchor_geographic",
            f"{name_a} near {name_b} (co-occurring fictional entities)",
            "anchor_geographic",
        ))

    return relations


# ---------------------------------------------------------------------------
# Phase entry-point
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 4b: Geographic Knowledge Edges ===")

    dd = data_dir(cfg)
    grounded_path = dd / "grounded_entities.jsonl"
    relations_path = dd / "relations.jsonl"

    if not relations_path.exists():
        log.error("relations.jsonl not found — run phase 4 first.")
        return

    # Already enriched? Check unless forced.
    existing_relations: List[SpatialRelation] = read_jsonl(relations_path, model=SpatialRelation)
    has_geo = any(r.extraction_method == "geographic_knowledge" for r in existing_relations)
    if has_geo and not force:
        log.info("Geographic edges already present in relations.jsonl, skipping (use --force).")
        return

    # Strip any previous geographic/anchor/transitive edges when forcing
    if has_geo:
        existing_relations = [
            r for r in existing_relations
            if r.extraction_method not in ("geographic_knowledge", "transitive_geographic", "anchor_geographic")
        ]

    geo_cfg = cfg.get("relations", {}).get("geographic_edges", {})
    min_mentions: int = geo_cfg.get("min_mentions", 2)
    geo_weight: float = geo_cfg.get("weight", 0.95)
    anchor_radius_km: float = geo_cfg.get("anchor_radius_km", 50.0)

    doc_filter = cfg["corpus"].get("doc_filter", [])
    doc_id = doc_filter[0] if doc_filter else "amazon_madeira_rivers"

    entities: List[GroundedEntity] = read_jsonl(grounded_path, model=GroundedEntity)
    name_to_entity: Dict[str, GroundedEntity] = {e.canonical_name: e for e in entities}

    amazon_entities = [e for e in entities if doc_id in e.doc_ids]
    real_entities = [
        e for e in amazon_entities
        if e.type == "real"
        and e.latitude is not None
        and e.longitude is not None
        and _amazon_mention_count(e, doc_id) >= min_mentions
    ]
    real_name_set = {e.canonical_name for e in real_entities}

    log.info(
        "Filtered to %d real entities (>=%d mentions in '%s') and %d fictional",
        len(real_entities), min_mentions, doc_id,
        sum(1 for e in amazon_entities if e.type == "fictional"),
    )

    # Bump the global relation-ID counter past existing IDs
    global _RID_COUNTER
    for r in existing_relations:
        try:
            num = int(r.relation_id.split("_")[-1])
            _RID_COUNTER = max(_RID_COUNTER, num)
        except (IndexError, ValueError):
            pass

    # Step 1: real ↔ real geographic edges
    geo_relations, geo_lookup = _build_real_edges(real_entities, geo_weight)
    log.info("Generated %d geographic-knowledge edges between %d real entity pairs",
             len(geo_relations), len(geo_relations) // 3)

    # Step 2: anchor edges for fictional entities
    anchor_relations = _build_anchor_edges(
        existing_relations, name_to_entity, real_name_set, anchor_radius_km,
    )
    log.info("Generated %d anchor-geographic edges for fictional entities",
             len(anchor_relations))

    # Write augmented file
    all_relations = existing_relations + geo_relations + anchor_relations
    write_jsonl(relations_path, all_relations, overwrite=True)
    log.info(
        "Phase 4b complete — relations.jsonl: %d total "
        "(was %d, +%d geographic, +%d anchor)",
        len(all_relations), len(existing_relations),
        len(geo_relations), len(anchor_relations),
    )
