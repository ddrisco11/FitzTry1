"""Phase 5 — Formal Spatial Constraint Model.

Converts extracted spatial relations into a formal constraint model:
- Real entities → fixed coordinates (km) in local planar system.
- Fictional entities → latent variables (x, y) to be inferred.
- Each relation → a penalty function + weight.

Output: data/constraints.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.io import read_jsonl, write_json, data_dir
from src.utils.schemas import (
    ConstraintModel,
    ConstraintSpec,
    CoordinateSystem,
    FixedEntitySpec,
    GroundedEntity,
    LatentEntitySpec,
    SpatialRelation,
)
from src.utils.geo import latlon_to_km

log = logging.getLogger(__name__)


def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 5: Formal Spatial Constraint Model ===")

    dd = data_dir(cfg)
    grounded_path = dd / "grounded_entities.jsonl"
    relations_path = dd / "relations.jsonl"
    constraints_path = dd / "constraints.json"
    constraints_path.parent.mkdir(parents=True, exist_ok=True)

    if constraints_path.exists() and not force:
        log.info("constraints.json exists, skipping (use --force to overwrite).")
        return

    c_cfg = cfg.get("constraints", {})
    origin_lat: float = c_cfg.get("projection_origin_lat", 40.7128)
    origin_lon: float = c_cfg.get("projection_origin_lon", -74.0060)
    epsilon: float = c_cfg.get("epsilon_direction_km", 1.0)
    d_near: float = c_cfg.get("d_near_km", 10.0)
    d_far: float = c_cfg.get("d_far_km", 50.0)
    sigma: float = c_cfg.get("sigma_distance_km", 5.0)
    co_weight: float = c_cfg.get("co_occurrence_weight", 0.1)

    grounded_entities: List[GroundedEntity] = read_jsonl(grounded_path, model=GroundedEntity)
    relations: List[SpatialRelation] = read_jsonl(relations_path, model=SpatialRelation)

    log.info("Loaded %d grounded entities and %d relations", len(grounded_entities), len(relations))

    # --- Build entity maps ---
    name_to_entity: Dict[str, GroundedEntity] = {e.canonical_name: e for e in grounded_entities}
    id_to_entity: Dict[str, GroundedEntity] = {e.entity_id: e for e in grounded_entities}

    # --- Convert real coords to local km ---
    fixed: List[FixedEntitySpec] = []
    latent: List[LatentEntitySpec] = []
    entity_km: Dict[str, Tuple[float, float]] = {}  # name → (x_km, y_km)

    for entity in grounded_entities:
        if entity.type == "real" and entity.latitude is not None and entity.longitude is not None:
            x, y = latlon_to_km(entity.latitude, entity.longitude, origin_lat, origin_lon)
            entity_km[entity.canonical_name] = (x, y)
            fixed.append(FixedEntitySpec(entity_id=entity.entity_id, name=entity.canonical_name, x=round(x, 4), y=round(y, 4)))
        else:
            latent.append(LatentEntitySpec(entity_id=entity.entity_id, name=entity.canonical_name))

    log.info("Fixed entities: %d, Latent entities: %d", len(fixed), len(latent))

    # --- Build constraint specs ---
    constraints: List[ConstraintSpec] = []
    constraint_idx = 0

    for rel in relations:
        # Resolve entity IDs
        e1 = name_to_entity.get(rel.entity_1)
        e2 = name_to_entity.get(rel.entity_2) if rel.entity_2 else None

        if e1 is None:
            log.debug("Entity not found in registry: %s", rel.entity_1)
            continue
        if rel.entity_2 and e2 is None:
            log.debug("Entity not found in registry: %s", rel.entity_2)
            continue

        # Skip self-constraints
        if e2 and e1.entity_id == e2.entity_id:
            log.debug("Skipping self-constraint: %s ↔ %s", e1.canonical_name, e2.canonical_name if e2 else "-")
            continue

        # Skip real↔real geographic-knowledge constraints: both entities are
        # fixed so the constraint is always trivially satisfied and would only
        # bloat the model / slow inference.
        if rel.extraction_method == "geographic_knowledge" and e2:
            if (e1.type == "real" and e1.latitude is not None
                    and e2.type == "real" and e2.latitude is not None):
                continue

        # Skip in_region constraints where both entities are real but very far apart
        # (these are usually extraction errors where the "region" entity is misidentified)
        if rel.type == "in_region" and e2 is not None:
            if e1.type == "real" and e2.type == "real":
                from src.utils.geo import haversine_km as _hkm
                if e1.latitude and e2.latitude:
                    dist = _hkm(e1.latitude, e1.longitude, e2.latitude, e2.longitude)
                    if dist > 500:
                        log.debug("Skipping implausible in_region (dist=%.0f km): %s in %s", dist, e1.canonical_name, e2.canonical_name)
                        continue
            elif e1.type == "fictional" and e2.type == "real":
                # Fictional entity placed "in" a real region far from story setting
                # Keep only if the real entity is within 500 km of origin
                if e2.canonical_name in entity_km:
                    cx, cy = entity_km[e2.canonical_name]
                    import math as _m
                    if _m.sqrt(cx**2 + cy**2) > 500:
                        log.debug("Skipping in_region — region entity far from story: %s in %s", e1.canonical_name, e2.canonical_name)
                        continue

        entity_ids = [e1.entity_id]
        if e2:
            entity_ids.append(e2.entity_id)

        # Build constraint params based on relation type
        params: dict = {}
        ctype = rel.type

        if ctype == "north_of":
            params = {"epsilon_km": epsilon}
        elif ctype == "south_of":
            params = {"epsilon_km": epsilon}
        elif ctype == "east_of":
            params = {"epsilon_km": epsilon}
        elif ctype == "west_of":
            params = {"epsilon_km": epsilon}
        elif ctype == "near":
            params = {"d_near_km": d_near}
        elif ctype == "far":
            params = {"d_far_km": d_far}
        elif ctype == "distance_approx":
            # Convert distance to km if in miles
            val = rel.distance_value or 10.0
            unit = rel.distance_unit or "km"
            if unit == "miles":
                val = val * 1.60934
            params = {"target_d_km": round(val, 2), "sigma_km": sigma}
        elif ctype == "across":
            params = {"d_near_km": d_near * 1.5, "epsilon_km": epsilon}
        elif ctype == "on_coast":
            params = {"coast_x_km": 0.0}  # placeholder; visualization handles
        elif ctype == "in_region":
            if e2 and e2.canonical_name in entity_km:
                cx, cy = entity_km[e2.canonical_name]
                params = {"centroid_x_km": cx, "centroid_y_km": cy, "radius_km": 20.0}
            else:
                log.debug("Skipping in_region without grounded centroid: %s", e1.canonical_name)
                continue
        elif ctype == "co_occurrence":
            params = {"d_near_km": d_near}
        else:
            params = {}

        constraint_idx += 1
        constraints.append(
            ConstraintSpec(
                constraint_id=f"c_{constraint_idx:04d}",
                type=ctype,
                entities=entity_ids,
                params=params,
                weight=rel.weight if ctype != "co_occurrence" else co_weight,
                source_relation_id=rel.relation_id,
            )
        )

    log.info("Built %d constraint specs (pre-curation)", len(constraints))

    # --- Curate latent entities: drop those with too few constraints ---
    # Only count constraint types that the inference engine can evaluate;
    # unimplemented types (e.g. on_coast) don't actually constrain the entity.
    _INFERABLE_TYPES = {
        "north_of", "south_of", "east_of", "west_of",
        "near", "far", "distance_approx", "across", "in_region", "co_occurrence",
    }
    min_constraints = c_cfg.get("min_constraints_per_latent", 1)
    latent_ids = {le.entity_id for le in latent}

    latent_constraint_count: Dict[str, int] = {eid: 0 for eid in latent_ids}
    for c in constraints:
        if c.type not in _INFERABLE_TYPES:
            continue
        for eid in c.entities:
            if eid in latent_constraint_count:
                latent_constraint_count[eid] += 1

    underconstrained = {
        eid for eid, cnt in latent_constraint_count.items()
        if cnt < min_constraints
    }

    if underconstrained:
        latent_name_map = {le.entity_id: le.name for le in latent}
        for eid in underconstrained:
            log.info(
                "Dropping underconstrained latent entity '%s' (%d constraints < %d minimum)",
                latent_name_map.get(eid, eid),
                latent_constraint_count[eid],
                min_constraints,
            )
        latent = [le for le in latent if le.entity_id not in underconstrained]
        constraints = [
            c for c in constraints
            if not any(eid in underconstrained for eid in c.entities)
        ]
        log.info(
            "Curated: removed %d underconstrained latent entities, %d constraints remain",
            len(underconstrained), len(constraints),
        )

    log.info("Final model: %d fixed, %d latent, %d constraints", len(fixed), len(latent), len(constraints))

    model = ConstraintModel(
        fixed_entities=fixed,
        latent_entities=latent,
        constraints=constraints,
        coordinate_system=CoordinateSystem(
            projection="equirectangular",
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            units="km",
        ),
    )

    write_json(constraints_path, model, overwrite=True)
    log.info("Phase 5 complete. Written to %s", constraints_path)
