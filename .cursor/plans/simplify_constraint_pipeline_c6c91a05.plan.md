---
name: Simplify constraint pipeline
overview: Replace the transitive propagation engine (phase4b) with simple anchor-radius constraints, fix the entity curation logic (phase5) so entities like East Egg survive, and fix the phase2 NER bug.
todos:
  - id: rewrite-phase4b
    content: "Rewrite phase4b: remove _build_transitive_edges, add _build_anchor_edges (simple near-anchor constraints for fictional entities)"
    status: pending
  - id: fix-phase5-curation
    content: "Fix phase5: count all constraints toward min_constraints_per_latent, lower default from 3 to 1"
    status: pending
  - id: update-configs
    content: Update config.yaml, config_gatsby.yaml, config_amazon.yaml with new parameters
    status: pending
  - id: verify-no-regressions
    content: "Dry-run: verify West Egg and East Egg both survive with consistent constraints"
    status: pending
isProject: false
---

# Simplify the Constraint Pipeline

The current pipeline over-engineers constraints: phase4b copies every real entity's distances to fictional entities transitively, creating dozens of contradictory geometric constraints. The fix is to replace that with simple "near anchor" constraints, and to relax the curation threshold so entities like East Egg don't get dropped.

## Changes

### 1. Fix phase2 NER filter (already applied)

[src/phase2_ner.py](src/phase2_ner.py) `_is_non_geographic` — check original casing, not lowered name. This was already applied in the previous conversation turn.

### 2. Rewrite phase4b transitive propagation

[src/phase4b_geographic_edges.py](src/phase4b_geographic_edges.py)

**Keep**: `_build_real_edges()` (lines 84-125) — real-to-real geographic edges are ground truth and stay.

**Remove**: `_build_transitive_edges()` (lines 132-211) — this is the source of contradictory constraints.

**Replace with** `_build_anchor_edges()` — a much simpler function that:

- For each fictional entity F, finds all real entities R that F co-occurs with (or has any textual relation to)
- Creates ONE `near` constraint per (F, R) pair with a configurable `anchor_radius_km` (default ~50 km)
- For each pair of fictional entities that co-occur, creates ONE `near` constraint between them

This means West Egg gets:

- "near New York" (~50 km radius)
- "near Long Island" (~50 km radius)

And East Egg gets:

- "near West Egg" (from co-occurrence)

No cascading contradictions. Far fewer constraints. Each one is defensible.

**Config change**: Replace `transitive_decay` and `max_transitive_targets` in the YAML with a single `anchor_radius_km` parameter.

### 3. Fix phase5 entity curation

[src/phase5_constraints.py](src/phase5_constraints.py) lines 184-218

Two changes:

- **Lower `min_constraints_per_latent` default** from 3 to 1 — even a single grounded constraint is better than dropping an entity. Config files will also be updated.
- **Count all constraint types** toward the minimum, not just non-co_occurrence. The reason co_occurrence was excluded was that it's "weak" — but a co_occurrence with a real entity is still a spatial signal, and excluding it is what killed East Egg. With the simplified constraint model, we want to be inclusive: entities with wide posteriors are more honest than dropped entities.

Specifically, change lines 188-194 from:

```python
for c in constraints:
    for eid in c.entities:
        if eid in latent_constraint_count:
            if c.type != "co_occurrence":
                latent_constraint_count[eid] += 1
```

to:

```python
for c in constraints:
    for eid in c.entities:
        if eid in latent_constraint_count:
            latent_constraint_count[eid] += 1
```

### 4. Update config files

- [config.yaml](config.yaml): set `min_constraints_per_latent: 1`
- [config_gatsby.yaml](config_gatsby.yaml): set `min_constraints_per_latent: 1`
- [config_amazon.yaml](config_amazon.yaml): replace `transitive_decay` / `max_transitive_targets` with `anchor_radius_km: 50`, set `min_constraints_per_latent: 1`

## What stays unchanged

- **Phases 1, 3, 4, 6, 7, 8** — corpus prep, grounding, text relation extraction, inference, convergence, visualization are all fine as-is. The inference engine (phase6) will naturally produce tighter posteriors because the constraints are now consistent.
- **Schemas** ([src/utils/schemas.py](src/utils/schemas.py)) — no changes needed
- **Geo utilities** ([src/utils/geo.py](src/utils/geo.py)) — no changes needed

## Expected effect

- **West Egg**: fewer constraints but all consistent; should converge to a tighter posterior near Long Island / north shore NYC area
- **East Egg**: survives curation via its co-occurrence with West Egg; gets placed near West Egg with a wide posterior (honest uncertainty)
- **Amazon text**: fictional entities get simple anchor constraints to nearby real places instead of contradictory transitive chains
- **Constraint satisfaction**: should jump from ~54% to much higher, since the constraints no longer contradict each other

