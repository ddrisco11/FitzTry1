# Probabilistic Reconstruction of Literary Geography

A reproducible computational pipeline that extracts geographic entities and spatial relationships from F. Scott Fitzgerald's literary corpus, constructs a probabilistic spatial model, and produces visualizations of inferred fictional geography.

## Overview

The output is **not** a single deterministic map — it is a **probability distribution over spatial configurations** consistent with textual evidence. Fictional places like East Egg and West Egg from *The Great Gatsby* are treated as latent variables, their positions inferred from constraints extracted from the text.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. Run full pipeline
python -m src.pipeline --config config.yaml

# 4. Run a specific phase
python -m src.pipeline --config config.yaml --phase 1

# 5. Force re-run even if outputs exist
python -m src.pipeline --config config.yaml --force
```

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `phase1_corpus_prep.py` | Download & clean corpus, sentence segmentation |
| 2 | `phase2_ner.py` | Named entity recognition, real/fictional classification |
| 3 | `phase3_grounding.py` | Geocode real entities via Nominatim |
| 4 | `phase4_relations.py` | Extract spatial relations (pattern + HuggingFace) |
| 5 | `phase5_constraints.py` | Build formal constraint model |
| 6 | `phase6_inference.py` | MCMC sampling (emcee ensemble sampler) |
| 7 | `phase7_convergence.py` | Convergence diagnostics & posterior summaries |
| 8 | `phase8_visualization.py` | Maps, heatmaps, constraint graph, ensemble plots |

## Key Outputs

- `data/entities.jsonl` — All geographic entities with real/fictional classification
- `data/grounded_entities.jsonl` — Real entities with lat/lon coordinates
- `data/relations.jsonl` — Extracted spatial relations with uncertainty scores
- `data/constraints.json` — Formal constraint model
- `data/samples/*.jsonl` — MCMC samples of fictional entity positions
- `data/convergence/*.json` — Posterior summaries (mean, std, credible regions)
- `visualizations/constraint_graph.html` — Interactive constraint network
- `visualizations/overlay_maps/full_map.html` — Folium map with real + inferred places
- `visualizations/heatmaps/*.png` — Posterior distribution heatmaps
- `visualizations/ensemble_samples/ensemble.png` — Ensemble of possible geographies

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config.yaml` to adjust:
- NER model (default: `en_core_web_lg`)
- Inference method (`emcee` or `metropolis`)
- MCMC parameters (samples, burn-in, beta)
- Visualization settings

## Corpus

*The Great Gatsby* (F. Scott Fitzgerald, 1925) is in the US public domain as of 2021. The pipeline automatically downloads it from Project Gutenberg on first run.

## Architecture Notes

- All coordinate arithmetic is done in a local planar (km) system to avoid mixing lat/lon with Euclidean distances.
- Fictional places are latent variables; the MCMC sampler finds configurations that satisfy extracted textual constraints.
- The `src/utils/schemas.py` Pydantic models define strict contracts between pipeline phases.
- Geocoding results are cached to avoid redundant API calls.
