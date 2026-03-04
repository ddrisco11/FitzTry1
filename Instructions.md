# Instructions: Probabilistic Reconstruction of Literary Geography

## Project Overview

Build a reproducible computational pipeline that extracts geographic entities and spatial relationships from F. Scott Fitzgerald's literary corpus, constructs a probabilistic spatial model from those constraints, and produces visualizations of the inferred fictional geography.

The output is **not** a single deterministic map — it is a **probability distribution over spatial configurations** consistent with textual evidence.

---

## Directory Structure

```
FitzTry1/
├── Instructions.md
├── requirements.txt
├── README.md
├── config.yaml                  # Global config (paths, API keys, model params)
├── corpus/
│   ├── raw/                     # Original plain-text source files
│   ├── cleaned/                 # Cleaned, sentence-segmented, tokenized
│   └── metadata.json            # Document-level metadata index
├── src/
│   ├── __init__.py
│   ├── phase1_corpus_prep.py
│   ├── phase2_ner.py
│   ├── phase3_grounding.py
│   ├── phase4_relations.py
│   ├── phase5_constraints.py
│   ├── phase6_inference.py
│   ├── phase7_convergence.py
│   ├── phase8_visualization.py
│   ├── pipeline.py              # Orchestrator: runs all phases end-to-end
│   └── utils/
│       ├── __init__.py
│       ├── io.py                # File I/O helpers
│       ├── schemas.py           # Pydantic models for all data schemas
│       └── geo.py               # Geocoding + coordinate utilities
├── data/
│   ├── entities.jsonl           # Phase 2 output
│   ├── grounded_entities.jsonl  # Phase 3 output
│   ├── relations.jsonl          # Phase 4 output
│   ├── constraints.json         # Phase 5 output
│   ├── samples/                 # Phase 6 output (sampled configurations)
│   └── convergence/             # Phase 7 output (statistics, diagnostics)
├── visualizations/              # Phase 8 output
│   ├── constraint_graph.html
│   ├── heatmaps/
│   ├── overlay_maps/
│   └── ensemble_samples/
├── notebooks/
│   └── exploration.ipynb        # Interactive exploration / demo
└── tests/
    ├── test_ner.py
    ├── test_relations.py
    ├── test_constraints.py
    └── test_inference.py
```

---

## Technology Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.11+ |
| NER | spaCy (`en_core_web_trf`) + custom rules |
| Dependency Parsing | spaCy transformer pipeline |
| Relation Extraction | spaCy matchers + optional HuggingFace RE model |
| Geocoding | `geopy` (Nominatim / OpenStreetMap) |
| Data Schemas | `pydantic` v2 |
| Probabilistic Inference | `numpy`, `scipy`, optionally `pymc` or `emcee` |
| Visualization | `folium` (maps), `networkx` + `pyvis` (graphs), `matplotlib` / `seaborn` (heatmaps) |
| Configuration | `pyyaml` |
| Testing | `pytest` |
| CLI | `click` or `argparse` |

All dependencies go in `requirements.txt` with pinned versions.

---

## Phase 1 — Corpus Preparation

**File:** `src/phase1_corpus_prep.py`

### Input
- Plain-text files of Fitzgerald works placed in `corpus/raw/`.
- For the MVP, include at minimum *The Great Gatsby* (public domain).

### Tasks

1. **Load** each `.txt` file from `corpus/raw/`.
2. **Clean** the text:
   - Strip Project Gutenberg headers/footers if present.
   - Normalize whitespace, fix encoding issues, strip non-text artifacts.
3. **Sentence-segment** using spaCy's sentencizer or a rule-based splitter.
4. **Tokenize** each sentence (spaCy `Doc` objects are fine; store token-level data only if needed downstream).
5. **Assign IDs**:
   - Each document gets a `doc_id` derived from filename (e.g., `gatsby`, `tender_is_the_night`).
   - Each sentence gets a `sentence_id` in the format `{doc_id}_sent_{n}` (0-indexed).
6. **Build metadata index** (`corpus/metadata.json`):

```json
[
  {
    "doc_id": "gatsby",
    "title": "The Great Gatsby",
    "year": 1925,
    "type": "novel",
    "source_file": "raw/the_great_gatsby.txt",
    "num_sentences": 3214
  }
]
```

7. **Write** cleaned, sentence-segmented output to `corpus/cleaned/` as JSONL:

```json
{"doc_id": "gatsby", "sentence_id": "gatsby_sent_0", "text": "In my younger and more vulnerable years..."}
```

### Output
- `corpus/cleaned/{doc_id}.jsonl` — one line per sentence.
- `corpus/metadata.json` — document-level index.

---

## Phase 2 — Named Entity Recognition

**File:** `src/phase2_ner.py`

### Objective
Extract all geographic entities from every sentence.

### Tasks

1. **Load** spaCy transformer model (`en_core_web_trf`).
2. **Process** each sentence from Phase 1 output.
3. **Filter** entities to geographic types:
   - `GPE` — Geo-Political Entity (cities, states, countries)
   - `LOC` — Location (mountains, rivers, regions)
   - `FAC` — Facility (buildings, airports, bridges)
4. **Deduplicate** entity mentions across the corpus. Build a canonical entity registry:
   - Normalize surface forms (e.g., "N.Y." → "New York").
   - Merge co-referent mentions.
5. **Classify** each unique entity as `real` or `fictional`:
   - Attempt geocoding lookup (Phase 3 preview). If the entity resolves to coordinates with high confidence → `real`.
   - If it fails geocoding AND is not in a curated gazetteer → `fictional`.
   - Maintain a small manual override dictionary for known Fitzgerald fictional places:
     ```python
     KNOWN_FICTIONAL = {"East Egg", "West Egg", "Valley of Ashes"}
     ```
6. **Write** output to `data/entities.jsonl`:

```json
{
  "entity_id": "e_001",
  "name": "East Egg",
  "canonical_name": "East Egg",
  "type": "fictional",
  "ner_label": "GPE",
  "mentions": [
    {"sentence_id": "gatsby_sent_44", "char_start": 12, "char_end": 20, "source_text": "...across the bay from East Egg..."}
  ],
  "mention_count": 37,
  "doc_ids": ["gatsby"]
}
```

### Notes
- Use `nlp.pipe()` with batching for throughput.
- Log any entity that the classifier is uncertain about (confidence < 0.7) for manual review.

---

## Phase 3 — Geographic Grounding

**File:** `src/phase3_grounding.py`

### Objective
Assign real-world coordinates to entities classified as `real`. Fictional entities remain ungrounded.

### Tasks

1. **Load** `data/entities.jsonl`.
2. **For each entity where `type == "real"`:**
   a. Query geocoding API via `geopy.geocoders.Nominatim` (respect rate limits — 1 req/sec).
   b. If multiple candidate results, disambiguate using:
      - Context window: surrounding sentences in the corpus.
      - Co-occurrence: which other grounded entities appear nearby in the text? Prefer the candidate geographically closer to co-occurring entities.
      - Prior: prefer US locations (Fitzgerald's primary setting) unless context clearly indicates otherwise.
   c. Assign:
      - `latitude`, `longitude`
      - `confidence` score (0.0–1.0) based on geocoder result quality and disambiguation certainty.
3. **For each entity where `type == "fictional"`:**
   - Set `latitude = null`, `longitude = null`.
   - These become **latent variables** in Phase 5.
4. **Write** output to `data/grounded_entities.jsonl`:

```json
{
  "entity_id": "e_002",
  "name": "New York",
  "type": "real",
  "latitude": 40.7128,
  "longitude": -74.0060,
  "confidence": 0.95
}
```

### Rate Limiting
- Cache all geocoding results to a local SQLite DB or JSON file to avoid redundant API calls.
- Use a `time.sleep(1.1)` between requests to Nominatim.

---

## Phase 4 — Spatial Relation Extraction

**File:** `src/phase4_relations.py`

### Objective
Extract pairwise spatial constraints between entities from the text.

### Target Relation Types

| Category | Relations |
|---|---|
| Directional | `north_of`, `south_of`, `east_of`, `west_of` |
| Proximity | `near`, `far` |
| Metric | `distance_approx` (with estimated value in miles/km) |
| Topological | `across`, `on_coast`, `in_region` |

### Method

1. **Load** cleaned sentences and entity mentions.
2. **For each sentence containing 2+ geographic entities:**
   a. Run spaCy dependency parse.
   b. Apply **pattern-based extraction** using `spaCy Matcher` or `DependencyMatcher`:

   Example patterns (non-exhaustive — expand these):

   | Text Pattern | Extracted Relation |
   |---|---|
   | "north of {B}" | `north_of(A, B)` |
   | "south of {B}" | `south_of(A, B)` |
   | "east of {B}" | `east_of(A, B)` |
   | "west of {B}" | `west_of(A, B)` |
   | "near {B}", "close to {B}", "next to {B}", "beside {B}" | `near(A, B)` |
   | "far from {B}", "a long way from {B}" | `far(A, B)` |
   | "across the bay from {B}", "across from {B}" | `across(A, B)` |
   | "on the coast", "on the shore" | `on_coast(A)` |
   | "in {B}" (where B is a region) | `in_region(A, B)` |
   | "{N} miles from {B}" | `distance_approx(A, B, N)` |

   c. Also extract **implicit proximity**: if two entities appear in the same sentence with no spatial language, record a weak `co_occurrence(A, B)` relation.

3. **Assign uncertainty scores** to each extracted relation:
   - Base confidence from extraction method (pattern match = 0.8, co-occurrence = 0.3).
   - Penalize for linguistic hedging ("perhaps near", "somewhere north of").
   - Penalize for metaphorical or non-literal usage (attempt to detect via dependency parse context).

4. **Write** output to `data/relations.jsonl`:

```json
{
  "relation_id": "r_001",
  "type": "across",
  "entity_1": "East Egg",
  "entity_2": "West Egg",
  "direction": null,
  "distance_value": null,
  "distance_unit": null,
  "weight": 0.85,
  "uncertainty": 0.15,
  "source_sentence_id": "gatsby_sent_44",
  "source_text": "...across the bay from East Egg...",
  "extraction_method": "pattern_match"
}
```

### Notes
- The pattern list above is a starter set. Expand it by reading through the corpus and identifying Fitzgerald's spatial language habits.
- Consider building a small evaluation set: manually annotate 50 sentences and compute precision/recall on the pattern extractor.

---

## Phase 5 — Formal Spatial Constraint Model

**File:** `src/phase5_constraints.py`

### Objective
Convert extracted relations into formal mathematical constraints over 2D coordinates.

### Variable Definitions

- **Real entities** `R_j`: fixed at grounded coordinates `(lat_j, lon_j)`. Convert to a local planar coordinate system (e.g., UTM or simple equirectangular projection centered on the corpus centroid) so all math operates in kilometers/miles.
- **Fictional entities** `F_i`: latent variables `(x_i, y_i)` to be inferred.

### Constraint Translation

Each relation maps to a soft constraint (a term in the energy function):

| Relation | Constraint | Energy Penalty |
|---|---|---|
| `north_of(A, B)` | `y_A > y_B + ε` | `max(0, y_B + ε - y_A)²` |
| `south_of(A, B)` | `y_A < y_B - ε` | `max(0, y_A - y_B + ε)²` |
| `east_of(A, B)` | `x_A > x_B + ε` | `max(0, x_B + ε - x_A)²` |
| `west_of(A, B)` | `x_A < x_B - ε` | `max(0, x_A - x_B + ε)²` |
| `near(A, B)` | `\|\|A - B\|\| < d_near` | `max(0, \|\|A - B\|\| - d_near)²` |
| `far(A, B)` | `\|\|A - B\|\| > d_far` | `max(0, d_far - \|\|A - B\|\|)²` |
| `distance_approx(A, B, D)` | `\|\|A - B\|\| ≈ D` | `(\|\|A - B\|\| - D)² / (2σ²)` |
| `across(A, B)` | body of water between; treat as `near` + directional hint | combine constraints |
| `in_region(A, B)` | `\|\|A - centroid(B)\|\| < radius(B)` | penalty if outside |

### Configuration Parameters (in `config.yaml`)

```yaml
constraints:
  epsilon_direction_km: 1.0
  d_near_km: 10.0
  d_far_km: 50.0
  sigma_distance_km: 5.0
  co_occurrence_weight: 0.1
```

### Tasks

1. Load `data/grounded_entities.jsonl` and `data/relations.jsonl`.
2. Convert all grounded coordinates to a local planar system (km).
3. Build a list of constraint objects, each with:
   - References to entity variables.
   - A penalty function (callable).
   - A weight (from the relation's weight field).
4. Write the constraint model specification to `data/constraints.json`:

```json
{
  "fixed_entities": [
    {"entity_id": "e_002", "name": "New York", "x": 0.0, "y": 0.0}
  ],
  "latent_entities": [
    {"entity_id": "e_001", "name": "East Egg", "init_x": null, "init_y": null}
  ],
  "constraints": [
    {
      "constraint_id": "c_001",
      "type": "near",
      "entities": ["e_001", "e_003"],
      "params": {"d_near_km": 10.0},
      "weight": 0.85,
      "source_relation_id": "r_001"
    }
  ],
  "coordinate_system": {
    "projection": "equirectangular",
    "origin_lat": 40.7128,
    "origin_lon": -74.0060,
    "units": "km"
  }
}
```

---

## Phase 6 — Probabilistic Inference

**File:** `src/phase6_inference.py`

### Objective
Sample coordinate configurations for fictional entities that minimize total constraint violation.

### Energy Function

```
E(config) = Σ_c  weight_c * penalty_c(config)
```

where the sum is over all constraints, and `config` is the vector of all latent `(x_i, y_i)`.

### Sampling Distribution

```
P(config) ∝ exp(-β * E(config))
```

`β` is an inverse temperature parameter controlling sharpness.

### Implementation — MVP (Simple Monte Carlo)

1. Initialize fictional entity positions randomly within a bounding box around the real entities (with some padding).
2. Compute `E(config)`.
3. Propose a new configuration by perturbing one entity's position with Gaussian noise.
4. Accept/reject via Metropolis criterion:
   - `ΔE = E(new) - E(old)`
   - Accept if `ΔE < 0`, else accept with probability `exp(-β * ΔE)`.
5. Repeat for `N` iterations (configurable, default 100,000).
6. Record samples every `thin` steps (default every 100).
7. Save samples to `data/samples/`:

```json
{"sample_id": 0, "entities": {"e_001": {"x": 12.3, "y": -4.5}, "e_003": {"x": 8.1, "y": -3.2}}, "energy": 2.34}
```

### Implementation — Extended (MCMC via `emcee` or `pymc`)

- Use `emcee` ensemble sampler for better mixing in high dimensions.
- Or formulate as a `pymc` model with custom potentials for each constraint.
- Run multiple chains and check convergence (see Phase 7).

### Configuration (`config.yaml`)

```yaml
inference:
  method: "metropolis"        # or "emcee" or "pymc"
  num_samples: 100000
  burn_in: 10000
  thin: 100
  beta: 1.0
  proposal_std_km: 2.0
  num_chains: 4
  random_seed: 42
```

### Output
- `data/samples/*.jsonl` — sampled configurations.
- Each sample includes the energy and all latent positions.

---

## Phase 7 — Convergence Diagnostics & Uncertainty

**File:** `src/phase7_convergence.py`

### Objective
Assess sampling quality and compute posterior summaries for each fictional entity's position.

### Tasks

1. **Load** all samples from `data/samples/`.
2. **Discard burn-in** (first `burn_in` samples).
3. **Convergence diagnostics** (if multiple chains):
   - Gelman-Rubin R-hat statistic for each latent variable. Target: R-hat < 1.1.
   - Trace plots (save as images).
   - Effective sample size (ESS).
4. **Posterior summaries** for each fictional entity:
   - **Posterior mean** `(x̄, ȳ)`.
   - **Posterior standard deviation** `(σ_x, σ_y)`.
   - **95% credible region** (highest posterior density or simple ellipse).
   - **Spatial entropy**: `H = -Σ p log p` over a discretized grid — higher entropy means more uncertain placement.
   - **Multimodality detection**: run a simple clustering (e.g., KMeans with k=1..5, pick k by BIC) on the samples for each entity. If k > 1, the posterior is multimodal.
5. **Constraint satisfaction score**:
   - For each sample, compute fraction of constraints satisfied (penalty below threshold).
   - Report mean and std across samples.
6. **Sensitivity analysis** (optional/extended):
   - Re-run inference dropping 10% of relations randomly. Measure how much posterior means shift. Report stability.
7. **Write** diagnostics to `data/convergence/`:

```json
{
  "entity_id": "e_001",
  "name": "East Egg",
  "posterior_mean": {"x": 12.1, "y": -4.3},
  "posterior_std": {"x": 1.2, "y": 0.8},
  "credible_region_95": {"type": "ellipse", "semi_major": 2.5, "semi_minor": 1.6, "angle_deg": 30},
  "spatial_entropy": 3.21,
  "num_modes": 1,
  "r_hat": {"x": 1.02, "y": 1.01},
  "ess": {"x": 834, "y": 912}
}
```

---

## Phase 8 — Visualization

**File:** `src/phase8_visualization.py`

### Objective
Produce publication-quality visualizations of the inferred literary geography.

### Required Outputs

#### 1. Constraint Graph
- **Library:** `networkx` + `pyvis` (interactive HTML) or `matplotlib` (static).
- Nodes = entities (color by real/fictional).
- Edges = extracted relations (label with type, thickness by weight).
- Save to `visualizations/constraint_graph.html`.

#### 2. Heatmaps of Posterior Distributions
- **Library:** `matplotlib` / `seaborn`.
- For each fictional entity, plot a 2D kernel density estimate of its sampled positions.
- Overlay the 95% credible ellipse.
- Save to `visualizations/heatmaps/{entity_name}.png`.

#### 3. Real-World Map Overlay
- **Library:** `folium`.
- Plot real entities as pinned markers.
- Plot fictional entity posterior means as markers with uncertainty circles.
- Optionally overlay heatmap tiles for fictional entity distributions.
- Save to `visualizations/overlay_maps/full_map.html`.

#### 4. Ensemble Sample Visualization
- **Library:** `matplotlib`.
- Plot N (e.g., 50) sampled configurations simultaneously — each sample is a set of points connected by constraint edges.
- Shows the "cloud" of possible geographies.
- Save to `visualizations/ensemble_samples/ensemble.png`.

#### 5. Cross-Novel Comparison (Extended)
- If multiple works are processed, produce side-by-side or overlaid maps showing how the inferred geography shifts across novels.

---

## Pipeline Orchestrator

**File:** `src/pipeline.py`

Runs all phases sequentially with a single command:

```bash
python -m src.pipeline --config config.yaml
```

Or run individual phases:

```bash
python -m src.pipeline --config config.yaml --phase 2
```

### Implementation

```python
import click
from src.phase1_corpus_prep import run as run_phase1
from src.phase2_ner import run as run_phase2
# ... etc.

PHASES = [run_phase1, run_phase2, run_phase3, run_phase4, run_phase5, run_phase6, run_phase7, run_phase8]

@click.command()
@click.option("--config", default="config.yaml")
@click.option("--phase", default=None, type=int, help="Run only this phase (1-8)")
def main(config, phase):
    cfg = load_config(config)
    if phase:
        PHASES[phase - 1](cfg)
    else:
        for p in PHASES:
            p(cfg)
```

Each phase function:
- Takes a config dict.
- Reads its inputs from the expected data paths.
- Writes its outputs to the expected data paths.
- Logs progress to stdout.
- Raises on failure with a clear error message.

---

## Data Schemas (Pydantic)

**File:** `src/utils/schemas.py`

Define strict Pydantic v2 models for every data interchange format:

- `SentenceRecord` — Phase 1 output
- `EntityMention` — sub-model
- `Entity` — Phase 2 output
- `GroundedEntity` — Phase 3 output
- `SpatialRelation` — Phase 4 output
- `Constraint` — Phase 5 output
- `Sample` — Phase 6 output
- `PosteriorSummary` — Phase 7 output

All JSONL reading/writing should validate through these models. This catches data corruption between phases early.

---

## Configuration File

**File:** `config.yaml`

```yaml
corpus:
  raw_dir: "corpus/raw"
  cleaned_dir: "corpus/cleaned"
  metadata_file: "corpus/metadata.json"

ner:
  spacy_model: "en_core_web_trf"
  entity_types: ["GPE", "LOC", "FAC"]
  fictional_overrides: ["East Egg", "West Egg", "Valley of Ashes"]
  confidence_threshold: 0.7

grounding:
  geocoder: "nominatim"
  user_agent: "fitzgerald_geo_project"
  cache_file: "data/geocode_cache.json"
  default_country_bias: "US"

relations:
  extraction_methods: ["pattern_match", "co_occurrence"]
  co_occurrence_weight: 0.3
  hedging_penalty: 0.2

constraints:
  epsilon_direction_km: 1.0
  d_near_km: 10.0
  d_far_km: 50.0
  sigma_distance_km: 5.0
  projection_origin_lat: 40.7128
  projection_origin_lon: -74.0060

inference:
  method: "metropolis"
  num_samples: 100000
  burn_in: 10000
  thin: 100
  beta: 1.0
  proposal_std_km: 2.0
  num_chains: 4
  random_seed: 42

visualization:
  output_dir: "visualizations"
  map_zoom: 10
  heatmap_resolution: 100
  ensemble_num_samples: 50
```

---

## Testing

**Directory:** `tests/`

Write `pytest` tests for each phase. At minimum:

| Test File | What to Test |
|---|---|
| `test_ner.py` | Known entities are extracted from a sample paragraph. Fictional vs. real classification is correct for test cases. |
| `test_relations.py` | Pattern matcher extracts correct relation types from hand-crafted sentences. Uncertainty scores are within expected range. |
| `test_constraints.py` | Constraint energy functions return 0 for satisfied constraints and positive values for violations. Direction constraints enforce correct axis inequality. |
| `test_inference.py` | Sampler reduces energy over iterations. Known toy problem (3 entities, 2 constraints) converges to expected positions. |

Run all tests:

```bash
pytest tests/ -v
```

---

## Build Order

When implementing, follow this exact sequence. Each phase depends on the outputs of the previous one.

1. **Set up project scaffolding**: directory structure, `requirements.txt`, `config.yaml`, `README.md`.
2. **Implement `src/utils/schemas.py`**: all Pydantic models first — this defines the contract between phases.
3. **Implement `src/utils/io.py`**: JSONL read/write with schema validation, config loader.
4. **Implement `src/utils/geo.py`**: coordinate projection utilities, distance calculations.
5. **Phase 1** — `src/phase1_corpus_prep.py` + place at least one text file in `corpus/raw/`.
6. **Phase 2** — `src/phase2_ner.py`.
7. **Phase 3** — `src/phase3_grounding.py`.
8. **Phase 4** — `src/phase4_relations.py`.
9. **Phase 5** — `src/phase5_constraints.py`.
10. **Phase 6** — `src/phase6_inference.py`.
11. **Phase 7** — `src/phase7_convergence.py`.
12. **Phase 8** — `src/phase8_visualization.py`.
13. **Pipeline orchestrator** — `src/pipeline.py`.
14. **Tests** for each phase.
15. **Iteration**: run the full pipeline on *The Great Gatsby*, inspect outputs, tune parameters.

---

## MVP Scope (Semester-Feasible Minimum)

For a working first version, the following subset is sufficient:

- **Single novel**: *The Great Gatsby*.
- **Entity extraction**: spaCy NER with manual fictional entity list.
- **Grounding**: Nominatim geocoding with simple US-bias disambiguation.
- **Relations**: 3 types — `near`, `far`, directional (`north/south/east/west_of`).
- **Constraints**: basic energy terms as specified.
- **Inference**: simple Metropolis sampler, single chain, 100k iterations.
- **Visualization**: constraint graph + one map overlay + one heatmap.

---

## Extended Scope

Once the MVP works end-to-end:

- Full Fitzgerald corpus (novels + short stories).
- MCMC via `emcee` with convergence diagnostics.
- Full uncertainty modeling (credible regions, entropy, multimodality).
- Sensitivity analysis (robustness to dropped relations).
- Cross-novel geographic comparison visualizations.
- Fine-tuned relation extraction model (train on annotated Fitzgerald sentences).
- Interactive web dashboard (e.g., Streamlit or Panel).

---

## Key Implementation Notes

1. **Reproducibility**: set random seeds everywhere. Log all config to output directories.
2. **Idempotency**: each phase should be re-runnable. If output files exist, either overwrite or skip based on a `--force` flag.
3. **Logging**: use Python `logging` module, not `print`. Log at INFO level for progress, DEBUG for details.
4. **Error handling**: fail fast with clear messages. If a phase's input files are missing, say which files and which phase produces them.
5. **No hardcoded paths**: everything goes through `config.yaml`.
6. **Coordinate systems**: always be explicit about whether you're in lat/lon or projected km. Never mix them. The `geo.py` utility module handles all conversions.
7. **Corpus legality**: only use public-domain texts. *The Great Gatsby* entered US public domain in 2021. Verify status of other works before including them.
