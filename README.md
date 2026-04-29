# Gatsby-100: Zero-Shot Spatial Role Labeling on Literary Prose

> **A small-scale benchmark and harness for evaluating open instruction-tuned LLMs as zero-shot extractors of spatial relations from literary text — built on a 100-sentence sample of *The Great Gatsby*.**

This repository's primary contribution is **`eval/`** — a self-contained
benchmark comparing three small open LLMs (Mistral-7B, Phi-3-mini, Qwen2.5-7B,
all served via Ollama) on a 100-sentence Spatial Role Labeling (SpRL) task
drawn uniformly at random from F. Scott Fitzgerald's *The Great Gatsby* (1925,
US public domain since 2021).

The downstream geographic inference pipeline (Phases 1–8 below) is retained
as a **demonstration of one downstream use** of the extracted relations: a
probabilistic reconstruction of the novel's fictional geography by MCMC
sampling. It is intentionally framed as a demonstration, not a primary
contribution — the cartographic problem is severely under-determined by
sparse textual evidence, and the resulting maps quantify uncertainty rather
than localise it.

---

## Table of Contents

1. [The SpRL Benchmark — Quickstart](#1-the-sprl-benchmark--quickstart)
2. [Task Definition](#2-task-definition)
3. [Evaluation Methodology](#3-evaluation-methodology)
4. [Models Compared](#4-models-compared)
5. [Annotation Schema](#5-annotation-schema)
6. [Limitations & Honest Scope](#6-limitations--honest-scope)
7. [Downstream Demonstration: Probabilistic Cartography](#7-downstream-demonstration-probabilistic-cartography)
8. [Repository Layout](#8-repository-layout)
9. [Installation](#9-installation)
10. [Testing](#10-testing)
11. [References](#11-references)

---

## 1. The SpRL Benchmark — Quickstart

```bash
# (a) Generate the sentence template (already committed; re-run to refresh).
python -m eval.sample_sentences --n 100 --seed 42

# (b) Hand-annotate eval/gold/sentences_to_annotate.jsonl per the schema.

# (c) Pull the three models in Ollama.
ollama pull mistral
ollama pull phi3
ollama pull qwen2.5

# (d) Extract.
python -m eval.extract --model mistral --tag mistral
python -m eval.extract --model phi3    --tag phi3
python -m eval.extract --model qwen2.5 --tag qwen2.5

# (e) Score and produce the comparison table + chart.
python -m eval.score --pred \
    eval/predictions/mistral.jsonl \
    eval/predictions/phi3.jsonl \
    eval/predictions/qwen2.5.jsonl
```

Outputs land in `eval/results/`: `scores.json`, `scores.md`, `per_relation.md`,
and `scores.png`. Full benchmark documentation is in [`eval/README.md`](eval/README.md).

---

## 2. Task Definition

We adopt the location-restricted form of **Spatial Role Labeling** (Kordjamshidi,
van Otterlo, & Moens, 2017). For each input sentence the system must emit a
list of triples

```
( location_1 ,  spatial_indicator ,  location_2 )
```

where:

- **location_1** (the *trajector*) — the entity being located,
- **location_2** (the *landmark*) — the reference entity it is located relative to,
- **spatial_indicator** — the verbatim cue word/phrase from the text expressing the relation,

with **both arguments restricted to LOCATION spans** (geographic places,
regions, buildings, named landmarks). Each triple also carries a
`semantic_type ⊆ {REGION, DIRECTION, DISTANCE}`.

Sentences with no valid LOCATION-LOCATION relation must produce an empty
list. A uniform random sample of 100 Gatsby sentences contains many such
sentences — they exercise model **precision** by punishing hallucinated
triples.

---

## 3. Evaluation Methodology

The 100 sentences are sampled with `random.Random(42).sample(corpus, 100)`,
filtered only by a minimum length of 20 characters to drop chapter headers
and stray punctuation lines. The resulting set is the **Gatsby-100**
benchmark.

### Matching schemes

A predicted triple matches a gold triple iff all three of `location_1`,
`location_2`, and `spatial_indicator` agree under one of:

- **strict** — exact match after lowercase + whitespace normalisation.
- **lenient** — token-set Jaccard ≥ 0.5 on each slot independently.

Per sentence, we greedily pair predictions with gold (each gold relation can
match at most once). Unmatched predictions are FP; unmatched gold are FN.

### Reported metrics

For every model:

| Metric | Description |
|---|---|
| **triple-level P / R / F1** | both *strict* and *lenient* schemes |
| **per-`semantic_type` P / R / F1** | breakdown over REGION / DIRECTION / DISTANCE |
| **empty-correct** | sentences where gold = ∅ and prediction = ∅ |
| **missed-on-non-empty** | gold ≠ ∅, prediction = ∅ (recall failure) |
| **hallucinated-on-empty** | gold = ∅, prediction ≠ ∅ (precision failure) |

The last two distinguish *conservative* from *hallucinatory* failure modes
— a useful axis when the underlying corpus has a heavy zero-relation tail
(common in literary text, where most sentences are about people or feeling
rather than places).

---

## 4. Models Compared

| Tag | Model (Ollama) | Params | Provider | Notes |
|---|---|---:|---|---|
| `mistral`  | `mistral`  | 7.2 B | Mistral AI | v0.3 instruct |
| `phi3`     | `phi3`     | 3.8 B | Microsoft | small, instruction-tuned |
| `qwen2.5`  | `qwen2.5`  | 7.6 B | Alibaba | strong on structured output |

All three are queried with the same system + user prompt (see
`eval/extract.py`, `SYSTEM_PROMPT`), `temperature=0.1`, `format="json"`.

The benchmark is intentionally restricted to **small open models served
locally** — no API-gated frontier models. The motivating question is
practical: *can a literary scholar, on commodity hardware, get usable
spatial-relation extraction without sending the corpus to a hosted API?*

---

## 5. Annotation Schema

Each line of `eval/gold/sentences_to_annotate.jsonl` is a single JSON object
the annotator edits in place. Empty `location_relations` lists are valid
and expected for the majority of sentences.

```json
{
  "annotation_id": "ann_001",
  "doc_id": "great_gatsby",
  "sentence_id": "great_gatsby_sent_498",
  "sentence": "\"I live at West Egg.",
  "location_relations": [
    {
      "location_1": "I",
      "location_2": "West Egg",
      "spatial_indicator": "live at",
      "semantic_type": ["REGION"]
    }
  ],
  "_notes": ""
}
```

The full annotation guidelines, including borderline cases, are in
[`eval/README.md`](eval/README.md).

---

## 6. Limitations & Honest Scope

This is a **small-scale, single-corpus, single-annotator** benchmark.
Concrete limits:

- **N = 100** sentences — adequate for qualitative model comparison, too small
  for statistical claims about either model. Confidence intervals on F1 are
  wide; differences smaller than ~0.05 should not be over-interpreted.
- **One author, one period, one register.** Findings will not generalise to
  e.g. modernist non-narrative prose, contemporary fiction, or non-English text.
- **Single annotator.** No inter-annotator agreement reported. Submission to
  any peer-reviewed venue should add a second pass on at least 30 sentences
  with Cohen's κ.
- **Zero-shot only.** No fine-tuning, no in-context examples, no chain-of-thought.
  The numbers are a *floor* on what each model can do, not a ceiling.
- **No non-LLM baseline.** A spaCy dependency-parse rule system would establish
  whether the LLMs are providing real value over classical tools; this is the
  most cost-effective addition to the harness.

This benchmark is best framed as a **resource paper** (LREC) or a **digital
humanities workshop submission** (CHR, DH). It is not, in its current form,
appropriate for a main-track NLP conference.

---

## 7. Downstream Demonstration: Probabilistic Cartography

The original project goal — reconstructing the spatial layout of Fitzgerald's
fictional places — is retained as a *downstream demonstration* of what one
might do with extracted SpRL triples. Phases 1–8 of `src/` implement:

1. **Phase 1** — Corpus prep (Project Gutenberg → cleaned sentences).
2. **Phase 2 / 2b** — NER + coreference (spaCy `en_core_web_lg`, CoReNer).
3. **Phase 3** — Geocoding of real places (Nominatim, cached).
4. **Phase 4** — SpRL extraction (the same Mistral/Ollama backend the eval harness benchmarks).
5. **Phase 5** — Compilation of triples into a formal energy model over a local planar coordinate frame.
6. **Phase 6** — MCMC inference on the latent positions of fictional places (`emcee` ensemble sampler).
7. **Phase 7** — Convergence diagnostics (R-hat, ESS, credible ellipses).
8. **Phase 8** — Visualisation (heatmaps, Folium overlays, ensemble cartograms, constraint graph).

### Outputs (committed)

- Per-entity posterior heatmaps in `visualizations/heatmaps/` (~40 PNGs).
- Joint-posterior ensemble cartogram in `visualizations/ensemble_samples/ensemble.png`.
- Interactive constraint graph in `visualizations/constraint_graph.html`.
- Folium overlay map in `visualizations/overlay_maps/full_map.html`.

### Honest framing

The cartographic problem is **severely under-determined**. *East Egg* has only
a handful of usable textual constraints; no sampler can pin a 2-D position
from such sparse evidence. The resulting heatmaps therefore *quantify
uncertainty* rather than *localise* the fictional places — they are honest
about how much the text actually constrains geography. This is the right
answer to the wrong question; the right question is the SpRL extraction
benchmark above.

### Run the cartography pipeline

```bash
python -m src.pipeline --config config.yaml          # full pipeline
python -m src.pipeline --config config.yaml --phase 6 --force
```

---

## 8. Repository Layout

```
FitzTry1/
├── eval/                         # ★ primary contribution
│   ├── sample_sentences.py       # uniform random 100-sentence sampler
│   ├── extract.py                # Ollama extractor (model-agnostic)
│   ├── score.py                  # strict + lenient scoring, charts
│   ├── README.md                 # benchmark docs
│   ├── gold/                     # gold annotations (hand-filled)
│   ├── predictions/              # per-model JSONL predictions
│   └── results/                  # scores.json, scores.md, scores.png
├── src/                          # downstream cartography pipeline
│   ├── pipeline.py
│   ├── phase1_corpus_prep.py … phase8_visualization.py
│   ├── phase_mistral_joint.py    # production Ollama / Mistral SpRL extractor
│   └── utils/{schemas,geo,io}.py
├── corpus/                       # raw + cleaned text
├── data/                         # phase outputs (entities, relations, samples)
├── visualizations/               # heatmaps, overlays, ensemble cartogram
├── tests/                        # 66 tests across 5 modules
├── config.yaml                   # primary configuration
└── requirements.txt
```

---

## 9. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Ollama (for the eval harness AND Phase 4)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral phi3 qwen2.5
```

Tested on Python 3.13.6 / macOS arm64. If `blis` fails to build on Python 3.13,
install spaCy with `pip install --only-binary :all: spacy`.

---

## 10. Testing

```bash
pytest tests/ -v   # 66 tests across NER, relations, constraints, inference
```

---

## 11. References

- Kordjamshidi, P., van Otterlo, M., & Moens, M.-F. (2017). *Spatial Role Labeling Annotation Scheme.* In Ide & Pustejovsky (Eds.), *Handbook of Linguistic Annotation*. Springer.
- Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013). *emcee: The MCMC Hammer.* PASP 125(925), 306–312.
- Goodman, J., & Weare, J. (2010). *Ensemble samplers with affine invariance.* CAMCoS 5(1), 65–80.
- Gelman, A., & Rubin, D. B. (1992). *Inference from Iterative Simulation Using Multiple Sequences.* Statistical Science 7(4), 457–472.
- Moretti, F. (1998). *Atlas of the European Novel, 1800–1900.* Verso.
- Piper, A. (2018). *Enumerations: Data and Literary Study.* University of Chicago Press.

---

*Built for the Wheel of Fortune Lab at Columbia. Corpus: F. Scott Fitzgerald,
*The Great Gatsby* (1925, US public domain since 2021).*
