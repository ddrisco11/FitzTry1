# Gatsby-100 — A small SpRL benchmark for literary prose

This directory contains the evaluation harness comparing small open LLMs as
zero-shot Spatial Role Labeling (SpRL) extractors on a 100-sentence sample
of *The Great Gatsby*.

## Layout

```
eval/
├── sample_sentences.py       # uniform random sampler (seed 42, no filtering)
├── extract.py                # Ollama-based extractor (model-agnostic)
├── score.py                  # strict + lenient triple-level scoring
├── gold/
│   └── sentences_to_annotate.jsonl   # YOU FILL THIS IN
├── predictions/
│   ├── mistral.jsonl
│   ├── phi3.jsonl
│   └── qwen2.5.jsonl
└── results/
    ├── scores.json
    ├── scores.md
    ├── per_relation.md
    └── scores.png
```

## End-to-end workflow

```bash
# 1. (Re)generate the random sample. Already committed at commit time.
python -m eval.sample_sentences --n 100 --seed 42

# 2. Hand-annotate eval/gold/sentences_to_annotate.jsonl per the schema below.

# 3. Pull the three models in Ollama (~6–8 GB total).
ollama pull mistral        # 7B, mistralai
ollama pull phi3           # 3.8B, microsoft
ollama pull qwen2.5        # 7B, alibaba

# 4. Run extraction for each model.
python -m eval.extract --model mistral --tag mistral
python -m eval.extract --model phi3    --tag phi3
python -m eval.extract --model qwen2.5 --tag qwen2.5

# 5. Score and produce comparison tables + chart.
python -m eval.score \
    --pred eval/predictions/mistral.jsonl \
           eval/predictions/phi3.jsonl \
           eval/predictions/qwen2.5.jsonl
```

## Annotation schema

Each line of `gold/sentences_to_annotate.jsonl` is a JSON object:

```json
{
  "annotation_id": "ann_000",
  "doc_id": "great_gatsby",
  "sentence_id": "great_gatsby_sent_42",
  "sentence": "I lived at West Egg, the — well, the less fashionable of the two ...",
  "location_relations": [
    {
      "location_1": "I",
      "location_2": "West Egg",
      "spatial_indicator": "lived at",
      "semantic_type": ["REGION"]
    }
  ],
  "_notes": "optional free-text"
}
```

### Rules

A relation is a triple `(location_1, spatial_indicator, location_2)` where
**both arguments must be LOCATION spans** — geographic places, regions,
buildings, or named landmarks. Reject relations where one argument is a
person, a body part, an abstract concept, or a temporal phrase.

`semantic_type` is a subset of:

| Type | Meaning | Example cue |
|---|---|---|
| `REGION` | Topological / containment | *in*, *at*, *within*, *part of* |
| `DIRECTION` | Cardinal or relative direction | *north of*, *across from*, *beyond* |
| `DISTANCE` | Metric or qualitative distance | *twenty miles from*, *near*, *far from* |

A single triple can carry multiple types (e.g. *"twenty miles north of the city"* → `["DIRECTION", "DISTANCE"]`).

### What to skip

- Sentences with no LOCATION-LOCATION relation → leave `location_relations: []`.
  This is the **majority** case and is the right answer; do not invent triples.
- Pronominal references with no clear antecedent in the same sentence.
- Possessives that don't denote location (*"Daisy's voice"*).
- Pure metaphor (*"a valley of ashes"* used as a name is fine; *"a valley of
  unease"* metaphorical is not).

### Tips

- Quote the spatial indicator **verbatim** from the sentence.
- Prefer the shortest informative `location_1` / `location_2` spans.
- Hard cases (*"the city"* with no antecedent) → mark as you read literally,
  add a one-line `_notes` if you want to flag it.

## Scoring

`score.py` reports two matchings:

- **strict:** all three slots (`location_1`, `location_2`, `spatial_indicator`)
  string-match exactly after lowercase + whitespace normalisation.
- **lenient:** token-set Jaccard ≥ 0.5 on each slot.

Per-model output: triple-level Precision/Recall/F1, per-`semantic_type`
breakdown, and three sentence-level diagnostics:

- *empty-correct* — the model correctly produced no triples.
- *missed-on-non-empty* — the model produced no triples when gold had ≥ 1.
- *hallucinated-on-empty* — the model produced ≥ 1 triple when gold had none.

The last two characterise different failure modes: **conservative** (high
precision, low recall) vs. **hallucinatory** (low precision, possibly high
recall).

## What this is not

- **Not an inter-annotator agreement study.** N=1 annotator. To submit anywhere
  citable, get a second pass on at least 30 sentences and compute Cohen's κ.
- **Not multi-domain.** Single corpus, single author, single period.
- **Not novel architecture work.** It's a zero-shot benchmark of off-the-shelf
  small open LLMs.

Best fit as a *resource paper* (LREC) or a workshop submission at a digital
humanities venue (CHR, DH).

## Citation

If you use the gold annotations, cite the corpus (Fitzgerald, *The Great
Gatsby*, 1925, US public domain) and this repository.
