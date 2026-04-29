# Non-Locations as Locations: An ISO-Space Spatial Graph Pipeline for Prose

> A document-agnostic pipeline that reads any English prose text and emits an
> ISO 24617-7-conformant graph of its spatial relations. Extraction is done
> zero-shot by a small open LLM; the resulting triples are rewritten into
> qualitative spatial calculi (RCC-8, CDC) and rendered as an interactive
> diagram. Each node carries an `entity_kind` tag — `place`, `deictic`,
> `person_locus`, or `common_locus` — making explicit the central observation
> of this work: in prose, entities that are not canonically locations
> nevertheless function as `SpatialEntity`s under the ISO-Space schema.

![Gatsby spatial graph](graphs/corpus/gatsby.svg)

> *The Great Gatsby spatial graph — 326 nodes, 241 edges. Blue edges are
> `QSLINK` (topological); red edges are `OLINK` (orientation). Open
> [`graphs/corpus/gatsby.html`](graphs/corpus/gatsby.html) for the
> interactive version.*

---

## Abstract

We present a reproducible pipeline that compiles, from arbitrary English
prose, a labeled directed multigraph of the text's spatial relations in the
ISO 24617-7 (ISO-Space) annotation tradition. The pipeline is one CLI command
end-to-end (`python -m src.cli run <file.txt>`) and contains no
document-specific code, names, or blocklists. A small open LLM (Mistral-7B
via Ollama, by default) performs zero-shot Spatial Role Labeling over
non-overlapping three-sentence windows; the verbatim spatial cues are
rewritten against an externally configurable cue lexicon into RCC-8 and
Cardinal-Direction-Calculus values; nodes are tagged with one of four
linguistic *entity kinds*. We argue, and our running example demonstrates,
that the entities most strongly connected in the resulting graph are often
*not* canonical places — first-person pronouns (`I`), deictic adverbs
(`here`), and persons (`Gatsby`) all function as legitimate
`SpatialEntity`s under the ISO-Space schema once one accepts the
trajector–landmark logic of the SpRL annotation. We treat this not as
extraction noise but as a feature of literary spatial language, and surface
it through the entity-kind tagging.

## 1. Introduction

Most "literary geography" projects assume that the locations in a novel are
the *places* in it — towns, buildings, rooms — and treat anything else
(pronouns, persons, common nouns) as extraction noise to be filtered out.
This pipeline rejects that assumption. Under the ISO 24617-7 schema, a
`SpatialEntity` is anything that can occupy the trajector or landmark slot
of a `QSLINK` (topological link) or `OLINK` (orientation link). Whether a
span *can* occupy that slot is a question of usage, not ontology. *"At
Gatsby's"* is a `TPP` (tangential proper part) relation between an unnamed
trajector and the metonymic region projected by the proper noun *Gatsby*;
*"here"* is an ego-deictic locus anchored to the speaker; *"I"* is the
deictic centre from which the rest of the spatial system is oriented.

We therefore design the pipeline so that:

1. The extractor is prompted to produce LOCATION-LOCATION relations and is
   not given any negative instruction about people, pronouns, or
   abstractions.
2. Every span the extractor produces becomes a first-class node.
3. A separate, post-hoc tagger annotates each node with an `entity_kind`
   ∈ {`place`, `deictic`, `person_locus`, `common_locus`} using only
   linguistic universals (a closed-class English deixis lexicon and a
   general-purpose NER model). No document-specific list is consulted.

The result: nothing is silently dropped, but a downstream consumer can
filter by kind if it wants only canonical places, or — more interestingly —
study the topology of the *non-place* sub-graph.

## 2. Related work and provenance of design choices

The pipeline assembles, rather than invents, ideas from four traditions.
*Spatial Role Labeling* (Kordjamshidi, van Otterlo & Moens 2017) supplies
the trajector–spatial-indicator–landmark triple as the unit of extraction.
*ISO-Space* (Pustejovsky, Moszkowicz & Verhagen 2015) supplies the
`SpatialEntity` / `QSLINK` / `OLINK` / `MEASURE` schema we serialise into.
*RCC-8* (Randell, Cui & Cohn 1992) supplies the topological calculus we
canonicalise topological cues into; *Cardinal Direction Calculus* (Frank
1991) supplies the orientation calculus for direction cues.

The "non-locations are locations too" position has antecedents in
deictic-frame work (Levinson 2003; Tenbrink 2007) and in metonymic-place
analysis (Talmy 2000), but no engineering choice in this codebase requires
those references — the position is realised here purely as a tagging policy
on top of an SpRL pipeline.

## 3. Method

### 3.1 Pipeline overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  any .txt    │ ──▶ │  src.ingest  │ ──▶ │   eval.extract_  │ ──▶ │  eval.to_graph   │
│  (novel,     │     │  clean +     │     │   corpus         │     │  cue → RCC-8/CDC │
│   memoir,    │     │  segment     │     │  zero-shot SpRL  │     │  + entity_kind   │
│   transcript)│     │              │     │  via Ollama      │     │  + top-k report  │
└──────────────┘     └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘
                            ▼                      ▼                        ▼
                  corpus/cleaned/          data/<doc>.relations    graphs/<doc>/
                    <doc>.jsonl              .jsonl                  <doc>.{graphml,
                                                                      dot, html,
                                                                      summary.json}
```

### 3.2 Stage A — Ingest (`src/ingest.py`)

Inputs: a single plain-text file. Outputs: `corpus/cleaned/<doc_id>.jsonl`.
The `doc_id` is derived from the input filename via an ASCII-fold + lowercase
slug (`slugify`). The stage:

- decodes the bytes with UTF-8, then UTF-8-with-BOM, then Latin-1 fallback;
- strips the Project Gutenberg header/footer if present (regex match; safe
  no-op on non-Gutenberg input);
- normalises line endings, control characters, smart quotes, em/en dashes,
  and ellipses;
- sentence-segments using spaCy (`en_core_web_lg`, falling back to
  `en_core_web_sm`, then a blank pipeline with `sentencizer`);
- processes in 100 kchar chunks with newline-aligned boundaries to avoid
  the spaCy memory ceiling on long documents;
- drops sentences shorter than 10 characters and collapses runs of
  identical consecutive sentences (a Project-Gutenberg formatting artefact).

### 3.3 Stage B — Extraction (`eval/extract_corpus.py`, `eval/extract.py`)

A non-overlapping window of three sentences (configurable via `--window`,
`--stride`) is sent to a local Ollama instance with the following system
prompt (`SYSTEM_PROMPT` in `eval/extract.py`):

> You are a careful annotator extracting LOCATION-LOCATION spatial
> relations from prose, following the Spatial Role Labeling (SpRL)
> annotation tradition. … Both `location_1` and `location_2` must be
> LOCATION spans — geographic places, buildings, regions, or named
> landmarks. … If the sentence contains no valid spatial relation, return
> an empty list. Do NOT invent relations. Precision matters more than
> recall.

Two design choices are deliberate:

1. The prompt frames the task as LOCATION-LOCATION but contains *no*
   negative instruction against persons, pronouns, body parts, or abstract
   concepts. We do not push the model toward such spans, but we do not
   suppress its judgement when the prose itself uses a person or a
   pronoun in a trajector or landmark slot.
2. The model is constrained to JSON output (`format: "json"`,
   `temperature: 0.1`) and bounded retries are issued on transient
   network or 5xx errors. Malformed JSON is recovered via a `{...}`
   regex fallback.

Resume safety: a partial trailing line in the output JSONL is detected
and rewritten on the next run before any append, so an interrupted
extraction never produces a malformed record.

### 3.4 Stage C — Graph compilation (`eval/to_graph.py`)

Each `(location_1, spatial_indicator, location_2)` triple becomes a
directed edge in a `networkx.MultiDiGraph`. The verbatim cue is mapped
against the YAML lexicon at `config/cue_lexicon.yaml` to assign:

- an **RCC-8** value (`QSLINK`): `DC`, `EC`, `PO`, `EQ`, `TPP`, `NTPP`,
  `TPPi`, `NTPPi`;
- a **CDC** value (`OLINK`): `N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`;
- a `MEASURE` (numeric value + unit) when a distance phrase is present.

The lexicon is patterns-only — the user can extend it without code edits.
Cues that match nothing remain on the edge as `indicator=...`, so
information is canonicalised but never lost.

### 3.5 Entity-kind tagging (`src/entity_kind.py`)

Every node is assigned one of four kinds:

| Kind            | Decision rule                                                                                                       |
|---              |---                                                                                                                  |
| `place`         | spaCy NER labels the surface span `GPE`, `LOC`, or `FAC`.                                                           |
| `deictic`       | The surface span (after stripping leading articles / determiners) is in the closed-class English deixis lexicon: 1st/2nd/3rd person pronouns, `here`, `there`, `where`, `everywhere`, `nowhere`, `this/that/the place`. |
| `person_locus`  | spaCy NER labels the surface span `PERSON`.                                                                          |
| `common_locus`  | Otherwise.                                                                                                           |

No document-specific list is consulted at any stage. The only "lists" in
the entire pipeline are (i) the closed-class English deixis lexicon (a
linguistic universal, not a corpus artefact) and (ii) the cue lexicon at
`config/cue_lexicon.yaml` (which the user owns).

### 3.6 Reports

For each run the graph builder prints, and writes to
`<name>.summary.json`:

- node and edge counts;
- the RCC-8 distribution and the CDC distribution;
- the `entity_kind` distribution;
- the **top-*k* most-connected entities** by combined in-degree + out-degree
  (default *k* = 10), each row carrying its `entity_kind`.

## 4. Quickstart

```bash
# end-to-end on a fresh document
python -m src.cli run path/to/text.txt --model mistral --svg

# explicit stages
python -m src.cli ingest  path/to/text.txt
python -m src.cli extract corpus/cleaned/<slug>.jsonl --model mistral
python -m src.cli graph   data/<slug>.relations.jsonl --name <slug> --svg
```

The `run` subcommand is a fixed-point composition of `ingest → extract →
graph`: idempotent, resumable (`--resume`), and bit-identical on the same
inputs. Outputs land in deterministic locations:

| Path                                       | Contents                                                |
|---                                         |---                                                      |
| `corpus/cleaned/<doc_id>.jsonl`            | One sentence per line, schema `{doc_id,sentence_id,text}` |
| `data/<doc_id>.relations.jsonl`            | One window per line; `LocationRelation` records         |
| `graphs/<doc_id>/<doc_id>.graphml`         | Archival multigraph (Gephi, yEd, NetworkX)              |
| `graphs/<doc_id>/<doc_id>.dot` (`.svg`)    | Graphviz source, optional rendered SVG                  |
| `graphs/<doc_id>/<doc_id>.html`            | Interactive PyVis force-directed view                   |
| `graphs/<doc_id>/<doc_id>.summary.json`    | Counts, distributions, top-*k* connectivity report      |

## 5. Results — *The Great Gatsby* as worked example

Run on the full 2 866-sentence cleaned corpus with `mistral:latest`, four
parallel workers, ~37 minutes wall-clock.

| Quantity                                  | Value      |
|---:                                       |---         |
| 3-sentence windows processed              | **955**    |
| Windows yielding ≥ 1 relation             | 171 (17.9 %) |
| Total `LocationRelation` triples          | **241**    |
| Errors / parse failures                   | 2          |
| Graph nodes                               | **326**    |
| Graph edges                               | **241**    |

### 5.1 RCC-8 distribution after cue rewriting

| RCC-8     | Count | Reading                                              |
|---        |---:   |---                                                   |
| `NTPP`    | 38    | non-tangential proper part (*"in West Egg"*)         |
| `TPP`     | 28    | tangential proper part (*"at Gatsby's house"*)       |
| `EC`      | 13    | externally connected (*"next to the Sound"*)         |
| `DC`      | 9     | disconnected (*"across the bay from"*)               |
| `PO`      | 2     | partial overlap                                      |
| `NTPPi`   | 1     | inverse NTPP (*"contains a small foul river"*)       |
| (verbatim)| 150   | cue not matched by lexicon — kept on edge            |

### 5.2 Top-10 most-connected entities

The headline empirical result of this pipeline is the following table.
Computed automatically (`<name>.summary.json`) on every run:

| Rank | Entity            | `entity_kind`   |
|---:  |---                |---              |
|  1   | I                 | `deictic`       |
|  2   | New York          | `place`         |
|  3   | Gatsby's house    | `common_locus`  |
|  4   | West Egg          | `place`         |
|  5   | here              | `deictic`       |
|  6   | Gatsby            | `person_locus`  |
|  7   | the house         | `common_locus`  |
|  8   | my house          | `common_locus`  |
|  9   | East Egg          | `place`         |
| 10   | Chicago           | `place`         |

Four of the ten most-connected `SpatialEntity`s in *The Great Gatsby* are
not canonical places. *I* is a first-person deictic centre; *here* is a
locative pro-form bound to whichever speaker is active in the surrounding
windows; *Gatsby* is metonymically a region (*"at Gatsby's"*, *"to
Gatsby's"*); *Gatsby's house* and *the house* and *my house* are common-
noun loci anchored deictically by their possessives. Filtering these out
would not improve the graph; it would erase exactly the spatial scaffolding
that the novel uses to organise the rest of its places.

## 6. Discussion — non-locations as locations under ISO-Space

ISO-Space defines a `SpatialEntity` extensionally — anything that occupies
a trajector or landmark slot in a `QSLINK` or `OLINK`. Three observations
follow from §5.2:

1. **First-person pronouns are deictic centres.** The high connectivity of
   *I* is not an extraction error; the novel is told in first person, so
   nearly every locative phrase ultimately resolves relative to the
   narrator's position. Treating *I* as a `SpatialEntity` with kind
   `deictic` makes that anchoring legible to the graph.

2. **Persons are metonymic regions.** *"At Gatsby's"* and *"with Gatsby"*
   denote (respectively) the residential region projected by Gatsby and
   the moving region co-located with him. Under ISO-Space the trajector
   is being located *at the region projected by* Gatsby; nothing in the
   schema requires the landmark to be a geographic place. Tagging such
   spans `person_locus` flags the metonymy without filtering the node.

3. **Common-noun loci dominate by mass, places dominate by named-entity
   recognisability.** *The house*, *the corner*, *the room* are
   high-frequency common-noun loci with only deictic / contextual
   anchoring. They are spatial in function but invisible to a name-based
   gazetteer; the kind tag `common_locus` distinguishes them from
   gazetteer-resolvable `place` nodes.

The graph is therefore best read as a representation of *spatial language*
in the text, not as a representation of geography. The maps in §8 are a
demonstration of what happens when one tries to push it toward geography —
the answer is "they quantify uncertainty rather than localise."

## 7. Limitations and honest scope

- **Zero-shot extraction.** No fine-tuning, no in-context examples; the
  numbers in §5 are a floor on what each backend model can do.
- **Cue lexicon is hand-curated** (now externalised to
  `config/cue_lexicon.yaml`). It covers common English topological /
  directional cues but is not exhaustive. Unmatched cues are preserved
  verbatim on the edge — the only failure mode is "not canonicalised",
  never "lost".
- **Entity-kind tagging is heuristic.** It uses a closed-class deixis
  lexicon and spaCy NER; it does not perform coreference, so two surface
  forms denoting the same locus (*"the house"* and *"Gatsby's house"*)
  appear as separate nodes. Coreference resolution is left as a downstream
  composition.
- **No ground truth.** A 100-window gold set is provisioned at
  `eval/gold/sentences_to_annotate.jsonl` with a scoring script
  (`eval/score.py`); the annotation pass is deferred. RCC-8 / CDC counts
  are descriptive output, not validated precision claims.

## 8. Downstream demonstration: probabilistic cartography

The original project goal — physically locating Fitzgerald's fictional
places on a map of Long Island — is retained in `src/phase{1..8}*.py` as
a downstream demonstration of one thing one can do with extracted SpRL
triples. It runs `emcee` MCMC over a constraint energy compiled from the
relations and emits posterior heatmaps, ensemble cartograms, and Folium
overlays in `visualizations/`. The cartographic problem is severely
under-determined by the available textual evidence (e.g., *East Egg* has
only a handful of usable spatial constraints), so the resulting heatmaps
quantify uncertainty rather than localise. The spatial graph in §5 is the
right deliverable for the underlying data; the maps are kept for
completeness.

```bash
python -m src.pipeline --config config.yaml   # full Phase 1–8 cartographic run
```

## 9. Repository layout

```
FitzTry1/
├── src/
│   ├── cli.py                 ★ single CLI entry point (ingest/extract/graph/run)
│   ├── ingest.py              ★ document-agnostic clean + segment
│   ├── entity_kind.py         ★ place / deictic / person_locus / common_locus tagger
│   ├── pipeline.py              legacy Phase 1–8 cartographic pipeline (downstream demo)
│   └── phase{1..8}*.py          legacy phases
├── eval/
│   ├── extract_corpus.py      ★ parallel SpRL extraction (Ollama)
│   ├── extract.py             ★ prompt + per-sentence extractor
│   ├── to_graph.py            ★ cue rewriter + GraphML/DOT/HTML emitter + summary
│   ├── score.py / sample_sentences.py    100-window validation harness (deferred)
│   └── gold/, predictions/, results/
├── config/
│   └── cue_lexicon.yaml       ★ user-editable RCC-8 / CDC / distance patterns
├── corpus/cleaned/<doc_id>.jsonl        cleaned-and-segmented inputs
├── data/<doc_id>.relations.jsonl        extracted LocationRelation records
├── graphs/<doc_id>/                     per-document graph artefacts
├── tests/                     66 tests
├── config.yaml                downstream cartographic config
└── requirements.txt
```

## 10. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if no CUDA

curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

brew install graphviz   # macOS;  apt-get install graphviz on Debian/Ubuntu
```

Tested on Python 3.13.6 / macOS arm64. If `blis` fails to build on Python
3.13, force binary wheels: `pip install --only-binary :all: spacy`.

## 11. References

- Frank, A. U. (1991). *Qualitative spatial reasoning about cardinal directions.* Auto-Carto 10.
- Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. (2013). *emcee: The MCMC Hammer.* PASP 125(925), 306–312.
- Kordjamshidi, P., van Otterlo, M., & Moens, M.-F. (2017). *Spatial Role Labeling Annotation Scheme.* In Ide & Pustejovsky (Eds.), *Handbook of Linguistic Annotation*. Springer.
- Levinson, S. C. (2003). *Space in Language and Cognition.* Cambridge University Press.
- Pustejovsky, J., Moszkowicz, J., & Verhagen, M. (2015). *ISO-Space: Annotating Static and Dynamic Spatial Information.* ISO/TS 24617-7.
- Randell, D. A., Cui, Z., & Cohn, A. G. (1992). *A spatial logic based on regions and connection.* KR '92.
- Talmy, L. (2000). *Toward a Cognitive Semantics, Vol. 1: Concept Structuring Systems.* MIT Press.
- Tenbrink, T. (2007). *Space, Time, and the Use of Language: An Investigation of Relationships.* De Gruyter.

---

*Built for an Independent Study in the Columbia Narrative Intelligence Lab.
Running example: F. Scott Fitzgerald, *The Great Gatsby* (1925, US public
domain since 2021). The pipeline is document-agnostic; substitute any
English prose at the `src.cli run` entry point.*
