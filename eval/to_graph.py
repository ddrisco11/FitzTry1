"""Build ISO-Space-style spatial graphs from extracted SpRL relations.

Takes a JSONL file of relations — either:
  - eval/predictions/<model>.jsonl  (LocationRelation format from extract.py)
  - eval/gold/sentences_to_annotate.jsonl  (same schema, hand-annotated)
  - data/relations.jsonl            (legacy SpatialRelation format)

and emits a labeled directed multigraph following the ISO 24617-7 (ISO-Space)
annotation tradition:

  - Nodes      = SpatialEntity / Place
  - Edges      = QSLINK (qualitative spatial link, RCC-8 valued) or
                 OLINK  (orientation link, cardinal-direction valued)
  - Edge attrs = {rcc8, cdc, distance, indicator, sentence_id, semantic_type}

The output is written in three forms:
  - GraphML   — archival / Gephi / yEd / NetworkX-roundtrip
  - DOT       — paper figures via Graphviz (`dot -Tsvg out.dot > out.svg`)
  - HTML      — interactive PyVis force-directed graph (open in any browser)

References
----------
- Pustejovsky, J., Moszkowicz, J., & Verhagen, M. (2015). ISO-Space:
  Annotating static and dynamic spatial information. ISO/TS 24617-7.
- Randell, D. A., Cui, Z., & Cohn, A. G. (1992). A spatial logic based on
  regions and connection. KR '92.
- Frank, A. U. (1991). Qualitative spatial reasoning about cardinal
  directions. Auto-Carto 10.

Usage
-----
    python -m eval.to_graph \
        --in eval/predictions/mistral.jsonl \
        --out-dir graphs/mistral

    python -m eval.to_graph \
        --in data/relations.jsonl \
        --out-dir graphs/pipeline
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

try:
    import yaml  # PyYAML
except ImportError:  # pragma: no cover
    yaml = None

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parents[1] / "config" / "cue_lexicon.yaml"


# ---------------------------------------------------------------------------
# Cue lexicon -> RCC-8 / CDC mapping
#
# RCC-8 values: DC (disconnected), EC (externally connected),
#               PO (partial overlap), EQ (equal),
#               TPP / NTPP (tangential / non-tangential proper part),
#               TPPi / NTPPi (inverses).
# CDC values:   N, NE, E, SE, S, SW, W, NW, EQ.
# ---------------------------------------------------------------------------

# Compiled at module load from the YAML lexicon (see config/cue_lexicon.yaml).
_RCC8_PATTERNS: List[Tuple[re.Pattern, str]] = []
_CDC_PATTERNS: List[Tuple[re.Pattern, str]] = []
_DISTANCE_RE: Optional[re.Pattern] = None


def load_lexicon(path: Optional[Path] = None) -> None:
    """Load (or reload) the cue lexicon from a YAML file. If PyYAML is
    unavailable or the file is missing, fall back to a minimal embedded
    default so the pipeline still runs."""
    global _RCC8_PATTERNS, _CDC_PATTERNS, _DISTANCE_RE
    path = Path(path) if path else DEFAULT_LEXICON_PATH

    cfg: Optional[Dict[str, Any]] = None
    if yaml is not None and path.exists():
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not cfg:
        # Embedded fallback (matches config/cue_lexicon.yaml at the time of
        # writing — kept tiny to avoid drift).
        cfg = {
            "rcc8": [
                {"pattern": r"\b(inside|within|in the|in a|in)\b", "label": "NTPP"},
                {"pattern": r"\b(contains?|houses?|holds?)\b",     "label": "NTPPi"},
                {"pattern": r"\b(at|on|part of)\b",                "label": "TPP"},
                {"pattern": r"\b(near|next to|adjacent to|beside)\b", "label": "EC"},
                {"pattern": r"\b(across from|far from|beyond)\b",  "label": "DC"},
            ],
            "cdc": [
                {"pattern": r"\bnorth(?:east|west)?\s+of\b", "label": "N"},
                {"pattern": r"\bsouth(?:east|west)?\s+of\b", "label": "S"},
                {"pattern": r"\beast of\b", "label": "E"},
                {"pattern": r"\bwest of\b", "label": "W"},
            ],
            "distance": {
                "pattern": r"(\d+(?:\.\d+)?)\s*(miles?|mi|kilometers?|kms?|km|meters?|m|feet|ft|yards?|yd)\b",
            },
        }

    _RCC8_PATTERNS = [(re.compile(e["pattern"], re.IGNORECASE), e["label"])
                      for e in cfg.get("rcc8", [])]
    _CDC_PATTERNS = [(re.compile(e["pattern"], re.IGNORECASE), e["label"])
                     for e in cfg.get("cdc", [])]
    dist = cfg.get("distance") or {}
    _DISTANCE_RE = re.compile(dist.get("pattern", r"(?!)"), re.IGNORECASE)


load_lexicon()


def _refine_cdc(label: str, matched_text: str) -> str:
    """Promote N/S to NE/NW/SE/SW when the matched cue contains east/west."""
    t = matched_text.lower()
    if label == "N":
        if "east" in t: return "NE"
        if "west" in t: return "NW"
    elif label == "S":
        if "east" in t: return "SE"
        if "west" in t: return "SW"
    return label


def map_indicator(cue: str) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str]]:
    """Return (rcc8, cdc, distance_value, distance_unit) for a free-text cue."""
    cue_l = (cue or "").lower().strip()
    rcc8: Optional[str] = None
    cdc: Optional[str] = None
    dist_val: Optional[float] = None
    dist_unit: Optional[str] = None

    for pat, label in _RCC8_PATTERNS:
        if pat.search(cue_l):
            rcc8 = label
            break

    for pat, label in _CDC_PATTERNS:
        m = pat.search(cue_l)
        if m:
            cdc = _refine_cdc(label, m.group(0))
            break

    if _DISTANCE_RE is not None:
        m = _DISTANCE_RE.search(cue_l)
        if m:
            dist_val = float(m.group(1))
            dist_unit = m.group(2).lower()

    return rcc8, cdc, dist_val, dist_unit


# ---------------------------------------------------------------------------
# Input loaders — auto-detect schema
# ---------------------------------------------------------------------------

def _iter_relations(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield normalised relation dicts in the LocationRelation schema:
        location_1, location_2, spatial_indicator, semantic_type, sentence_id, source_text

    Input file MUST follow the SentenceLocationRelations schema — one JSON
    object per line with keys `sentence`, `sentence_ids` (or `sentence_id`),
    and `location_relations: [{location_1, location_2, spatial_indicator,
    semantic_type}]`. Legacy SpatialRelation files (entity_1/entity_2/type)
    are no longer accepted; re-run extraction to produce the new schema.
    """
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)

            if "location_relations" not in rec:
                raise ValueError(
                    f"{path}:{lineno} — record is not in the LocationRelation schema "
                    f"(missing `location_relations`). Re-run `eval/extract_corpus.py` "
                    f"or `eval/extract.py` to produce the new schema."
                )

            sids = rec.get("sentence_ids") or [rec.get("sentence_id", "")]
            for r in rec["location_relations"]:
                yield {
                    "location_1": r.get("location_1", ""),
                    "location_2": r.get("location_2", ""),
                    "spatial_indicator": r.get("spatial_indicator", ""),
                    "semantic_type": r.get("semantic_type", []),
                    "sentence_id": sids[0] if sids else "",
                    "source_text": rec.get("sentence", ""),
                }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(relations: Iterable[Dict[str, Any]],
                tag_entities: bool = True) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    edge_count: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for r in relations:
        loc1 = (r["location_1"] or "").strip()
        loc2 = (r["location_2"] or "").strip()
        if not loc1 or not loc2:
            continue
        cue = (r["spatial_indicator"] or "").strip()
        rcc8, cdc, dist_val, dist_unit = map_indicator(cue)

        # ISO-Space link kind: OLINK if any direction signal, else QSLINK.
        link_kind = "OLINK" if cdc else "QSLINK"

        for node in (loc1, loc2):
            if node not in g:
                g.add_node(node, iso_space_type="SpatialEntity",
                           mention_count=0, entity_kind="")
            g.nodes[node]["mention_count"] += 1

        key = (loc1, loc2, link_kind)
        edge_count[key] += 1
        g.add_edge(
            loc1, loc2,
            iso_space_link=link_kind,
            rcc8=rcc8 or "",
            cdc=cdc or "",
            distance_value=dist_val if dist_val is not None else "",
            distance_unit=dist_unit or "",
            indicator=cue,
            semantic_type=",".join(r.get("semantic_type") or []),
            sentence_id=r.get("sentence_id", ""),
        )

    if tag_entities:
        annotate_entity_kinds(g)
    return g


def annotate_entity_kinds(g: nx.MultiDiGraph) -> None:
    """Tag every node with an `entity_kind` ∈ {place, deictic, person_locus,
    common_locus}. Fully portable — no document-specific lists."""
    try:
        from src.entity_kind import classify_all
    except ImportError:
        return
    surfaces = list(g.nodes())
    kinds = classify_all(surfaces)
    for n, k in kinds.items():
        g.nodes[n]["entity_kind"] = k


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def top_connected(g: nx.MultiDiGraph, k: int = 10) -> List[Dict[str, Any]]:
    """Top-k entities by combined in+out degree on the multigraph."""
    rows: List[Dict[str, Any]] = []
    for n in g.nodes():
        rows.append({
            "entity": n,
            "degree": g.in_degree(n) + g.out_degree(n),
            "in_degree": g.in_degree(n),
            "out_degree": g.out_degree(n),
            "mention_count": g.nodes[n].get("mention_count", 0),
            "entity_kind": g.nodes[n].get("entity_kind", ""),
        })
    rows.sort(key=lambda r: (-r["degree"], -r["mention_count"], r["entity"]))
    return rows[:k]


def write_summary(g: nx.MultiDiGraph, path: Path, top_k: int = 10) -> None:
    """Write a JSON summary: counts, RCC-8/CDC distributions, entity-kind
    distribution, and the top-k most-connected entities."""
    rcc = defaultdict(int)
    cdc = defaultdict(int)
    for _, _, d in g.edges(data=True):
        rcc[d.get("rcc8") or "—"] += 1
        cdc[d.get("cdc") or "—"] += 1

    kind = defaultdict(int)
    for _, d in g.nodes(data=True):
        kind[d.get("entity_kind") or "—"] += 1

    summary = {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "rcc8_distribution": dict(rcc),
        "cdc_distribution": dict(cdc),
        "entity_kind_distribution": dict(kind),
        "top_connected": top_connected(g, k=top_k),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _edge_label(d: Dict[str, Any]) -> str:
    parts = []
    if d.get("rcc8"):
        parts.append(d["rcc8"])
    if d.get("cdc"):
        parts.append(d["cdc"])
    if d.get("distance_value") not in ("", None):
        parts.append(f"{d['distance_value']}{d.get('distance_unit', '')}")
    if not parts:
        parts.append(d.get("indicator", "") or "?")
    return " / ".join(parts)


def write_graphml(g: nx.MultiDiGraph, path: Path) -> None:
    # GraphML attribute values must be primitives — already enforced by build_graph.
    nx.write_graphml(g, path)


def write_dot(g: nx.MultiDiGraph, path: Path) -> None:
    """Write a Graphviz DOT file by hand (no pygraphviz dependency)."""
    def esc(s: Any) -> str:
        return str(s).replace('"', '\\"').replace("\n", " ")

    lines = [
        "digraph SpatialGraph {",
        '  graph [rankdir=LR, fontname="Helvetica", overlap=false, splines=true];',
        '  node  [shape=ellipse, fontname="Helvetica", style=filled, fillcolor="#eef5ff"];',
        '  edge  [fontname="Helvetica", fontsize=10];',
    ]
    kind_fill = {
        "place":         "#eef5ff",  # cool blue (canonical places)
        "deictic":       "#fff2cc",  # warm yellow (ego-deictic loci)
        "person_locus":  "#ffe6e6",  # warm pink (metonymic person-as-region)
        "common_locus":  "#eaeaea",  # neutral grey (common-noun loci)
    }
    for n, d in g.nodes(data=True):
        fill = kind_fill.get(d.get("entity_kind", ""), "#eef5ff")
        kind = d.get("entity_kind") or "—"
        lines.append(
            f'  "{esc(n)}" '
            f'[label="{esc(n)}\\n({d.get("mention_count", 0)})", '
            f'fillcolor="{fill}", tooltip="kind={esc(kind)}"];'
        )
    for u, v, d in g.edges(data=True):
        label = _edge_label(d)
        color = "#1f77b4" if d.get("iso_space_link") == "QSLINK" else "#d62728"
        lines.append(
            f'  "{esc(u)}" -> "{esc(v)}" '
            f'[label="{esc(label)}", color="{color}", tooltip="{esc(d.get("indicator", ""))}"];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_html(g: nx.MultiDiGraph, path: Path, title: str = "Spatial graph") -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        print("pyvis not installed; skipping HTML output.")
        return
    net = Network(height="800px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut()
    kind_color = {
        "place":         "#4a90e2",
        "deictic":       "#f5a623",
        "person_locus":  "#d0021b",
        "common_locus":  "#9b9b9b",
    }
    for n, d in g.nodes(data=True):
        kind = d.get("entity_kind") or "—"
        net.add_node(
            n,
            label=n,
            title=f"mentions: {d.get('mention_count', 0)}<br>kind: {kind}",
            value=d.get("mention_count", 1),
            color=kind_color.get(kind, "#4a90e2"),
        )
    for u, v, d in g.edges(data=True):
        label = _edge_label(d)
        color = "#1f77b4" if d.get("iso_space_link") == "QSLINK" else "#d62728"
        title_html = (
            f"link={d.get('iso_space_link')}<br>"
            f"rcc8={d.get('rcc8') or '—'}<br>"
            f"cdc={d.get('cdc') or '—'}<br>"
            f"indicator={d.get('indicator', '')}<br>"
            f"sentence_id={d.get('sentence_id', '')}"
        )
        net.add_edge(u, v, label=label, title=title_html, color=color)
    html = net.generate_html(notebook=False)
    path.write_text(f"<!-- {title} -->\n" + html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", required=True,
                    help="JSONL of relations (eval/predictions/*.jsonl, eval/gold/*.jsonl, "
                         "or data/relations.jsonl).")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write graph.graphml / graph.dot / graph.html.")
    ap.add_argument("--name", default="graph",
                    help="Base filename for outputs (default: 'graph').")
    ap.add_argument("--title", default=None,
                    help="Title shown in the interactive HTML (default: derived from input).")
    ap.add_argument("--lexicon", default=None,
                    help=f"Path to a cue-lexicon YAML (default: {DEFAULT_LEXICON_PATH}).")
    ap.add_argument("--top-k", type=int, default=10,
                    help="How many top-connected entities to report (default: 10).")
    ap.add_argument("--no-tag", action="store_true",
                    help="Skip entity-kind tagging (place / deictic / person_locus / common_locus).")
    args = ap.parse_args()

    if args.lexicon:
        load_lexicon(Path(args.lexicon))

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = build_graph(_iter_relations(in_path), tag_entities=not args.no_tag)
    print(f"Built graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

    rcc_counts = defaultdict(int)
    cdc_counts = defaultdict(int)
    for _, _, d in g.edges(data=True):
        rcc_counts[d.get("rcc8") or "—"] += 1
        cdc_counts[d.get("cdc") or "—"] += 1
    print("RCC-8 distribution:", dict(rcc_counts))
    print("CDC distribution:  ", dict(cdc_counts))

    if not args.no_tag:
        kind_counts = defaultdict(int)
        for _, d in g.nodes(data=True):
            kind_counts[d.get("entity_kind") or "—"] += 1
        print("Entity-kind distribution:", dict(kind_counts))

    top = top_connected(g, k=args.top_k)
    print(f"\nTop-{args.top_k} most-connected entities:")
    print(f"  {'rank':>4}  {'degree':>6}  {'kind':<14}  entity")
    for i, row in enumerate(top, 1):
        print(f"  {i:>4}  {row['degree']:>6}  {row['entity_kind']:<14}  {row['entity']}")

    base = out_dir / args.name
    write_graphml(g, base.with_suffix(".graphml"))
    write_dot(g, base.with_suffix(".dot"))
    write_html(g, base.with_suffix(".html"),
               title=args.title or f"Spatial graph: {in_path.name}")
    write_summary(g, base.with_suffix(".summary.json"), top_k=args.top_k)
    print(f"\nWrote {base}.graphml, {base}.dot, {base}.html, {base}.summary.json")
    print(f"Render the DOT to SVG with:  dot -Tsvg {base}.dot -o {base}.svg")


if __name__ == "__main__":
    main()
