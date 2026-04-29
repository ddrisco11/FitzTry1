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


# ---------------------------------------------------------------------------
# Cue lexicon -> RCC-8 / CDC mapping
#
# RCC-8 values: DC (disconnected), EC (externally connected),
#               PO (partial overlap), EQ (equal),
#               TPP / NTPP (tangential / non-tangential proper part),
#               TPPi / NTPPi (inverses).
# CDC values:   N, NE, E, SE, S, SW, W, NW, EQ.
# ---------------------------------------------------------------------------

_CUE_TO_RCC8 = [
    # (regex, RCC-8 label)
    (r"\b(inside|within|in the|in a|in)\b",            "NTPP"),
    (r"\b(contains?|houses?|holds?)\b",                "NTPPi"),
    (r"\b(part of|belongs to|member of)\b",            "TPP"),
    (r"\b(at|on)\b",                                   "TPP"),
    (r"\b(adjacent to|next to|next door|beside|by|alongside|bordering|borders|on the edge of|on the border of)\b",
                                                       "EC"),
    (r"\b(across from|opposite|facing)\b",             "DC"),
    (r"\b(near|close to|not far from|nearby)\b",       "EC"),
    (r"\b(far from|away from|distant from|miles? from|kilometers? from)\b", "DC"),
    (r"\b(beyond|past|outside)\b",                     "DC"),
    (r"\b(overlapping|crosses|crossing)\b",            "PO"),
    (r"\b(same as|identical to|equal to)\b",           "EQ"),
]

_CUE_TO_CDC = [
    (r"\bnorth(?:east|west)?\s+of\b",
        lambda m: "NE" if "east" in m.group(0).lower() else "NW" if "west" in m.group(0).lower() else "N"),
    (r"\bsouth(?:east|west)?\s+of\b",
        lambda m: "SE" if "east" in m.group(0).lower() else "SW" if "west" in m.group(0).lower() else "S"),
    (r"\beast of\b", lambda m: "E"),
    (r"\bwest of\b", lambda m: "W"),
    (r"\babove\b",   lambda m: "N"),
    (r"\bbelow\b",   lambda m: "S"),
    (r"\bleft of\b", lambda m: "W"),
    (r"\bright of\b", lambda m: "E"),
]

_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(miles?|mi|kilometers?|kms?|km|meters?|m|feet|ft|yards?|yd)\b",
    re.IGNORECASE,
)


def map_indicator(cue: str) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[str]]:
    """Return (rcc8, cdc, distance_value, distance_unit) for a free-text cue."""
    cue_l = (cue or "").lower().strip()
    rcc8: Optional[str] = None
    cdc: Optional[str] = None
    dist_val: Optional[float] = None
    dist_unit: Optional[str] = None

    for pat, label in _CUE_TO_RCC8:
        if re.search(pat, cue_l):
            rcc8 = label
            break

    for pat, fn in _CUE_TO_CDC:
        m = re.search(pat, cue_l)
        if m:
            cdc = fn(m)
            break

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

def build_graph(relations: Iterable[Dict[str, Any]]) -> nx.MultiDiGraph:
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
                g.add_node(node, iso_space_type="SpatialEntity", mention_count=0)
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

    return g


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
    for n, d in g.nodes(data=True):
        lines.append(f'  "{esc(n)}" [label="{esc(n)}\\n({d.get("mention_count", 0)})"];')
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
    for n, d in g.nodes(data=True):
        net.add_node(
            n,
            label=n,
            title=f"mentions: {d.get('mention_count', 0)}",
            value=d.get("mention_count", 1),
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
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g = build_graph(_iter_relations(in_path))
    print(f"Built graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")

    rcc_counts = defaultdict(int)
    cdc_counts = defaultdict(int)
    for _, _, d in g.edges(data=True):
        rcc_counts[d.get("rcc8") or "—"] += 1
        cdc_counts[d.get("cdc") or "—"] += 1
    print("RCC-8 distribution:", dict(rcc_counts))
    print("CDC distribution:  ", dict(cdc_counts))

    base = out_dir / args.name
    write_graphml(g, base.with_suffix(".graphml"))
    write_dot(g, base.with_suffix(".dot"))
    write_html(g, base.with_suffix(".html"),
               title=args.title or f"Spatial graph: {in_path.name}")
    print(f"Wrote {base}.graphml, {base}.dot, {base}.html")
    print(f"Render the DOT to SVG with:  dot -Tsvg {base}.dot -o {base}.svg")


if __name__ == "__main__":
    main()
