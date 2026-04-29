"""Single CLI entry point for the spatial-graph pipeline.

Usage
-----
    # End-to-end on a brand-new document:
    python -m src.cli run path/to/novel.txt --model mistral

    # Just clean + segment (write corpus/cleaned/<slug>.jsonl):
    python -m src.cli ingest path/to/novel.txt

    # Run extraction over an already-cleaned corpus:
    python -m src.cli extract corpus/cleaned/<slug>.jsonl --model mistral

    # Compile graph + summary from an existing relations file:
    python -m src.cli graph data/<slug>.relations.jsonl

The pipeline is fully document-agnostic. The doc_id is derived from the
input filename (or `--doc-id`); no hand-curated, document-specific list
is consulted at any stage.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Subcommand: ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> Path:
    from src.ingest import ingest_file
    return ingest_file(
        input_path=Path(args.input),
        cleaned_dir=Path(args.cleaned_dir),
        doc_id=args.doc_id,
        spacy_model=args.spacy_model,
    )


# ---------------------------------------------------------------------------
# Subcommand: extract  (delegates to eval.extract_corpus.main via argv shim)
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> Path:
    from eval import extract_corpus
    argv = [
        "extract_corpus",
        "--corpus", str(args.corpus),
        "--model", args.model,
        "--host", args.host,
        "--window", str(args.window),
        "--stride", str(args.stride),
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--min-chars", str(args.min_chars),
    ]
    if args.out:
        argv += ["--out", str(args.out)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.resume:
        argv += ["--resume"]

    saved, sys.argv = sys.argv, argv
    try:
        extract_corpus.main()
    finally:
        sys.argv = saved

    if args.out:
        return Path(args.out)
    # Mirror the default that extract_corpus.main computes.
    import json
    with Path(args.corpus).open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    doc_id = first.get("doc_id") or Path(args.corpus).stem
    return Path("data") / f"{doc_id}.relations.jsonl"


# ---------------------------------------------------------------------------
# Subcommand: graph
# ---------------------------------------------------------------------------

def cmd_graph(args: argparse.Namespace) -> Path:
    from eval import to_graph
    out_dir = Path(args.out_dir) if args.out_dir else Path("graphs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "to_graph",
        "--in", str(args.relations),
        "--out-dir", str(out_dir),
        "--name", args.name,
        "--top-k", str(args.top_k),
    ]
    if args.lexicon:
        argv += ["--lexicon", str(args.lexicon)]
    if args.no_tag:
        argv += ["--no-tag"]
    if args.title:
        argv += ["--title", args.title]

    saved, sys.argv = sys.argv, argv
    try:
        to_graph.main()
    finally:
        sys.argv = saved

    base = out_dir / args.name
    # Optionally render SVG with Graphviz, if available on PATH.
    if args.svg:
        try:
            subprocess.run(
                ["dot", "-Tsvg", f"{base}.dot", "-o", f"{base}.svg"],
                check=True,
            )
            log.info("Rendered %s.svg", base)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            log.warning("dot not available or failed (%s); skipping SVG.", e)
    return out_dir


# ---------------------------------------------------------------------------
# Subcommand: run  (ingest + extract + graph)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    from src.ingest import slugify
    doc_id = args.doc_id or slugify(Path(args.input).stem)

    cleaned = cmd_ingest(argparse.Namespace(
        input=args.input,
        cleaned_dir=args.cleaned_dir,
        doc_id=doc_id,
        spacy_model=args.spacy_model,
    ))

    relations = cmd_extract(argparse.Namespace(
        corpus=cleaned,
        out=None,
        model=args.model,
        host=args.host,
        window=args.window,
        stride=args.stride,
        workers=args.workers,
        timeout=args.timeout,
        min_chars=args.min_chars,
        limit=args.limit,
        resume=args.resume,
    ))

    cmd_graph(argparse.Namespace(
        relations=relations,
        out_dir=None,
        name=doc_id,
        top_k=args.top_k,
        lexicon=args.lexicon,
        no_tag=args.no_tag,
        title=None,
        svg=args.svg,
    ))


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------

def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="mistral",
                   help="Ollama model name (e.g. mistral, mistral:latest, phi3, qwen2.5).")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--min-chars", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")


def _add_graph_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--lexicon", default=None,
                   help="Path to a cue-lexicon YAML (default: config/cue_lexicon.yaml).")
    p.add_argument("--no-tag", action="store_true",
                   help="Skip entity-kind tagging.")
    p.add_argument("--svg", action="store_true",
                   help="Also render <name>.svg via `dot` (Graphviz).")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="src.cli")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_ingest = sp.add_parser("ingest", help="Clean + segment a plain-text file.")
    p_ingest.add_argument("input", help="Path to a .txt file (any prose).")
    p_ingest.add_argument("--cleaned-dir", default="corpus/cleaned")
    p_ingest.add_argument("--doc-id", default=None,
                          help="Override the auto-derived document slug.")
    p_ingest.add_argument("--spacy-model", default="en_core_web_lg")
    p_ingest.set_defaults(func=cmd_ingest)

    p_extract = sp.add_parser("extract", help="Run zero-shot SpRL extraction.")
    p_extract.add_argument("corpus", help="Cleaned-corpus JSONL.")
    p_extract.add_argument("--out", default=None)
    _add_extract_args(p_extract)
    p_extract.set_defaults(func=cmd_extract)

    p_graph = sp.add_parser("graph", help="Build ISO-Space graph + summary.")
    p_graph.add_argument("relations", help="JSONL of LocationRelation records.")
    p_graph.add_argument("--out-dir", default=None)
    p_graph.add_argument("--name", required=True,
                         help="Base filename (and graphs/<name>/ directory).")
    p_graph.add_argument("--title", default=None)
    _add_graph_args(p_graph)
    p_graph.set_defaults(func=cmd_graph)

    p_run = sp.add_parser("run", help="Ingest + extract + graph end-to-end.")
    p_run.add_argument("input", help="Path to a .txt file (any prose).")
    p_run.add_argument("--cleaned-dir", default="corpus/cleaned")
    p_run.add_argument("--doc-id", default=None)
    p_run.add_argument("--spacy-model", default="en_core_web_lg")
    _add_extract_args(p_run)
    _add_graph_args(p_run)
    p_run.set_defaults(func=cmd_run)

    return ap


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
