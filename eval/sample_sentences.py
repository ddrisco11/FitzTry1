"""Sample N random k-sentence windows from the cleaned corpus to seed hand annotation.

Sampling is uniformly random with a fixed seed over valid window START
positions — no content filtering, no curation. Each record concatenates k
consecutive sentences (default k=3), giving the annotator (and the models)
local context for cross-sentence relations.

Windows that overlap are allowed (independent draws); reduce --n if you
want disjoint windows.

Output schema (one JSON object per line):
    annotation_id      e.g. "ann_000"
    doc_id             source document id
    sentence_ids       list[str] of the k sentence_ids in order
    sentence           the k sentences joined with single spaces
    location_relations []  (annotator fills in)
    _notes             "" (free-text)

Usage:
    python -m eval.sample_sentences \
        --corpus corpus/cleaned/great_gatsby.jsonl \
        --out eval/gold/sentences_to_annotate.jsonl \
        --n 100 --window 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/cleaned/great_gatsby.jsonl")
    ap.add_argument("--out", default="eval/gold/sentences_to_annotate.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--window", type=int, default=3,
                    help="Number of consecutive sentences per annotation chunk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=20,
                    help="Drop near-empty sentences (chapter headers, '\"', single tokens) "
                         "below this length BEFORE windowing. Set 0 to disable.")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if args.min_chars and len(text) < args.min_chars:
                continue
            sentences.append(rec)

    n_sents = len(sentences)
    n_windows = n_sents - args.window + 1
    if n_windows < args.n:
        raise SystemExit(
            f"Corpus has {n_sents} sentences -> {n_windows} possible {args.window}-sentence windows, "
            f"but --n={args.n}. Lower --n or --window."
        )

    print(f"Loaded {n_sents} sentences; sampling {args.n} non-replacement windows of size {args.window} "
          f"from {n_windows} candidates.")

    rng = random.Random(args.seed)
    start_indices = rng.sample(range(n_windows), args.n)

    with out_path.open("w", encoding="utf-8") as f:
        for i, start in enumerate(start_indices):
            window = sentences[start : start + args.window]
            joined = " ".join(s.get("text", "").replace("\n", " ").strip() for s in window).strip()
            template = {
                "annotation_id": f"ann_{i:03d}",
                "doc_id": window[0].get("doc_id", ""),
                "sentence_ids": [s.get("sentence_id", "") for s in window],
                "sentence": joined,
                "location_relations": [],
                "_notes": "",
            }
            f.write(json.dumps(template, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} annotation templates to {out_path}")
    print(f"Each record holds {args.window} consecutive sentences joined into the `sentence` field.")
    print("Each `location_relations` list is empty — fill in triples per eval/README.md.")


if __name__ == "__main__":
    main()
