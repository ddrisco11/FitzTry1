"""Sample N random sentences from the cleaned corpus to seed hand annotation.

Sampling is uniformly random with a fixed seed — no content filtering.
Sentences with no spatial relations are valid (and expected) gold examples;
they exercise model precision (false positives).

Usage:
    python -m eval.sample_sentences \
        --corpus corpus/cleaned/great_gatsby.jsonl \
        --out eval/gold/sentences_to_annotate.jsonl \
        --n 100 --seed 42
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=20,
                    help="Drop near-empty sentences (chapter headers, '\"', single tokens) "
                         "below this length. Set 0 to disable.")
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

    print(f"Loaded {len(sentences)} sentences (after min-chars filter); sampling {args.n}.")

    rng = random.Random(args.seed)
    sample = rng.sample(sentences, args.n)

    with out_path.open("w", encoding="utf-8") as f:
        for i, rec in enumerate(sample):
            template = {
                "annotation_id": f"ann_{i:03d}",
                "doc_id": rec.get("doc_id", ""),
                "sentence_id": rec.get("sentence_id", ""),
                "sentence": rec.get("text", "").replace("\n", " ").strip(),
                "location_relations": [],
                "_notes": "",
            }
            f.write(json.dumps(template, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} annotation templates to {out_path}")
    print("Each record has an empty `location_relations` list — fill in triples per the schema in eval/README.md.")


if __name__ == "__main__":
    main()
