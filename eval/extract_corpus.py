"""Run zero-shot SpRL extraction over the full cleaned corpus.

Produces `data/location_relations.jsonl` in the LocationRelation schema —
one record per 3-sentence non-overlapping window:

    {
      "doc_id":      "great_gatsby",
      "window_id":   42,
      "sentence_ids": ["..._sent_126", "..._sent_127", "..._sent_128"],
      "sentence":    "<the three sentences joined>",
      "model":       "mistral:latest",
      "location_relations": [
        {"location_1": ..., "location_2": ..., "spatial_indicator": ...,
         "semantic_type": [...]}
      ]
    }

This is the single source of truth for downstream graph building; pass the
output to `eval/to_graph.py` to render ISO-Space graphs.

Usage:
    python -m eval.extract_corpus --model mistral --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from eval.extract import call_ollama, parse_relations


def load_sentences(corpus: Path, min_chars: int) -> List[Dict[str, Any]]:
    out = []
    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text", "").strip()
            if min_chars and len(text) < min_chars:
                continue
            out.append(rec)
    return out


def make_windows(sentences: List[Dict[str, Any]], window: int, stride: int) -> List[Dict[str, Any]]:
    windows = []
    for start in range(0, len(sentences) - window + 1, stride):
        chunk = sentences[start : start + window]
        joined = " ".join(s.get("text", "").replace("\n", " ").strip() for s in chunk).strip()
        windows.append({
            "window_id": len(windows),
            "doc_id": chunk[0].get("doc_id", ""),
            "sentence_ids": [s.get("sentence_id", "") for s in chunk],
            "sentence": joined,
        })
    return windows


def load_done_window_ids(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(int(rec["window_id"]))
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/cleaned/great_gatsby.jsonl")
    ap.add_argument("--out", default="data/location_relations.jsonl")
    ap.add_argument("--model", default="mistral",
                    help="Ollama model name (e.g. mistral, mistral:latest, phi3, qwen2.5)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--stride", type=int, default=3,
                    help="Sentence stride between windows (default = window, i.e. non-overlapping).")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N windows (smoke-test).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip windows already present in --out.")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(corpus, args.min_chars)
    windows = make_windows(sentences, args.window, args.stride)
    if args.limit:
        windows = windows[: args.limit]

    done = load_done_window_ids(out_path) if args.resume else set()
    if not args.resume and out_path.exists():
        out_path.unlink()  # fresh run

    todo = [w for w in windows if w["window_id"] not in done]
    print(f"Corpus: {len(sentences)} sentences -> {len(windows)} windows of size "
          f"{args.window} (stride {args.stride}). Already done: {len(done)}. "
          f"Pending: {len(todo)}. Workers: {args.workers}.")

    write_lock = Lock()
    counters = {"processed": 0, "errors": 0, "relations": 0}
    t0 = time.time()
    out_f = out_path.open("a", encoding="utf-8")

    def worker(win: Dict[str, Any]) -> Dict[str, Any]:
        try:
            raw = call_ollama(args.host, args.model, win["sentence"],
                              timeout=args.timeout)
            rels = parse_relations(raw)
            err = None
        except Exception as e:
            rels = []
            err = repr(e)
        return {
            "doc_id": win["doc_id"],
            "window_id": win["window_id"],
            "sentence_ids": win["sentence_ids"],
            "sentence": win["sentence"],
            "model": args.model,
            "location_relations": rels,
            "_error": err,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(worker, w): w for w in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with write_lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                counters["processed"] += 1
                if rec.get("_error"):
                    counters["errors"] += 1
                counters["relations"] += len(rec["location_relations"])
                p, e, r = counters["processed"], counters["errors"], counters["relations"]
                if p % 10 == 0 or p == len(todo):
                    elapsed = time.time() - t0
                    rate = p / elapsed if elapsed else 0
                    eta = (len(todo) - p) / rate if rate else 0
                    print(f"  [{p:4d}/{len(todo)}] relations={r}  errors={e}  "
                          f"elapsed={elapsed:6.1f}s  ETA={eta:5.0f}s", flush=True)

    out_f.close()
    print(f"DONE — processed={counters['processed']} relations={counters['relations']} "
          f"errors={counters['errors']} -> {out_path}")


if __name__ == "__main__":
    main()
