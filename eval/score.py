"""Score model SpRL predictions against gold annotations.

Matching schemes
----------------
- strict:  predicted triple matches gold iff (location_1, location_2,
           spatial_indicator) all string-match (case- and whitespace-
           normalised, exact).
- lenient: token-set overlap >= 0.5 on each of the three slots.

A predicted relation is a TP if it matches some unmatched gold relation in
the same sentence. Unmatched gold = FN. Unmatched predicted = FP.

Outputs
-------
- eval/results/scores.json     — full numeric breakdown
- eval/results/scores.md       — markdown table for the README
- eval/results/scores.png      — bar chart of F1 across models
- eval/results/per_relation.md — per-semantic-type breakdown

Usage
-----
    python -m eval.score \
        --gold eval/gold/sentences_to_annotate.jsonl \
        --pred eval/predictions/mistral.jsonl \
              eval/predictions/phi3.jsonl \
              eval/predictions/qwen2.5.jsonl \
        --out-dir eval/results
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS_RE.sub(" ", (s or "").strip().lower())


def tokens(s: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", norm(s)))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def triple_match_strict(p: Dict[str, Any], g: Dict[str, Any]) -> bool:
    return (
        norm(p["location_1"]) == norm(g["location_1"])
        and norm(p["location_2"]) == norm(g["location_2"])
        and norm(p["spatial_indicator"]) == norm(g["spatial_indicator"])
    )


def triple_match_lenient(p: Dict[str, Any], g: Dict[str, Any], thresh: float = 0.5) -> bool:
    return (
        jaccard(tokens(p["location_1"]), tokens(g["location_1"])) >= thresh
        and jaccard(tokens(p["location_2"]), tokens(g["location_2"])) >= thresh
        and jaccard(tokens(p["spatial_indicator"]), tokens(g["spatial_indicator"])) >= thresh
    )


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_one(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]],
              matcher) -> Dict[str, Any]:
    """Score a model's predictions against the gold set, joining on annotation_id."""
    gold_by_id = {g["annotation_id"]: g for g in gold}
    pred_by_id = {p["annotation_id"]: p for p in pred}

    tp = fp = fn = 0
    per_type_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    sentence_level = {"correct_empty_tp": 0, "missed_empty": 0, "halluc_on_empty": 0}

    for ann_id, g_rec in gold_by_id.items():
        g_rels = list(g_rec.get("location_relations", []))
        p_rels = list(pred_by_id.get(ann_id, {}).get("location_relations", []))

        if not g_rels and not p_rels:
            sentence_level["correct_empty_tp"] += 1
        elif not g_rels and p_rels:
            sentence_level["halluc_on_empty"] += 1
        elif g_rels and not p_rels:
            sentence_level["missed_empty"] += 1

        matched_g = [False] * len(g_rels)
        for p in p_rels:
            hit = -1
            for j, g in enumerate(g_rels):
                if matched_g[j]:
                    continue
                if matcher(p, g):
                    hit = j
                    break
            if hit >= 0:
                matched_g[hit] = True
                tp += 1
                for st in g_rels[hit].get("semantic_type", []) or ["UNTYPED"]:
                    per_type_counts[st]["tp"] += 1
            else:
                fp += 1
                for st in (p.get("semantic_type") or ["UNTYPED"]):
                    per_type_counts[st]["fp"] += 1
        for j, m in enumerate(matched_g):
            if not m:
                fn += 1
                for st in g_rels[j].get("semantic_type", []) or ["UNTYPED"]:
                    per_type_counts[st]["fn"] += 1

    p, r, f = prf(tp, fp, fn)
    per_type = {
        st: dict(zip(("precision", "recall", "f1"),
                     prf(c["tp"], c["fp"], c["fn"])),
                 **c)
        for st, c in per_type_counts.items()
    }
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f,
        "per_semantic_type": per_type,
        "sentence_level": sentence_level,
    }


def render_md(results: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    lines.append("# SpRL extraction — model comparison\n")
    lines.append("## Overall (triple-level)\n")
    lines.append("| model | scheme | precision | recall | F1 | TP | FP | FN |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for tag, r in results.items():
        for scheme in ("strict", "lenient"):
            s = r[scheme]
            lines.append(
                f"| {tag} | {scheme} | {s['precision']:.3f} | {s['recall']:.3f} | "
                f"{s['f1']:.3f} | {s['tp']} | {s['fp']} | {s['fn']} |"
            )
    lines.append("\n## Sentence-level behaviour (lenient)\n")
    lines.append("| model | empty-correct | missed-on-non-empty | hallucinated-on-empty |")
    lines.append("|---|---:|---:|---:|")
    for tag, r in results.items():
        s = r["lenient"]["sentence_level"]
        lines.append(
            f"| {tag} | {s['correct_empty_tp']} | {s['missed_empty']} | {s['halluc_on_empty']} |"
        )
    return "\n".join(lines) + "\n"


def render_per_type_md(results: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# Per-semantic-type breakdown (lenient)\n",
             "| model | type | precision | recall | F1 | TP | FP | FN |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for tag, r in results.items():
        per_type = r["lenient"]["per_semantic_type"]
        for st, s in sorted(per_type.items()):
            lines.append(
                f"| {tag} | {st} | {s['precision']:.3f} | {s['recall']:.3f} | "
                f"{s['f1']:.3f} | {s['tp']} | {s['fp']} | {s['fn']} |"
            )
    return "\n".join(lines) + "\n"


def render_chart(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return
    tags = list(results.keys())
    strict_f1 = [results[t]["strict"]["f1"] for t in tags]
    lenient_f1 = [results[t]["lenient"]["f1"] for t in tags]
    x = np.arange(len(tags))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, strict_f1, w, label="strict")
    ax.bar(x + w / 2, lenient_f1, w, label="lenient")
    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    ax.set_ylabel("Triple-level F1")
    ax.set_ylim(0, 1)
    ax.set_title("SpRL extraction F1 — Gatsby-100 benchmark")
    ax.legend()
    for i, (s, l) in enumerate(zip(strict_f1, lenient_f1)):
        ax.text(i - w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, l + 0.01, f"{l:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold/sentences_to_annotate.jsonl")
    ap.add_argument("--pred", nargs="+", required=True,
                    help="One or more eval/predictions/<tag>.jsonl files.")
    ap.add_argument("--out-dir", default="eval/results")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_jsonl(Path(args.gold))
    n_gold_rels = sum(len(g.get("location_relations", [])) for g in gold)
    n_gold_empty = sum(1 for g in gold if not g.get("location_relations"))
    print(f"Gold: {len(gold)} sentences, {n_gold_rels} relations, {n_gold_empty} empty.")

    results: Dict[str, Dict[str, Any]] = {}
    for path_str in args.pred:
        path = Path(path_str)
        tag = path.stem
        pred = load_jsonl(path)
        results[tag] = {
            "strict": score_one(gold, pred, triple_match_strict),
            "lenient": score_one(gold, pred, triple_match_lenient),
            "n_predictions": sum(len(p.get("location_relations", [])) for p in pred),
        }
        s = results[tag]["lenient"]
        print(f"  {tag}: lenient P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")

    (out_dir / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "scores.md").write_text(render_md(results), encoding="utf-8")
    (out_dir / "per_relation.md").write_text(render_per_type_md(results), encoding="utf-8")
    render_chart(results, out_dir / "scores.png")
    print(f"Wrote results to {out_dir}/")


if __name__ == "__main__":
    main()
