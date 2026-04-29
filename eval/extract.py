"""Run a small open LLM via Ollama as a zero-shot SpRL extractor.

Reads `eval/gold/sentences_to_annotate.jsonl`, sends each sentence to the
specified Ollama model, parses the JSON response into LocationRelation
records, and writes `eval/predictions/<model_tag>.jsonl` with the same
record IDs as the gold file (so scoring is a simple inner join).

Tested with: mistral, phi3, qwen2.5 (any chat-instruct Ollama model works).

Usage:
    python -m eval.extract --model mistral --tag mistral
    python -m eval.extract --model phi3    --tag phi3
    python -m eval.extract --model qwen2.5 --tag qwen2.5

    # Run all three:
    for m in mistral phi3 qwen2.5; do
        python -m eval.extract --model "$m" --tag "$m"
    done
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests


SYSTEM_PROMPT = """\
You are a careful annotator extracting LOCATION-LOCATION spatial relations from
literary prose, following the Spatial Role Labeling (SpRL) annotation tradition.

A relation is a triple (location_1, spatial_indicator, location_2) where:
  - location_1 (trajector): the entity being located.
  - location_2 (landmark):  the reference entity it is located relative to.
  - spatial_indicator:      the verbatim word or phrase from the sentence
                            expressing the spatial relationship
                            (e.g. "across the bay from", "twenty miles from",
                            "north of", "in", "near").

BOTH location_1 and location_2 must be LOCATION spans — geographic places,
buildings, regions, or named landmarks. Do NOT extract relations where one
argument is a person, a body part, a piece of furniture, an abstract concept,
or a temporal phrase.

Each relation also carries a semantic_type list, a subset of:
  ["REGION", "DIRECTION", "DISTANCE"]

If the sentence contains no valid LOCATION-LOCATION relation, return an empty
list. Do NOT invent relations. Precision matters more than recall.

OUTPUT FORMAT — return a single JSON object exactly like:
{
  "location_relations": [
    {
      "location_1": "<trajector text>",
      "location_2": "<landmark text>",
      "spatial_indicator": "<verbatim cue>",
      "semantic_type": ["REGION"]
    }
  ]
}
Return ONLY this JSON object. No prose, no markdown fences.
"""

USER_TEMPLATE = """\
Sentence:
\"\"\"{sentence}\"\"\"

Extract the location-location spatial relations.
"""


def call_ollama(host: str, model: str, sentence: str, timeout: int = 90,
                temperature: float = 0.1) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sentence=sentence)},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "top_p": 0.9, "num_predict": 512},
    }
    r = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_relations(raw: str) -> List[Dict[str, Any]]:
    """Robust JSON parse — fall back to first {...} block on prose drift."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    rels = obj.get("location_relations", []) if isinstance(obj, dict) else []
    cleaned: List[Dict[str, Any]] = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        loc1 = (r.get("location_1") or "").strip()
        loc2 = (r.get("location_2") or "").strip()
        cue = (r.get("spatial_indicator") or "").strip()
        if not (loc1 and loc2 and cue):
            continue
        sem = r.get("semantic_type", [])
        if isinstance(sem, str):
            sem = [sem]
        sem = [s.upper() for s in sem if isinstance(s, str)]
        cleaned.append({
            "location_1": loc1,
            "location_2": loc2,
            "spatial_indicator": cue,
            "semantic_type": sem,
        })
    return cleaned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="eval/gold/sentences_to_annotate.jsonl",
                    help="Input file. Only the `sentence` field is read; gold relations are ignored.")
    ap.add_argument("--out-dir", default="eval/predictions")
    ap.add_argument("--model", required=True,
                    help="Ollama model name (e.g. mistral, phi3, qwen2.5)")
    ap.add_argument("--tag", default=None,
                    help="Filename tag for outputs. Defaults to --model with ':' replaced by '_'.")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N sentences (for smoke tests).")
    args = ap.parse_args()

    tag = args.tag or args.model.replace(":", "_").replace("/", "_")
    out_path = Path(args.out_dir) / f"{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs: List[Dict[str, Any]] = []
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            inputs.append(json.loads(line))
    if args.limit:
        inputs = inputs[: args.limit]

    print(f"[{tag}] running model={args.model} on {len(inputs)} sentences -> {out_path}")
    t0 = time.time()
    n_relations = 0
    n_errors = 0

    with out_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(inputs):
            sentence = rec["sentence"]
            try:
                raw = call_ollama(args.host, args.model, sentence, timeout=args.timeout)
                rels = parse_relations(raw)
            except Exception as e:
                rels = []
                n_errors += 1
                print(f"  [{i:03d}] ERROR: {e}", file=sys.stderr)

            n_relations += len(rels)
            out.write(json.dumps({
                "annotation_id": rec["annotation_id"],
                "sentence_id": rec.get("sentence_id", ""),
                "sentence": sentence,
                "model": args.model,
                "location_relations": rels,
            }, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:03d}/{len(inputs)}] {n_relations} relations so far "
                      f"({elapsed:.1f}s elapsed, {n_errors} errors)")

    elapsed = time.time() - t0
    print(f"[{tag}] DONE — {n_relations} relations, {n_errors} errors, {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
