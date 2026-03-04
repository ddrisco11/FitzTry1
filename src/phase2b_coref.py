"""Phase 2b — Coreference Resolution (CoReNer).

Uses the aiola/roberta-large-corener multi-task model to resolve coreference
chains across the cleaned corpus, then propagates known entity names to
coreferent spans.  This increases the number of sentences where named entities
co-occur, directly improving spatial relation extraction in phase 4.

New mentions are appended to entities.jsonl with source_text indicating
the coref resolution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from src.utils.io import read_jsonl, read_json, write_jsonl, iter_jsonl, data_dir
from src.utils.schemas import Entity, EntityMention, SentenceRecord

log = logging.getLogger(__name__)

# RoBERTa-large context window is 512 subword tokens.
# Leave headroom for special tokens and span enumeration overhead.
_MAX_WINDOW_TOKENS = 400
_WINDOW_OVERLAP_SENTS = 5


# ---------------------------------------------------------------------------
# CoReNer model management
# ---------------------------------------------------------------------------

def _load_corener(model_name: str):
    """Load CoReNer model and tokenizer."""
    from transformers import AutoTokenizer
    from corener.models import Corener

    log.info("Loading CoReNer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Corener.from_pretrained(model_name)
    model.eval()
    log.info("CoReNer model loaded.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Entity matching helpers
# ---------------------------------------------------------------------------

def _build_entity_name_index(
    entities: List[Entity],
) -> Dict[str, str]:
    """Build lookup: lowercased canonical name → canonical name."""
    exact_map: Dict[str, str] = {}
    for e in entities:
        exact_map[e.canonical_name.lower()] = e.canonical_name
    return exact_map


def _match_mention_to_entity(
    mention_text: str,
    exact_map: Dict[str, str],
) -> Optional[str]:
    """Match a coref mention string to a known entity. Returns canonical_name or None."""
    lower = mention_text.strip().lower()

    if lower in exact_map:
        return exact_map[lower]

    for prefix in ("the ", "a ", "an "):
        if lower.startswith(prefix):
            rest = lower[len(prefix):]
            if rest in exact_map:
                return exact_map[rest]

    for key, canon in exact_map.items():
        if len(key) >= 4 and key in lower:
            return canon

    return None


# ---------------------------------------------------------------------------
# Token-to-character mapping
# ---------------------------------------------------------------------------

def _map_tokens_to_chars(tokens: List[str], text: str) -> List[Tuple[int, int]]:
    """Map word-level tokens back to (char_start, char_end) in the original text."""
    offsets = []
    pos = 0
    for token in tokens:
        idx = text.find(token, pos)
        if idx < 0:
            while pos < len(text) and text[pos] == " ":
                pos += 1
            offsets.append((pos, pos + len(token)))
            pos += len(token)
        else:
            offsets.append((idx, idx + len(token)))
            pos = idx + len(token)
    return offsets


def _char_offset_to_sentence(
    char_offset: int,
    sentence_boundaries: List[Tuple[int, int, str]],
) -> Optional[Tuple[str, int]]:
    """Map a character offset in concatenated window text back to a
    sentence_id and local char offset within that sentence."""
    for start, end, sent_id in sentence_boundaries:
        if start <= char_offset < end:
            return sent_id, char_offset - start
    return None


# ---------------------------------------------------------------------------
# Window building
# ---------------------------------------------------------------------------

def _build_windows(
    sentences: List[SentenceRecord],
    tokenizer,
    max_tokens: int = _MAX_WINDOW_TOKENS,
    overlap: int = _WINDOW_OVERLAP_SENTS,
) -> List[List[SentenceRecord]]:
    """Build overlapping windows of sentences respecting the subword token budget."""
    windows: List[List[SentenceRecord]] = []
    i = 0
    while i < len(sentences):
        window: List[SentenceRecord] = []
        token_count = 0
        j = i
        while j < len(sentences):
            n_tokens = len(tokenizer.encode(sentences[j].text, add_special_tokens=False))
            if token_count + n_tokens > max_tokens and window:
                break
            window.append(sentences[j])
            token_count += n_tokens
            j += 1
        windows.append(window)
        step = max(1, len(window) - overlap)
        i += step
        if j >= len(sentences):
            break
    return windows


# ---------------------------------------------------------------------------
# Window processing
# ---------------------------------------------------------------------------

def _process_window(
    model,
    tokenizer,
    sentences: List[SentenceRecord],
    exact_map: Dict[str, str],
    existing_mention_keys: Set[Tuple[str, str]],
) -> List[Tuple[str, EntityMention]]:
    """Run CoReNer coref on a window of sentences and return new mentions."""
    from corener.data import MTLDataset
    from corener.utils.prediction import convert_model_output

    parts: List[str] = []
    boundaries: List[Tuple[int, int, str]] = []
    offset = 0
    for sent in sentences:
        text = sent.text.replace("\n", " ")
        start = offset
        parts.append(text)
        offset += len(text) + 1  # +1 for space separator
        boundaries.append((start, start + len(text), sent.sentence_id))

    full_text = " ".join(parts)

    dataset = MTLDataset(
        types=model.config.types,
        tokenizer=tokenizer,
        train_mode=False,
    )
    dataset.read_dataset([full_text])

    try:
        example = dataset.get_example(0)
    except Exception:
        log.debug("Skipping window (tokenization issue)")
        return []

    if example.encodings.shape[-1] > 512:
        log.debug("Skipping window (exceeds 512-token limit: %d)", example.encodings.shape[-1])
        return []

    try:
        with torch.no_grad():
            output = model(
                input_ids=example.encodings,
                context_masks=example.context_masks,
                entity_masks=example.entity_masks,
                entity_sizes=example.entity_sizes,
                entity_spans=example.entity_spans,
                entity_sample_masks=example.entity_sample_masks,
                inference=True,
            )
    except RuntimeError as e:
        log.debug("Skipping window (runtime error: %s)", e)
        return []

    parsed = convert_model_output(output=output, batch=example, dataset=dataset)
    if not parsed:
        return []

    result = parsed[0] if isinstance(parsed, list) else parsed
    tokens = result.get("tokens", [])
    clusters = result.get("clusters", [])

    if not tokens or not clusters:
        return []

    char_offsets = _map_tokens_to_chars(tokens, full_text)
    new_mentions: List[Tuple[str, EntityMention]] = []

    for cluster in clusters:
        matched_entity: Optional[str] = None
        for mention in cluster:
            mention_text = " ".join(mention["span"])
            canon = _match_mention_to_entity(mention_text, exact_map)
            if canon is not None:
                matched_entity = canon
                break

        if matched_entity is None:
            continue

        for mention in cluster:
            mention_text = " ".join(mention["span"])
            start_tok = mention["start"]
            end_tok = mention["end"]

            if start_tok >= len(char_offsets) or end_tok > len(char_offsets):
                continue

            char_start_abs = char_offsets[start_tok][0]
            char_end_abs = char_offsets[end_tok - 1][1]

            loc = _char_offset_to_sentence(char_start_abs, boundaries)
            if loc is None:
                continue
            sent_id, local_start = loc

            key = (matched_entity, sent_id)
            if key in existing_mention_keys:
                continue
            existing_mention_keys.add(key)

            if mention_text.strip().lower() == matched_entity.lower():
                continue

            local_end = local_start + (char_end_abs - char_start_abs)

            new_mentions.append((
                matched_entity,
                EntityMention(
                    sentence_id=sent_id,
                    char_start=local_start,
                    char_end=local_end,
                    source_text=f"[coref→{matched_entity}] ...{mention_text}...",
                ),
            ))

    return new_mentions


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 2b: Coreference Resolution (CoReNer) ===")

    if cfg.get("ner", {}).get("skip_phase2b_coref", False):
        log.info("Skipping Phase 2b (skip_phase2b_coref=true in config)")
        return

    dd = data_dir(cfg)
    entities_path = dd / "entities.jsonl"

    if not entities_path.exists():
        log.error("entities.jsonl not found — run phase 2 first.")
        return

    entities: List[Entity] = read_jsonl(entities_path, model=Entity)
    original_mention_count = sum(e.mention_count for e in entities)

    has_coref = any(
        "[coref" in m.source_text
        for e in entities
        for m in e.mentions
    )
    if has_coref and not force:
        log.info("Coref mentions already present in entities.jsonl, skipping (use --force).")
        return

    if has_coref:
        for e in entities:
            e.mentions = [m for m in e.mentions if "[coref" not in m.source_text]
            e.mention_count = len(e.mentions)
        original_mention_count = sum(e.mention_count for e in entities)

    exact_map = _build_entity_name_index(entities)
    entity_by_name: Dict[str, Entity] = {e.canonical_name: e for e in entities}

    existing_keys: Set[Tuple[str, str]] = set()
    for e in entities:
        for m in e.mentions:
            existing_keys.add((e.canonical_name, m.sentence_id))

    # Load CoReNer model
    model_name = cfg["ner"].get("corener_model", "aiola/roberta-large-corener")
    log.info("Loading CoReNer model for coreference resolution...")
    model, tokenizer = _load_corener(model_name)
    log.info("CoReNer model loaded.")

    # Read corpus
    cleaned_dir = Path(cfg["corpus"]["cleaned_dir"])
    doc_filter = set(cfg["corpus"].get("doc_filter", []))
    metadata_path = Path(cfg["corpus"]["metadata_file"])
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        doc_ids = [m["doc_id"] for m in metadata if not doc_filter or m["doc_id"] in doc_filter]
    else:
        doc_ids = []

    if doc_filter and not doc_ids:
        for did in doc_filter:
            if (cleaned_dir / f"{did}.jsonl").exists():
                doc_ids.append(did)

    total_new_mentions = 0

    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found: %s", cleaned_path)
            continue

        sentences = list(iter_jsonl(cleaned_path, model=SentenceRecord))
        log.info("Running coref on doc '%s' (%d sentences)", doc_id, len(sentences))

        windows = _build_windows(sentences, tokenizer)
        log.info("  Split into %d overlapping windows", len(windows))

        doc_new = 0
        for wi, window in enumerate(windows):
            if wi > 0 and wi % 25 == 0:
                log.info("  Processed %d/%d windows", wi, len(windows))
            new_mentions = _process_window(
                model, tokenizer, window, exact_map, existing_keys
            )

            for canon_name, mention in new_mentions:
                entity = entity_by_name.get(canon_name)
                if entity is None:
                    continue
                entity.mentions.append(mention)
                entity.mention_count = len(entity.mentions)
                doc_new += 1

        total_new_mentions += doc_new
        log.info("  Added %d coref mentions for doc '%s'", doc_new, doc_id)

    write_jsonl(entities_path, entities, overwrite=True)

    new_total = sum(e.mention_count for e in entities)
    log.info(
        "Phase 2b complete. Mentions: %d → %d (+%d from coref). Written to %s",
        original_mention_count, new_total, total_new_mentions, entities_path,
    )

    for e in entities:
        coref_count = sum(1 for m in e.mentions if "[coref" in m.source_text)
        if coref_count > 0:
            log.info(
                "  %s: %d original + %d coref = %d total mentions",
                e.canonical_name, e.mention_count - coref_count, coref_count, e.mention_count,
            )
