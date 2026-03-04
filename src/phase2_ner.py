"""Phase 2 — Named Entity Recognition (CoReNer).

Extracts geographic entities using the aiola/roberta-large-corener multi-task
model. Processes text in multi-sentence chunks (up to ~450 subword tokens) to
give the model the cross-sentence context it needs for accurate NER, and to
enable within-chunk coreference. Coref clusters are also extracted here;
phase 2b adds cross-chunk coref from overlapping windows.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

from src.utils.io import iter_jsonl, read_json, write_jsonl, data_dir
from src.utils.schemas import Entity, EntityMention, SentenceRecord

log = logging.getLogger(__name__)

# Canonical fictional Fitzgerald places (always classified as fictional)
KNOWN_FICTIONAL: Set[str] = {
    "East Egg",
    "West Egg",
    "Valley of Ashes",
    "Eggs",
}

ENTITY_BLOCKLIST: Set[str] = {
    "Jordan", "Myrtle", "Chester", "Daisy", "Daisy's", "Gatsby",
    "Gatsby's", "Tom", "Nick", "Wilson", "Wolfsheim", "Eckhaust",
    "Katspaugh", "the De Jongs", "Kellehers", "Buchanan", "Baker",
    "Meyer", "Klipspringer",
    "the Gasoline Pump", "the enchanted metropolitan twilight",
    "Georgian Colonial", "Baedeker", "Beluga",
    "colocasie", "syphiliticos morbos", "Iviartea", "Jriartea",
    "Paullinia", "Piscidia Evythrina", "Bignonia chica",
    "Wullschlaegelia", "Rutacea", "Siphonia elastica",
    "the Siphonia elastica", "Bothrops", "pomatum",
    "Kintinc AN ALLIGATOR", "THE WILD INDIAN",
    "East", "the Middle West", "the fresher sea",
    "milreis",
}

SURFACE_NORM: Dict[str, str] = {
    "N.Y.": "New York",
    "N.Y": "New York",
    "NYC": "New York City",
    "N.Y.C.": "New York City",
    "L.I.": "Long Island",
    "L.I": "Long Island",
}


def normalize_surface(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return SURFACE_NORM.get(normalized, normalized)


def _make_entity_id(idx: int) -> str:
    return f"e_{idx:04d}"


def _is_blocked(name: str, blocklist: Set[str]) -> bool:
    return name in blocklist


def _looks_like_ocr_artifact(name: str) -> bool:
    if len(name) < 3:
        return True
    if re.search(r"\d", name) and not re.match(r"^\d+\s*(st|nd|rd|th)\s+", name, re.I):
        return True
    if sum(1 for c in name if c in "&@#$%^*+=<>{}[]|\\~`") > 0:
        return True
    return False


def _is_non_geographic(name: str) -> bool:
    stripped = name.strip()
    lower = stripped.lower()
    if len(lower) <= 2:
        return True
    if lower.endswith("'s") or lower.endswith("\u2019s"):
        return True
    if " " not in stripped and stripped[0].islower():
        return True
    return False


# ---------------------------------------------------------------------------
# CoReNer helpers
# ---------------------------------------------------------------------------

def _load_corener(model_name: str):
    from transformers import AutoTokenizer
    from corener.models import Corener
    log.info("Loading CoReNer model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Corener.from_pretrained(model_name)
    model.eval()
    log.info("CoReNer model loaded.")
    return model, tokenizer


def _map_tokens_to_chars(tokens: List[str], text: str) -> List[Tuple[int, int]]:
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


def _build_chunks(
    sentences: List[SentenceRecord],
    tokenizer,
    max_tokens: int = 380,
) -> List[Tuple[str, List[SentenceRecord], List[Tuple[int, int, str]]]]:
    """Build overlapping chunks of sentences that fit within max_tokens.
    CoReNer uses spacy+tokenizer which can produce more tokens than
    tokenizer.encode(plain_text); use conservative limit (512 - headroom).
    Returns list of (chunk_text, sentences_in_chunk, boundaries).
    boundaries: (char_start, char_end, sentence_id) for each sentence.
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents: List[SentenceRecord] = []
        chunk_parts: List[str] = []
        offset = 0
        boundaries: List[Tuple[int, int, str]] = []

        while i < len(sentences):
            sent = sentences[i]
            new_text = sent.text.replace("\n", " ")
            trial_parts = chunk_parts + [new_text]
            trial_text = " ".join(trial_parts)
            n_tokens = len(tokenizer.encode(trial_text, add_special_tokens=True))

            if n_tokens > max_tokens and chunk_sents:
                break

            start = offset
            chunk_parts.append(new_text)
            chunk_sents.append(sent)
            offset += len(new_text) + 1
            boundaries.append((start, start + len(new_text), sent.sentence_id))
            i += 1

        if chunk_sents:
            chunk_text = " ".join(chunk_parts)
            chunks.append((chunk_text, chunk_sents, boundaries))

    return chunks


def _char_offset_to_span(
    char_start: int,
    char_end: int,
    boundaries: List[Tuple[int, int, str]],
) -> Tuple[str, int, int]:
    """Map char span in chunk to (sentence_id, local_start, local_end)."""
    for start, end, sent_id in boundaries:
        if start <= char_start < end:
            local_start = char_start - start
            local_end = min(char_end, end) - start
            return sent_id, local_start, local_end
    if boundaries:
        start, end, sent_id = boundaries[-1][0], boundaries[-1][1], boundaries[-1][2]
        return sent_id, char_start - start, min(char_end, end) - start
    raise ValueError("Empty boundaries")


def _classify_entity(name: str, known_fictional: Set[str], _: float) -> str:
    if name in known_fictional:
        return "fictional"
    real_indicators = {
        "New York", "Long Island", "Manhattan", "Brooklyn", "Queens",
        "Bronx", "Connecticut", "New Jersey", "Chicago", "London",
        "Paris", "Europe", "America", "United States", "Oxford",
        "Louisville", "Minnesota", "Santa Barbara", "San Francisco",
        "Venice", "Rome", "Ohio", "Indiana", "Missouri",
        "Normandy", "Marseilles", "Versailles", "Castile",
        "France", "England", "Germany", "Spain", "Italy",
        "Brazil", "Rio de Janeiro", "Manaus", "Amazon",
        "Madeira", "Bolivia", "Peru", "Buenos Aires",
        "Montenegro", "Wisconsin", "Lake Superior", "Lake Michigan",
        "Atlantic", "Pacific", "Mediterranean", "Fifth Avenue",
        "Broadway", "Wall Street", "Central Park",
        "São Paulo", "Sao Paulo", "Belém", "Belem", "Pará", "Para",
    }
    if name in real_indicators:
        return "real"
    real_suffixes = (
        "City", " Island", " Islands", " River", " Lake", " Street",
        " Avenue", " Road", " Bay", " County", " State", " Province",
        " Ocean", " Sea", " Mountain", " Mountains", " Valley",
        " Creek", " Harbor", " Harbour", " Port", " Bridge",
        " Station", " Airport",
    )
    if any(name.endswith(sfx) for sfx in real_suffixes):
        return "real"
    real_prefixes = (
        "Lake ", "Mount ", "Cape ", "Port ", "Fort ", "San ", "São ",
        "Rio ", "Sierra ", "Serra ",
    )
    if any(name.startswith(pfx) for pfx in real_prefixes):
        return "real"
    return "uncertain"


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 2: Named Entity Recognition (CoReNer, chunked) ===")

    cleaned_dir = Path(cfg["corpus"]["cleaned_dir"])
    metadata_file = Path(cfg["corpus"]["metadata_file"])
    dd = data_dir(cfg)
    entities_path = dd / "entities.jsonl"
    entities_path.parent.mkdir(parents=True, exist_ok=True)

    if entities_path.exists() and not force:
        log.info("entities.jsonl exists, skipping (use --force to overwrite).")
        return

    metadata = read_json(metadata_file)
    doc_filter = set(cfg["corpus"].get("doc_filter", []))
    doc_ids = [m["doc_id"] for m in metadata if not doc_filter or m["doc_id"] in doc_filter]

    model_name = cfg["ner"].get("corener_model", "aiola/roberta-large-corener")
    model, tokenizer = _load_corener(model_name)

    from corener.data import MTLDataset
    from corener.utils.prediction import convert_model_output

    entity_types: List[str] = cfg["ner"].get("entity_types", ["GPE", "LOC", "FAC"])
    fictional_overrides: List[str] = cfg["ner"].get("fictional_overrides", [])
    confidence_threshold: float = cfg["ner"].get("confidence_threshold", 0.7)
    max_chunk_tokens: int = cfg["ner"].get("max_chunk_tokens", 450)
    min_mentions: int = cfg["ner"].get("min_mention_count", 2)
    extra_blocklist: List[str] = cfg["ner"].get("entity_blocklist", [])

    known_fictional = KNOWN_FICTIONAL | set(fictional_overrides)
    blocklist = ENTITY_BLOCKLIST | set(extra_blocklist)
    registry: Dict[str, dict] = {}

    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found for doc_id '%s': %s", doc_id, cleaned_path)
            continue

        log.info("Processing doc_id '%s' from %s", doc_id, cleaned_path)
        sentences = list(iter_jsonl(cleaned_path, model=SentenceRecord))
        chunks = _build_chunks(sentences, tokenizer, max_tokens=max_chunk_tokens)
        log.info("  %d sentences in %d chunks", len(sentences), len(chunks))

        for chunk_idx, (chunk_text, chunk_sents, boundaries) in enumerate(chunks):
            if chunk_idx > 0 and chunk_idx % 25 == 0:
                log.info("  Processed %d/%d chunks", chunk_idx, len(chunks))
            dataset = MTLDataset(
                types=model.config.types,
                tokenizer=tokenizer,
                train_mode=False,
            )
            dataset.read_dataset([chunk_text])

            try:
                example = dataset.get_example(0)
            except Exception as e:
                log.debug("Skipping chunk %d (tokenization issue): %s", chunk_idx, e)
                continue

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

            parsed = convert_model_output(output=output, batch=example, dataset=dataset)
            if not parsed:
                continue

            result = parsed[0] if isinstance(parsed, list) else parsed
            tokens = result.get("tokens", [])
            if not tokens:
                continue

            char_offsets = _map_tokens_to_chars(tokens, chunk_text)

            for ent in result.get("entities", []):
                if ent["type"] not in entity_types:
                    continue
                start_tok, end_tok = ent["start"], ent["end"]
                if start_tok >= len(char_offsets) or end_tok > len(char_offsets):
                    continue

                char_start = char_offsets[start_tok][0]
                char_end = char_offsets[end_tok - 1][1]
                sent_id, local_start, local_end = _char_offset_to_span(char_start, char_end, boundaries)
                sent_text = next(s.text for s in chunk_sents if s.sentence_id == sent_id)

                surface = normalize_surface(" ".join(ent["span"]))
                canon = surface

                if canon not in registry:
                    registry[canon] = {
                        "name": canon,
                        "canonical_name": canon,
                        "ner_label": ent["type"],
                        "mentions": [],
                        "doc_ids": set(),
                    }

                mention = EntityMention(
                    sentence_id=sent_id,
                    char_start=local_start,
                    char_end=local_end,
                    source_text=f"...{sent_text[max(0, local_start-20):local_end+20]}...",
                )
                registry[canon]["mentions"].append(mention)
                registry[canon]["doc_ids"].add(doc_id)

            canon_lower_set = {c.lower(): c for c in registry}
            for cluster in result.get("clusters", []):
                matched_canon = None
                for m in cluster:
                    mention_text = " ".join(m["span"]).strip()
                    lower_mention = mention_text.lower()
                    if lower_mention in canon_lower_set:
                        matched_canon = canon_lower_set[lower_mention]
                        break
                    for key, canon in canon_lower_set.items():
                        if len(key) >= 4 and key in lower_mention:
                            matched_canon = canon
                            break
                    if matched_canon:
                        break
                if not matched_canon:
                    continue
                existing_keys = {(m.sentence_id, m.char_start, m.char_end) for m in registry[matched_canon]["mentions"]}
                for m in cluster:
                    mention_text = " ".join(m["span"])
                    if mention_text.strip().lower() == matched_canon.lower():
                        continue
                    start_tok, end_tok = m["start"], m["end"]
                    if start_tok >= len(char_offsets) or end_tok > len(char_offsets):
                        continue
                    char_start = char_offsets[start_tok][0]
                    char_end = char_offsets[end_tok - 1][1]
                    try:
                        sent_id, local_start, local_end = _char_offset_to_span(char_start, char_end, boundaries)
                    except ValueError:
                        continue
                    if (sent_id, local_start, local_end) in existing_keys:
                        continue
                    existing_keys.add((sent_id, local_start, local_end))
                    registry[matched_canon]["mentions"].append(
                        EntityMention(
                            sentence_id=sent_id,
                            char_start=local_start,
                            char_end=local_end,
                            source_text=f"[coref→{matched_canon}] ...{mention_text}...",
                        )
                    )
                    registry[matched_canon]["doc_ids"].add(doc_id)

    log.info("Found %d unique geographic entities (pre-filter)", len(registry))

    filtered_registry: Dict[str, dict] = {}
    for canon, info in registry.items():
        if _is_blocked(canon, blocklist):
            continue
        if _looks_like_ocr_artifact(canon):
            continue
        if _is_non_geographic(canon):
            continue
        if len(info["mentions"]) < min_mentions and canon not in known_fictional:
            continue
        filtered_registry[canon] = info

    log.info("Filtered: %d removed, %d remain", len(registry) - len(filtered_registry), len(filtered_registry))

    entities = []
    for idx, (canon, info) in enumerate(filtered_registry.items()):
        entity_type = _classify_entity(canon, known_fictional, confidence_threshold)
        entities.append(
            Entity(
                entity_id=_make_entity_id(idx),
                name=info["name"],
                canonical_name=canon,
                type=entity_type,
                ner_label=info["ner_label"],
                mentions=info["mentions"],
                mention_count=len(info["mentions"]),
                doc_ids=sorted(info["doc_ids"]),
            )
        )

    log.info(
        "Classification: %d real, %d fictional, %d uncertain",
        sum(1 for e in entities if e.type == "real"),
        sum(1 for e in entities if e.type == "fictional"),
        sum(1 for e in entities if e.type == "uncertain"),
    )
    write_jsonl(entities_path, entities, overwrite=True)
    log.info("Phase 2 complete. %d entities written to %s", len(entities), entities_path)
