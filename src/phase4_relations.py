"""Phase 4 — Spatial Relation Extraction.

Extracts pairwise spatial constraints between geographic entities from the
cleaned corpus using:
  1. spaCy pattern-based matching (high confidence).
  2. Co-occurrence (weak proximity signal).
  3. HuggingFace zero-shot classification (medium confidence, optional).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import spacy
from spacy.matcher import Matcher

from src.utils.io import iter_jsonl, read_jsonl, write_jsonl, read_json, data_dir
from src.utils.schemas import (
    GroundedEntity,
    SentenceRecord,
    SpatialRelation,
)

log = logging.getLogger(__name__)

_RELATION_ID_COUNTER = 0

HEDGING_WORDS = {
    "perhaps", "maybe", "possibly", "probably", "seem", "seemed",
    "appears", "apparently", "might", "may", "somewhat", "sort of",
    "kind of", "roughly", "approximately",
}

# Zero-shot candidate labels → relation type
HF_LABEL_MAP = {
    "north of": "north_of",
    "south of": "south_of",
    "east of": "east_of",
    "west of": "west_of",
    "near or close to": "near",
    "far from": "far",
    "across from": "across",
    "on the coast or shore": "on_coast",
    "in the same region": "in_region",
    "unrelated locations": None,
}


def _next_rid() -> str:
    global _RELATION_ID_COUNTER
    _RELATION_ID_COUNTER += 1
    return f"r_{_RELATION_ID_COUNTER:04d}"


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

def _build_patterns(nlp) -> Matcher:
    """Build spaCy Matcher patterns for spatial relations."""
    matcher = Matcher(nlp.vocab)

    # north/south/east/west of
    for direction in ("north", "south", "east", "west"):
        matcher.add(
            f"{direction}_of",
            [
                [{"LOWER": direction}, {"LOWER": "of"}],
                [{"LOWER": direction}, {"LOWER": "-"}, {"LOWER": "of"}],
            ],
        )

    # near variants
    matcher.add(
        "near",
        [
            [{"LOWER": "near"}],
            [{"LOWER": "close"}, {"LOWER": "to"}],
            [{"LOWER": "next"}, {"LOWER": "to"}],
            [{"LOWER": "beside"}],
            [{"LOWER": "adjacent"}, {"LOWER": "to"}],
            [{"LOWER": "by"}, {"LOWER": "the"}],
        ],
    )

    # far variants
    matcher.add(
        "far",
        [
            [{"LOWER": "far"}, {"LOWER": "from"}],
            [{"LOWER": "a"}, {"LOWER": "long"}, {"LOWER": "way"}, {"LOWER": "from"}],
            [{"LOWER": "distant"}, {"LOWER": "from"}],
        ],
    )

    # across
    matcher.add(
        "across",
        [
            [{"LOWER": "across"}, {"LOWER": "the"}, {"LOWER": {"IN": ["bay", "river", "water", "sound", "harbor", "harbour"]}}],
            [{"LOWER": "across"}, {"LOWER": "from"}],
            [{"LOWER": "opposite"}],
        ],
    )

    # on coast
    matcher.add(
        "on_coast",
        [
            [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": {"IN": ["coast", "shore", "waterfront", "beach", "bay"]}}],
            [{"LOWER": "along"}, {"LOWER": "the"}, {"LOWER": {"IN": ["coast", "shore", "waterfront"]}}],
        ],
    )

    # in region (handled separately via entity context)
    matcher.add(
        "in_region",
        [
            [{"LOWER": "in"}, {"ENT_TYPE": {"IN": ["GPE", "LOC"]}}],
        ],
    )

    return matcher


# ---------------------------------------------------------------------------
# Distance pattern (regex-based, e.g. "20 miles from")
# ---------------------------------------------------------------------------

_DISTANCE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(miles?|kilometers?|km|mi)\s+from",
    re.IGNORECASE,
)


def _extract_distance(text: str) -> Optional[Tuple[float, str]]:
    """Return (value, unit) if a distance pattern is found, else None."""
    m = _DISTANCE_RE.search(text)
    if m:
        value = float(m.group(1))
        unit_raw = m.group(2).lower()
        unit = "km" if unit_raw.startswith("k") else "miles"
        return value, unit
    return None


# ---------------------------------------------------------------------------
# Hedging / metaphor detection
# ---------------------------------------------------------------------------

def _hedging_penalty(text_lower: str) -> float:
    """Return a [0, 0.4] penalty based on hedging language in the sentence."""
    count = sum(1 for w in HEDGING_WORDS if w in text_lower)
    return min(0.4, count * 0.1)


def _is_likely_metaphorical(text_lower: str) -> bool:
    """Crude check for metaphorical spatial language."""
    meta_phrases = [
        "a world away", "worlds apart", "in another world",
        "at the heart of", "in the shadow of", "at the crossroads",
    ]
    return any(p in text_lower for p in meta_phrases)


# ---------------------------------------------------------------------------
# HuggingFace zero-shot helper
# ---------------------------------------------------------------------------

_hf_pipeline = None


def _get_hf_pipeline(model_name: str):
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline as hf_pipeline_fn
        log.info("Loading HuggingFace zero-shot model: %s", model_name)
        _hf_pipeline = hf_pipeline_fn(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # CPU
        )
    return _hf_pipeline


def _hf_classify(
    text: str, model_name: str, min_score: float
) -> Optional[Tuple[str, float]]:
    """
    Classify a sentence into a spatial relation type using zero-shot.
    Returns (relation_type, score) or None if below threshold.
    """
    try:
        pipe = _get_hf_pipeline(model_name)
        labels = list(HF_LABEL_MAP.keys())
        result = pipe(text, candidate_labels=labels, multi_label=False)
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        relation_type = HF_LABEL_MAP.get(top_label)
        if relation_type is not None and top_score >= min_score:
            return relation_type, top_score
    except Exception as exc:
        log.debug("HF classification error: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract_from_sentence(
    sent_text: str,
    sent_id: str,
    entity_names_in_sent: List[str],
    nlp,
    matcher: Matcher,
    extraction_methods: List[str],
    cfg: dict,
) -> List[SpatialRelation]:
    """Extract spatial relations from a single sentence."""
    relations: List[SpatialRelation] = []
    if len(entity_names_in_sent) < 1:
        return relations

    text_lower = sent_text.lower()
    if _is_likely_metaphorical(text_lower):
        return relations

    base_penalty = _hedging_penalty(text_lower)

    doc = nlp(sent_text)
    matches = matcher(doc)

    # Collect matched relation types and their token spans
    pattern_hits: List[Tuple[str, int, int]] = []
    for match_id, start, end in matches:
        rel_type = nlp.vocab.strings[match_id]
        pattern_hits.append((rel_type, start, end))

    if "pattern_match" in extraction_methods and pattern_hits:
        for rel_type, start, end in pattern_hits:
            confidence = max(0.4, 0.8 - base_penalty)
            weight = confidence

            # For directional relations, try to assign entity_1 and entity_2
            # by looking at entities before/after the relation token
            ents_before = [
                e for e in entity_names_in_sent
                if sent_text.lower().find(e.lower()) < doc[start].idx
            ]
            ents_after = [
                e for e in entity_names_in_sent
                if sent_text.lower().find(e.lower()) >= doc[end - 1].idx + len(doc[end - 1].text)
            ]

            e1 = ents_before[-1] if ents_before else (entity_names_in_sent[0] if entity_names_in_sent else None)
            e2 = ents_after[0] if ents_after else (entity_names_in_sent[1] if len(entity_names_in_sent) > 1 else None)

            if e1 is None:
                continue
            if rel_type == "on_coast":
                e2 = None  # unary

            # Check for distance pattern
            dist_match = _extract_distance(sent_text)
            if dist_match and rel_type not in ("near", "far"):
                rel_type = "distance_approx"

            relations.append(
                SpatialRelation(
                    relation_id=_next_rid(),
                    type=rel_type,
                    entity_1=e1,
                    entity_2=e2,
                    direction=None,
                    distance_value=dist_match[0] if dist_match else None,
                    distance_unit=dist_match[1] if dist_match else None,
                    weight=round(weight, 3),
                    uncertainty=round(1.0 - weight, 3),
                    source_sentence_id=sent_id,
                    source_text=sent_text[:200],
                    extraction_method="pattern_match",
                )
            )

    # Co-occurrence: all pairs of entities in same sentence
    if "co_occurrence" in extraction_methods and len(entity_names_in_sent) >= 2:
        co_weight = cfg.get("co_occurrence_weight", 0.3)
        for i in range(len(entity_names_in_sent)):
            for j in range(i + 1, len(entity_names_in_sent)):
                # Only add if no pattern relation was found between this pair
                e1, e2 = entity_names_in_sent[i], entity_names_in_sent[j]
                already = any(
                    (r.entity_1 == e1 and r.entity_2 == e2)
                    or (r.entity_1 == e2 and r.entity_2 == e1)
                    for r in relations
                )
                if not already:
                    relations.append(
                        SpatialRelation(
                            relation_id=_next_rid(),
                            type="co_occurrence",
                            entity_1=e1,
                            entity_2=e2,
                            weight=round(co_weight, 3),
                            uncertainty=round(1.0 - co_weight, 3),
                            source_sentence_id=sent_id,
                            source_text=sent_text[:200],
                            extraction_method="co_occurrence",
                        )
                    )

    # HuggingFace zero-shot (only for sentences with 2+ entities, no pattern hit)
    use_hf = "hf_zero_shot" in extraction_methods and cfg.get("use_hf_model", False)
    if use_hf and len(entity_names_in_sent) >= 2 and not pattern_hits:
        hf_result = _hf_classify(
            sent_text,
            cfg.get("hf_model", "facebook/bart-large-mnli"),
            cfg.get("hf_min_score", 0.5),
        )
        if hf_result:
            rel_type, score = hf_result
            e1 = entity_names_in_sent[0]
            e2 = entity_names_in_sent[1] if len(entity_names_in_sent) > 1 else None
            if rel_type == "on_coast":
                e2 = None
            weight = round(score * (1.0 - base_penalty), 3)
            relations.append(
                SpatialRelation(
                    relation_id=_next_rid(),
                    type=rel_type,
                    entity_1=e1,
                    entity_2=e2,
                    weight=weight,
                    uncertainty=round(1.0 - weight, 3),
                    source_sentence_id=sent_id,
                    source_text=sent_text[:200],
                    extraction_method="hf_zero_shot",
                )
            )

    return relations


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 4: Spatial Relation Extraction ===")

    cleaned_dir = Path(cfg["corpus"]["cleaned_dir"])
    metadata_file = Path(cfg["corpus"]["metadata_file"])
    dd = data_dir(cfg)
    grounded_path = dd / "grounded_entities.jsonl"
    relations_path = dd / "relations.jsonl"
    relations_path.parent.mkdir(parents=True, exist_ok=True)

    if relations_path.exists() and not force:
        log.info("relations.jsonl exists, skipping (use --force to overwrite).")
        return

    rel_cfg = cfg.get("relations", {})
    extraction_methods: List[str] = rel_cfg.get(
        "extraction_methods", ["pattern_match", "co_occurrence"]
    )

    # Build entity name set for quick lookup
    grounded_entities: List[GroundedEntity] = read_jsonl(grounded_path, model=GroundedEntity)
    entity_names: Set[str] = {e.canonical_name for e in grounded_entities}
    name_to_entity: Dict[str, GroundedEntity] = {e.canonical_name: e for e in grounded_entities}

    log.info("Loaded %d grounded entities", len(grounded_entities))

    # Load spaCy
    model_name = cfg["ner"].get("spacy_model", "en_core_web_lg")
    log.info("Loading spaCy model: %s", model_name)
    nlp = spacy.load(model_name)
    matcher = _build_patterns(nlp)

    doc_filter = set(cfg["corpus"].get("doc_filter", []))
    if metadata_file.exists():
        metadata = read_json(metadata_file)
        doc_ids = [m["doc_id"] for m in metadata if not doc_filter or m["doc_id"] in doc_filter]
    else:
        doc_ids = []
    if doc_filter and not doc_ids:
        for did in doc_filter:
            if (cleaned_dir / f"{did}.jsonl").exists():
                doc_ids.append(did)

    all_relations: List[SpatialRelation] = []

    for doc_id in doc_ids:
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"
        if not cleaned_path.exists():
            log.warning("Cleaned file not found: %s", cleaned_path)
            continue

        log.info("Extracting relations from doc_id '%s'", doc_id)
        sentences = list(iter_jsonl(cleaned_path, model=SentenceRecord))

        # Build per-sentence entity list
        sent_id_to_entities: Dict[str, List[str]] = defaultdict(list)
        for entity in grounded_entities:
            for mention in entity.mentions:
                if mention.sentence_id.startswith(doc_id):
                    sent_id_to_entities[mention.sentence_id].append(entity.canonical_name)

        for sent in sentences:
            ents_in_sent = list(set(sent_id_to_entities.get(sent.sentence_id, [])))
            if len(ents_in_sent) < 1:
                continue

            relations = _extract_from_sentence(
                sent.text,
                sent.sentence_id,
                ents_in_sent,
                nlp,
                matcher,
                extraction_methods,
                rel_cfg,
            )
            all_relations.extend(relations)

    # --- Post-extraction cleanup ---
    pre_filter_count = len(all_relations)

    # Remove self-relations (entity_1 == entity_2)
    all_relations = [
        r for r in all_relations
        if r.entity_2 is None or r.entity_1 != r.entity_2
    ]
    n_self = pre_filter_count - len(all_relations)

    # Remove relations where either entity is not in the grounded entity set
    all_relations = [
        r for r in all_relations
        if r.entity_1 in entity_names and (r.entity_2 is None or r.entity_2 in entity_names)
    ]
    n_missing = (pre_filter_count - n_self) - len(all_relations)

    log.info(
        "Extracted %d relations (removed %d self-relations, %d with unknown entities)",
        len(all_relations), n_self, n_missing,
    )
    log.info(
        "  Breakdown: %d pattern, %d co_occurrence, %d hf_zero_shot",
        sum(1 for r in all_relations if r.extraction_method == "pattern_match"),
        sum(1 for r in all_relations if r.extraction_method == "co_occurrence"),
        sum(1 for r in all_relations if r.extraction_method == "hf_zero_shot"),
    )

    write_jsonl(relations_path, all_relations, overwrite=True)
    log.info("Phase 4 complete. Written to %s", relations_path)
