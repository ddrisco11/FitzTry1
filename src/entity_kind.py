"""Entity-kind tagging for nodes in a spatial graph.

This module classifies each node surface form into one of:

  - ``place``           — a recognised geographic / facility entity
                          (spaCy NER labels GPE, LOC, FAC).
  - ``deictic``         — a closed-class deictic pronoun or locative
                          adverb (``I``, ``me``, ``my``, ``you``, ``here``,
                          ``there``, ``where``, …). These are linguistic
                          universals of English; the list is finite.
  - ``person_locus``    — a span tagged PERSON by NER. Under ISO-Space,
                          the metonymic / social-spatial reading is what
                          licenses these as ``SpatialEntity``s
                          (e.g. *"at Gatsby's"* ≡ Gatsby's residence).
  - ``common_locus``    — anything else. Mostly common-noun loci
                          (``the room``, ``the house``, ``the corner``).

Crucially, **no document-specific blocklist is used**. The classifier
takes a surface string and decides on the strength of (i) a closed-class
deixis lexicon and (ii) a general-purpose NER model. The same code runs
on any English prose without modification.

The four-way tag is propagated onto graph nodes so a downstream consumer
can filter or colour them, but nothing is silently dropped — every entity
the extractor surfaces remains a first-class node.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Iterable, Dict

log = logging.getLogger(__name__)


# Closed-class English deictics. These are linguistic universals; extend
# the set with care. Stripped of articles before lookup.
DEICTIC_TOKENS = frozenset({
    # 1st person
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    # 2nd person
    "you", "your", "yours", "yourself", "yourselves",
    # 3rd person reflexive (rarely a locus, but possible)
    "himself", "herself", "itself", "themselves",
    # locative pro-forms
    "here", "there", "where", "everywhere", "anywhere",
    "somewhere", "nowhere", "elsewhere",
    "this place", "that place", "the place",
})

# Strip leading articles / possessives before the deixis check so that
# "the here" or "this here" still tag as deictic.
_STRIP_LEADING = re.compile(
    r"^(?:the|a|an|this|that|these|those|some|any|no)\s+",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_ner(model: str = "en_core_web_lg"):
    """Load a spaCy NER pipeline; fall back to en_core_web_sm if needed.
    Returns ``None`` if no spaCy model is installed (caller falls back to
    deixis + heuristic only)."""
    import spacy
    for name in (model, "en_core_web_sm"):
        try:
            return spacy.load(name, disable=["lemmatizer", "attribute_ruler"])
        except OSError:
            continue
    log.warning("No spaCy model available; entity_kind will skip NER.")
    return None


def _normalise(s: str) -> str:
    return _STRIP_LEADING.sub("", (s or "").strip()).lower()


def classify(surface: str, ner=None) -> str:
    """Return the entity-kind tag for a single surface form."""
    norm = _normalise(surface)
    if not norm:
        return "common_locus"
    if norm in DEICTIC_TOKENS:
        return "deictic"

    if ner is None:
        ner = _load_ner()
    if ner is None:
        # No NER available — fall back to a conservative heuristic:
        # capitalised single tokens are *probably* proper nouns.
        if surface.strip() and surface.strip()[0].isupper() and " " not in surface.strip():
            return "person_locus"
        return "common_locus"

    doc = ner(surface)
    labels = {ent.label_ for ent in doc.ents}
    if labels & {"GPE", "LOC", "FAC"}:
        return "place"
    if "PERSON" in labels:
        return "person_locus"
    return "common_locus"


def classify_all(surfaces: Iterable[str], model: str = "en_core_web_lg") -> Dict[str, str]:
    """Classify a batch of surface forms; cached per surface."""
    ner = _load_ner(model)
    out: Dict[str, str] = {}
    for s in surfaces:
        if s in out:
            continue
        out[s] = classify(s, ner=ner)
    return out
