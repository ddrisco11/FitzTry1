"""Document-agnostic corpus preparation.

Reads a single plain-text file, normalises it, segments it into sentences,
and writes a JSONL file in the SentenceRecord schema:

    {"doc_id": "<slug>", "sentence_id": "<slug>_sent_N", "text": "..."}

Designed to handle:
  - Project Gutenberg headers / footers (auto-detected; safe no-op otherwise).
  - Mixed encodings (utf-8, latin-1) via best-effort decoding.
  - Smart-quote / em-dash / control-char normalisation.
  - Dialogue, em-dashes, and ellipses in sentence segmentation.

There is no document-specific logic here. The same code path runs for
*The Great Gatsby*, a travelogue, a transcript, or any other prose input.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List, Optional

import spacy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Slug / doc_id derivation
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Filesystem-safe identifier derived from a filename or title."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = _SLUG_RE.sub("_", name.lower()).strip("_")
    return name or "doc"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_PG_START_RE = re.compile(
    r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}",
    re.IGNORECASE | re.DOTALL,
)
_PG_END_RE = re.compile(
    r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)


def read_text(path: Path) -> str:
    """Read a text file, falling back from UTF-8 to latin-1 on failure."""
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def strip_gutenberg(text: str) -> str:
    """Strip Project Gutenberg header/footer if present. No-op otherwise."""
    m = _PG_START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _PG_END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def normalise(text: str) -> str:
    """Normalise whitespace, control characters, and Unicode punctuation."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = (text
            .replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
            .replace("—", "--").replace("–", "-")
            .replace("…", "..."))
    return text


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------

_SPACY_DISABLE = ["ner", "tagger", "lemmatizer", "attribute_ruler"]


def _load_spacy(model: str = "en_core_web_lg") -> "spacy.language.Language":
    try:
        nlp = spacy.load(model, disable=_SPACY_DISABLE)
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm", disable=_SPACY_DISABLE)
        except OSError:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
    nlp.max_length = 4_000_000
    return nlp


def segment(text: str, nlp=None, min_chars: int = 10,
            chunk_size: int = 100_000) -> List[str]:
    """Sentence-segment a long string. Sentences shorter than `min_chars`
    are dropped (epigraph fragments, page numbers, etc.). Long inputs are
    processed chunk-by-chunk to avoid spaCy's memory ceiling."""
    if nlp is None:
        nlp = _load_spacy()

    sentences: List[str] = []
    n = len(text)
    pos = 0
    while pos < n:
        end = min(pos + chunk_size, n)
        if end < n:
            nl = text.find("\n", end)
            if nl != -1:
                end = nl
        chunk = text[pos:end]
        for sent in nlp(chunk).sents:
            s = sent.text.strip()
            if len(s) >= min_chars:
                sentences.append(s)
        pos = end if end > pos else n
    return sentences


def deduplicate_consecutive(sents: Iterable[str]) -> List[str]:
    """Drop runs of identical sentences (PG formatting artefacts)."""
    out: List[str] = []
    prev: Optional[str] = None
    for s in sents:
        if s != prev:
            out.append(s)
            prev = s
    return out


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def ingest_file(input_path: Path,
                cleaned_dir: Path,
                doc_id: Optional[str] = None,
                spacy_model: str = "en_core_web_lg") -> Path:
    """Ingest one plain-text file. Returns the path of the cleaned JSONL."""
    input_path = Path(input_path)
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    doc_id = doc_id or slugify(input_path.stem)
    out_path = cleaned_dir / f"{doc_id}.jsonl"

    log.info("Ingesting %s -> %s (doc_id=%s)", input_path, out_path, doc_id)
    raw = read_text(input_path)
    text = normalise(strip_gutenberg(raw))

    nlp = _load_spacy(spacy_model)
    sents = deduplicate_consecutive(segment(text, nlp=nlp))

    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(sents):
            rec = {"doc_id": doc_id,
                   "sentence_id": f"{doc_id}_sent_{i}",
                   "text": s}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d sentences to %s", len(sents), out_path)
    return out_path
