"""Phase 1 — Corpus Preparation.

Downloads The Great Gatsby from Project Gutenberg (if absent), cleans the
text, sentence-segments it with spaCy, and writes JSONL + metadata.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import List

import requests
import spacy

from src.utils.io import load_config, write_jsonl, write_json
from src.utils.schemas import SentenceRecord

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project Gutenberg helpers
# ---------------------------------------------------------------------------

_PG_START_RE = re.compile(
    r"\*{3}\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*{3}",
    re.IGNORECASE | re.DOTALL,
)
_PG_END_RE = re.compile(
    r"\*{3}\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.IGNORECASE | re.DOTALL,
)

# Known corpus entries: (filename_stem, url, title, year, doc_type)
CORPUS_MANIFEST = [
    (
        "the_great_gatsby",
        None,  # url resolved from config
        "The Great Gatsby",
        1925,
        "novel",
    ),
    (
        "amazon_madeira_rivers",
        None,  # local file, no download needed
        "The Amazon and Madeira Rivers",
        1875,
        "travelogue",
    ),
]


def _download_text(url: str, dest: Path) -> None:
    """Download a URL to *dest* with a progress indicator."""
    log.info("Downloading %s → %s", url, dest)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    log.info("Download complete: %d bytes", len(resp.content))


def _strip_pg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer from plain-text file."""
    # Strip header
    m = _PG_START_RE.search(text)
    if m:
        text = text[m.end():]
    # Strip footer
    m = _PG_END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def _clean_text(text: str) -> str:
    """Normalize whitespace and fix common encoding artefacts."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove form-feed and other control characters except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse runs of blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common Gutenberg smart-quote artefacts
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    return text


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    """Execute Phase 1: corpus preparation."""
    log.info("=== Phase 1: Corpus Preparation ===")

    raw_dir = Path(cfg["corpus"]["raw_dir"])
    cleaned_dir = Path(cfg["corpus"]["cleaned_dir"])
    metadata_file = Path(cfg["corpus"]["metadata_file"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    gatsby_url = cfg["corpus"].get("gatsby_gutenberg_url")

    # Load spaCy for sentence segmentation (lightweight pipeline)
    log.info("Loading spaCy model for sentence segmentation...")
    try:
        nlp = spacy.load("en_core_web_lg", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    except OSError:
        log.warning("en_core_web_lg not found, falling back to en_core_web_sm")
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
        except OSError:
            # Use blank English with sentencizer as last resort
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")

    # Increase max_length for long documents
    nlp.max_length = 2_000_000

    metadata: List[dict] = []
    doc_filter = set(cfg["corpus"].get("doc_filter", []))

    for stem, _url, title, year, doc_type in CORPUS_MANIFEST:
        doc_id = _make_doc_id(stem)
        if doc_filter and doc_id not in doc_filter:
            log.info("Skipping '%s' — not in doc_filter %s", title, doc_filter)
            continue
        raw_path = raw_dir / f"{stem}.txt"
        cleaned_path = cleaned_dir / f"{doc_id}.jsonl"

        # --- Download if needed ---
        if not raw_path.exists():
            url = gatsby_url if stem == "the_great_gatsby" else _url
            if url is None:
                raise ValueError(f"No URL configured for {stem}")
            _download_text(url, raw_path)
        else:
            log.info("Raw file already exists: %s", raw_path)

        if cleaned_path.exists() and not force:
            log.info("Cleaned file exists, skipping (use --force to overwrite): %s", cleaned_path)
            # Still need to read sentence count for metadata
            with cleaned_path.open() as fh:
                num_sentences = sum(1 for line in fh if line.strip())
        else:
            # --- Clean ---
            raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
            # Only strip PG boilerplate for Gutenberg downloads
            if stem == "the_great_gatsby":
                text = _strip_pg_boilerplate(raw_text)
            else:
                text = raw_text
            text = _clean_text(text)

            log.info("Segmenting sentences for '%s' (%d chars)...", title, len(text))

            # --- Sentence segmentation ---
            # doc_id already computed above from _make_doc_id(stem)

            records: List[SentenceRecord] = []
            # Process in chunks to avoid memory issues with very long texts
            chunk_size = 100_000
            sent_idx = 0
            for chunk_start in range(0, len(text), chunk_size):
                chunk = text[chunk_start : chunk_start + chunk_size]
                # Don't split mid-sentence: extend to next newline
                if chunk_start + chunk_size < len(text):
                    end = text.find("\n", chunk_start + chunk_size)
                    if end != -1:
                        chunk = text[chunk_start:end]

                spacy_doc = nlp(chunk)
                for sent in spacy_doc.sents:
                    sent_text = sent.text.strip()
                    if len(sent_text) < 10:
                        continue  # skip trivially short fragments
                    records.append(
                        SentenceRecord(
                            doc_id=doc_id,
                            sentence_id=f"{doc_id}_sent_{sent_idx}",
                            text=sent_text,
                        )
                    )
                    sent_idx += 1

                if chunk_start + len(chunk) >= len(text):
                    break

            num_sentences = len(records)
            log.info("Extracted %d sentences from '%s'", num_sentences, title)

            # Deduplicate consecutive identical sentences (PG formatting artefacts)
            seen = set()
            unique_records: List[SentenceRecord] = []
            for r in records:
                if r.text not in seen:
                    unique_records.append(r)
                    seen.add(r.text)
            records = unique_records
            num_sentences = len(records)

            write_jsonl(cleaned_path, records, overwrite=True)

        # --- Metadata ---
        metadata.append(
            {
                "doc_id": doc_id,
                "title": title,
                "year": year,
                "type": doc_type,
                "source_file": str(raw_path),
                "num_sentences": num_sentences,
            }
        )

    write_json(metadata_file, metadata, overwrite=True)
    log.info("Phase 1 complete.  Metadata written to %s", metadata_file)


def _make_doc_id(stem: str) -> str:
    """Convert a filename stem to a clean doc_id."""
    return stem.replace("the_", "", 1) if stem.startswith("the_") else stem
