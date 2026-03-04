"""File I/O helpers: JSONL read/write with schema validation, config loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, List, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    """Load and return the YAML configuration file as a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    log.info("Loaded config from %s", path)
    return cfg


def data_dir(cfg: dict) -> Path:
    """Return the configured data output directory (default: 'data')."""
    return Path(cfg.get("data_dir", "data"))


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path, model: Type[T] | None = None) -> List[T | dict]:
    """
    Read a JSONL file.  If *model* is provided, each line is parsed and
    validated through that Pydantic model.  Otherwise raw dicts are returned.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected input file '{path}' does not exist.  "
            f"Make sure the preceding pipeline phase has been run."
        )
    records: List[T | dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if model is not None:
                try:
                    records.append(model.model_validate(raw))
                except ValidationError as exc:
                    log.warning("Schema validation failed at %s line %d: %s", path, lineno, exc)
            else:
                records.append(raw)
    log.debug("Read %d records from %s", len(records), path)
    return records


def iter_jsonl(path: str | Path, model: Type[T] | None = None) -> Iterator[T | dict]:
    """Streaming version of read_jsonl — yields one record at a time."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected input file '{path}' does not exist.  "
            f"Make sure the preceding pipeline phase has been run."
        )
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if model is not None:
                try:
                    yield model.model_validate(raw)
                except ValidationError as exc:
                    log.warning("Schema validation failed at %s line %d: %s", path, lineno, exc)
            else:
                yield raw


def write_jsonl(path: str | Path, records: List[BaseModel | dict], overwrite: bool = True) -> None:
    """Write a list of Pydantic models or dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        log.info("Skipping write — file exists and overwrite=False: %s", path)
        return
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            if isinstance(record, BaseModel):
                fh.write(record.model_dump_json() + "\n")
            else:
                fh.write(json.dumps(record) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def write_json(path: str | Path, data: dict | list | BaseModel, overwrite: bool = True) -> None:
    """Write a single JSON file (not JSONL)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        log.info("Skipping write — file exists and overwrite=False: %s", path)
        return
    with path.open("w", encoding="utf-8") as fh:
        if isinstance(data, BaseModel):
            fh.write(data.model_dump_json(indent=2))
        else:
            json.dump(data, fh, indent=2)
    log.info("Wrote JSON to %s", path)


def read_json(path: str | Path, model: Type[T] | None = None) -> T | dict:
    """Read a JSON file, optionally validating through a Pydantic model."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected input file '{path}' does not exist.  "
            f"Make sure the preceding pipeline phase has been run."
        )
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if model is not None:
        return model.model_validate(raw)
    return raw
