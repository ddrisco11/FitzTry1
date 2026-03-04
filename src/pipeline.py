"""Pipeline orchestrator — runs all phases end-to-end or individually.

Usage:
    python -m src.pipeline --config config.yaml
    python -m src.pipeline --config config.yaml --phase 2
    python -m src.pipeline --config config.yaml --phase 1 --force
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.utils.io import load_config

# Phase imports
from src import (
    phase1_corpus_prep,
    phase2_ner,
    phase2b_coref,
    phase3_grounding,
    phase4_relations,
    phase4b_geographic_edges,
    phase5_constraints,
    phase6_inference,
    phase7_convergence,
    phase8_visualization,
)

PHASES = [
    phase1_corpus_prep,
    phase2_ner,
    phase2b_coref,
    phase3_grounding,
    phase4_relations,
    phase4b_geographic_edges,
    phase5_constraints,
    phase6_inference,
    phase7_convergence,
    phase8_visualization,
]

PHASE_NAMES = [
    "Corpus Preparation",
    "Named Entity Recognition",
    "Coreference Resolution",
    "Geographic Grounding",
    "Spatial Relation Extraction",
    "Geographic Knowledge Edges",
    "Formal Constraint Model",
    "Probabilistic Inference",
    "Convergence Diagnostics",
    "Visualization",
]


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


@click.command()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.option("--phase", default=None, type=int, help="Run only this phase (1–8). Omit to run all.")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing outputs.")
@click.option("--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
def main(config: str, phase: int | None, force: bool, verbose: bool) -> None:
    """Probabilistic Literary Geography Pipeline."""
    _setup_logging(verbose)
    log = logging.getLogger("pipeline")

    cfg_path = Path(config)
    if not cfg_path.exists():
        log.error("Config file not found: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)
    log.info("Loaded config from %s", cfg_path)

    if phase is not None:
        if not 1 <= phase <= len(PHASES):
            log.error("--phase must be between 1 and %d (4b = 5)", len(PHASES))
            sys.exit(1)
        _run_phase(phase - 1, cfg, force, log)
    else:
        log.info("Running full pipeline (%d phases)", len(PHASES))
        for i in range(len(PHASES)):
            _run_phase(i, cfg, force, log)

    log.info("Pipeline complete.")


def _run_phase(idx: int, cfg: dict, force: bool, log: logging.Logger) -> None:
    name = PHASE_NAMES[idx]
    module = PHASES[idx]
    log.info("━━━ Phase %d: %s ━━━", idx + 1, name)
    try:
        module.run(cfg, force=force)
    except Exception as exc:
        log.exception("Phase %d (%s) failed: %s", idx + 1, name, exc)
        raise SystemExit(f"Aborting: Phase {idx + 1} ({name}) failed.") from exc


if __name__ == "__main__":
    main()
