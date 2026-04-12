"""Pipeline orchestrator — runs all phases end-to-end or individually.

Strategy B stage ordering (when ner.backend == "mistral_joint"):

  1. Cleaning           — sentence-segmented corpus
  2. Mistral Extraction — joint NER + spatial relation extraction (Ollama)
  3. Coreference        — cross-span coref on Mistral-produced entities
  4. Geographic Grounding — Nominatim geocoding
  5. Graph Building     — finalise relations.jsonl using grounded entities
  6. Geographic Edges   — real<->real + anchor constraints
  7. Constraints        — formal constraint model
  8. Inference          — probabilistic positioning (emcee / Metropolis)
  9. Convergence        — diagnostics
 10. Visualisation      — maps and plots

Legacy ordering (when ner.backend == "corener"):

  1. Cleaning
  2. CoReNer NER
  3. Coreference
  4. Geographic Grounding
  5. Mistral Relations (old Phase 4)
  6. Geographic Edges
  7. Constraints
  8. Inference
  9. Convergence
 10. Visualisation

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

# Phase/stage imports — all paths
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
    phase_mistral_joint,
    phase_graph_build,
)

# ---------------------------------------------------------------------------
# Strategy B — Mistral-first pipeline (default)
# ---------------------------------------------------------------------------

PHASES_STRATEGY_B = [
    phase1_corpus_prep,       # 1: Cleaning
    phase_mistral_joint,      # 2: Joint Mistral NER + Relations
    phase2b_coref,            # 3: Coreference Resolution
    phase3_grounding,         # 4: Geographic Grounding
    phase_graph_build,        # 5: Graph Building
    phase4b_geographic_edges, # 6: Geographic Knowledge Edges
    phase5_constraints,       # 7: Formal Constraint Model
    phase6_inference,         # 8: Probabilistic Inference
    phase7_convergence,       # 9: Convergence Diagnostics
    phase8_visualization,     # 10: Visualisation
]

PHASE_NAMES_STRATEGY_B = [
    "Cleaning",
    "Mistral Joint Extraction",
    "Coreference Resolution",
    "Geographic Grounding",
    "Graph Building",
    "Geographic Knowledge Edges",
    "Formal Constraint Model",
    "Probabilistic Inference",
    "Convergence Diagnostics",
    "Visualisation",
]

# ---------------------------------------------------------------------------
# Legacy — CoReNer-first pipeline
# ---------------------------------------------------------------------------

PHASES_LEGACY = [
    phase1_corpus_prep,       # 1: Cleaning
    phase2_ner,               # 2: CoReNer NER
    phase2b_coref,            # 3: Coreference Resolution
    phase3_grounding,         # 4: Geographic Grounding
    phase4_relations,         # 5: Mistral Relations (old Phase 4)
    phase4b_geographic_edges, # 6: Geographic Knowledge Edges
    phase5_constraints,       # 7: Formal Constraint Model
    phase6_inference,         # 8: Probabilistic Inference
    phase7_convergence,       # 9: Convergence Diagnostics
    phase8_visualization,     # 10: Visualisation
]

PHASE_NAMES_LEGACY = [
    "Cleaning",
    "Named Entity Recognition (CoReNer)",
    "Coreference Resolution",
    "Geographic Grounding",
    "Spatial Relation Extraction (Mistral)",
    "Geographic Knowledge Edges",
    "Formal Constraint Model",
    "Probabilistic Inference",
    "Convergence Diagnostics",
    "Visualisation",
]

# Backwards compatibility aliases
PHASES = PHASES_STRATEGY_B
PHASE_NAMES = PHASE_NAMES_STRATEGY_B


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def _select_pipeline(cfg: dict):
    """Return (phases, phase_names) based on ner.backend config."""
    backend = cfg.get("ner", {}).get("backend", "mistral_joint")
    if backend == "corener":
        return PHASES_LEGACY, PHASE_NAMES_LEGACY
    # Default: Strategy B
    return PHASES_STRATEGY_B, PHASE_NAMES_STRATEGY_B


@click.command()
@click.option("--config", default="config.yaml", show_default=True, help="Path to config.yaml")
@click.option("--phase", default=None, type=int, help="Run only this phase (1-N). Omit to run all.")
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

    phases, phase_names = _select_pipeline(cfg)
    backend = cfg.get("ner", {}).get("backend", "mistral_joint")
    log.info("Pipeline backend: %s (%d stages)", backend, len(phases))

    if phase is not None:
        if not 1 <= phase <= len(phases):
            log.error("--phase must be between 1 and %d", len(phases))
            sys.exit(1)
        _run_phase(phase - 1, phases, phase_names, cfg, force, log)
    else:
        log.info("Running full pipeline (%d stages)", len(phases))
        for i in range(len(phases)):
            _run_phase(i, phases, phase_names, cfg, force, log)

    log.info("Pipeline complete.")


def _run_phase(
    idx: int,
    phases: list,
    phase_names: list,
    cfg: dict,
    force: bool,
    log: logging.Logger,
) -> None:
    name = phase_names[idx]
    module = phases[idx]
    log.info("━━━ Stage %d: %s ━━━", idx + 1, name)
    try:
        module.run(cfg, force=force)
    except Exception as exc:
        log.exception("Stage %d (%s) failed: %s", idx + 1, name, exc)
        raise SystemExit(f"Aborting: Stage {idx + 1} ({name}) failed.") from exc


if __name__ == "__main__":
    main()
