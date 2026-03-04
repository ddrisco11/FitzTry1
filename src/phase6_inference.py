"""Phase 6 — Probabilistic Inference.

Samples coordinate configurations for fictional entities using:
- Metropolis-Hastings MCMC (single or multiple chains), or
- emcee ensemble sampler (selected via config.yaml inference.method).

Energy function: E(config) = Σ_c weight_c * penalty_c(config)
Sampling distribution: P(config) ∝ exp(-β * E(config))
"""

from __future__ import annotations

import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.utils.io import read_json, write_json, data_dir
from src.utils.schemas import ConstraintModel, Sample, EntityPosition
from src.utils.geo import bounding_box_km

log = logging.getLogger(__name__)

# Numeric type codes for constraint types (avoids string comparison in hot loop)
_T_NORTH_OF = 0
_T_SOUTH_OF = 1
_T_EAST_OF = 2
_T_WEST_OF = 3
_T_NEAR = 4
_T_FAR = 5
_T_DIST_APPROX = 6
_T_ACROSS = 7
_T_IN_REGION = 8
_T_CO_OCCURRENCE = 9
_T_SKIP = -1

_TYPE_MAP = {
    "north_of": _T_NORTH_OF,
    "south_of": _T_SOUTH_OF,
    "east_of": _T_EAST_OF,
    "west_of": _T_WEST_OF,
    "near": _T_NEAR,
    "far": _T_FAR,
    "distance_approx": _T_DIST_APPROX,
    "across": _T_ACROSS,
    "in_region": _T_IN_REGION,
    "co_occurrence": _T_CO_OCCURRENCE,
    "on_coast": _T_SKIP,
}


# ---------------------------------------------------------------------------
# Compiled constraint representation for fast NumPy evaluation
# ---------------------------------------------------------------------------

def _compile_constraints(
    model: ConstraintModel,
    latent_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compile constraints into parallel NumPy arrays for vectorised evaluation.

    Returns (ctypes, idx1, idx2, weights, params_arr) where:
      - ctypes[i]  = integer type code
      - idx1[i]    = flat-index into param vector for entity1 x (or -1 if fixed)
      - idx2[i]    = flat-index into param vector for entity2 x (or -1 if fixed)
      - weights[i] = constraint weight
      - params_arr[i, :] = [p0, p1, p2, fx1, fy1, fx2, fy2]
        p0/p1/p2 = constraint-specific parameters (epsilon, d_near, target_d, sigma, radius, etc.)
        fx1/fy1  = fixed coords for entity1 (used when idx1 == -1)
        fx2/fy2  = fixed coords for entity2 (used when idx2 == -1)
    """
    fixed_xy: Dict[str, Tuple[float, float]] = {
        f.entity_id: (f.x, f.y) for f in model.fixed_entities
    }
    latent_idx: Dict[str, int] = {eid: i for i, eid in enumerate(latent_names)}

    n = len(model.constraints)
    ctypes = np.full(n, _T_SKIP, dtype=np.int8)
    idx1 = np.full(n, -1, dtype=np.int32)
    idx2 = np.full(n, -1, dtype=np.int32)
    weights = np.zeros(n, dtype=np.float64)
    params_arr = np.zeros((n, 7), dtype=np.float64)

    keep = 0
    for c in model.constraints:
        tc = _TYPE_MAP.get(c.type, _T_SKIP)
        if tc == _T_SKIP:
            continue

        # Unary in_region: centroid comes from params, only needs entity1
        if tc == _T_IN_REGION and len(c.entities) == 1:
            eid1 = c.entities[0]
            if eid1 in latent_idx:
                i1 = latent_idx[eid1] * 2
                fx1, fy1 = 0.0, 0.0
            elif eid1 in fixed_xy:
                i1 = -1
                fx1, fy1 = fixed_xy[eid1]
            else:
                continue
            p = c.params
            if "centroid_x_km" not in p and "centroid_y_km" not in p:
                continue
            p0 = p.get("centroid_x_km", 0.0)
            p1 = p.get("centroid_y_km", 0.0)
            p2 = p.get("radius_km", 20.0)
            ctypes[keep] = tc
            idx1[keep] = i1
            idx2[keep] = -1
            weights[keep] = c.weight
            params_arr[keep] = [p0, p1, p2, fx1, fy1, 0.0, 0.0]
            keep += 1
            continue

        if len(c.entities) < 2:
            continue

        eid1, eid2 = c.entities[0], c.entities[1]

        # Resolve entity 1
        if eid1 in latent_idx:
            i1 = latent_idx[eid1] * 2
            fx1, fy1 = 0.0, 0.0
        elif eid1 in fixed_xy:
            i1 = -1
            fx1, fy1 = fixed_xy[eid1]
        else:
            continue

        # Resolve entity 2
        if eid2 in latent_idx:
            i2 = latent_idx[eid2] * 2
            fx2, fy2 = 0.0, 0.0
        elif eid2 in fixed_xy:
            i2 = -1
            fx2, fy2 = fixed_xy[eid2]
        else:
            continue

        p = c.params
        if tc in (_T_NORTH_OF, _T_SOUTH_OF, _T_EAST_OF, _T_WEST_OF):
            p0 = p.get("epsilon_km", 1.0)
            p1 = p2 = 0.0
        elif tc == _T_NEAR:
            p0 = p.get("d_near_km", 10.0)
            p1 = p2 = 0.0
        elif tc == _T_FAR:
            p0 = p.get("d_far_km", 50.0)
            p1 = p2 = 0.0
        elif tc == _T_DIST_APPROX:
            p0 = p.get("target_d_km", 10.0)
            p1 = p.get("sigma_km", 5.0)
            p2 = 0.0
        elif tc == _T_ACROSS:
            p0 = p.get("d_near_km", 15.0)
            p1 = p.get("epsilon_km", 1.0)
            p2 = 0.0
        elif tc == _T_IN_REGION:
            p0 = p.get("centroid_x_km", fx2)
            p1 = p.get("centroid_y_km", fy2)
            p2 = p.get("radius_km", 20.0)
        elif tc == _T_CO_OCCURRENCE:
            p0 = p.get("d_near_km", 10.0)
            p1 = p2 = 0.0
        else:
            p0 = p1 = p2 = 0.0

        ctypes[keep] = tc
        idx1[keep] = i1
        idx2[keep] = i2
        weights[keep] = c.weight
        params_arr[keep] = [p0, p1, p2, fx1, fy1, fx2, fy2]
        keep += 1

    return ctypes[:keep], idx1[:keep], idx2[:keep], weights[:keep], params_arr[:keep]


def build_energy_fn(
    model: ConstraintModel,
    latent_names: List[str],
) -> Callable[[np.ndarray], float]:
    """Build a fast energy function using pre-compiled constraint arrays."""
    ctypes, idx1_arr, idx2_arr, w_arr, p_arr = _compile_constraints(model, latent_names)
    n_constraints = len(ctypes)

    log.info("Compiled %d constraints for fast evaluation", n_constraints)

    def energy(params: np.ndarray) -> float:
        total = 0.0
        for k in range(n_constraints):
            tc = ctypes[k]
            w = w_arr[k]
            i1 = idx1_arr[k]
            i2 = idx2_arr[k]

            if i1 >= 0:
                x1 = params[i1]; y1 = params[i1 + 1]
            else:
                x1 = p_arr[k, 3]; y1 = p_arr[k, 4]

            if i2 >= 0:
                x2 = params[i2]; y2 = params[i2 + 1]
            else:
                x2 = p_arr[k, 5]; y2 = p_arr[k, 6]

            if tc == _T_NORTH_OF:
                v = (y2 + p_arr[k, 0]) - y1
                if v > 0.0: total += w * v * v
            elif tc == _T_SOUTH_OF:
                v = y1 - (y2 - p_arr[k, 0])
                if v > 0.0: total += w * v * v
            elif tc == _T_EAST_OF:
                v = (x2 + p_arr[k, 0]) - x1
                if v > 0.0: total += w * v * v
            elif tc == _T_WEST_OF:
                v = x1 - (x2 - p_arr[k, 0])
                if v > 0.0: total += w * v * v
            elif tc == _T_NEAR or tc == _T_CO_OCCURRENCE:
                dx = x2 - x1; dy = y2 - y1
                dist = (dx*dx + dy*dy) ** 0.5
                v = dist - p_arr[k, 0]
                if v > 0.0: total += w * v * v
            elif tc == _T_FAR:
                dx = x2 - x1; dy = y2 - y1
                dist = (dx*dx + dy*dy) ** 0.5
                v = p_arr[k, 0] - dist
                if v > 0.0: total += w * v * v
            elif tc == _T_DIST_APPROX:
                dx = x2 - x1; dy = y2 - y1
                dist = (dx*dx + dy*dy) ** 0.5
                diff = dist - p_arr[k, 0]
                sigma = p_arr[k, 1]
                total += w * diff * diff / (2.0 * sigma * sigma)
            elif tc == _T_ACROSS:
                dx = x2 - x1; dy = y2 - y1
                dist = (dx*dx + dy*dy) ** 0.5
                v = dist - p_arr[k, 0]
                if v > 0.0: total += w * v * v
                ve = (x2 + p_arr[k, 1]) - x1
                if ve > 0.0: total += w * 0.3 * ve * ve
            elif tc == _T_IN_REGION:
                cx = p_arr[k, 0]; cy = p_arr[k, 1]; r = p_arr[k, 2]
                dx = x1 - cx; dy = y1 - cy
                dist = (dx*dx + dy*dy) ** 0.5
                v = dist - r
                if v > 0.0: total += w * v * v

        return total

    return energy


# ---------------------------------------------------------------------------
# Metropolis-Hastings
# ---------------------------------------------------------------------------

def _metropolis_chain(
    energy_fn: Callable[[np.ndarray], float],
    init_params: np.ndarray,
    n_steps: int,
    proposal_std: float,
    beta: float,
    thin: int,
    rng: np.random.Generator,
    chain_id: int = 0,
) -> List[dict]:
    """Run one Metropolis chain with per-entity block proposals and adaptive step size."""
    params = init_params.copy()
    E = energy_fn(params)
    n_latent = len(params) // 2
    samples = []
    n_accepted = 0

    adapt_interval = 500
    target_accept = 0.30
    current_std = proposal_std
    window_accepted = 0
    window_total = 0

    for step in range(n_steps):
        entity_idx = rng.integers(0, n_latent)
        proposal = params.copy()
        proposal[2 * entity_idx]     += rng.normal(0, current_std)
        proposal[2 * entity_idx + 1] += rng.normal(0, current_std)

        E_new = energy_fn(proposal)
        dE = E_new - E

        if dE < 0 or rng.random() < np.exp(-beta * dE):
            params = proposal
            E = E_new
            n_accepted += 1
            window_accepted += 1

        window_total += 1

        if window_total >= adapt_interval:
            rate = window_accepted / window_total
            if rate < target_accept * 0.8:
                current_std *= 0.8
            elif rate > target_accept * 1.2:
                current_std *= 1.2
            window_accepted = 0
            window_total = 0

        if (step + 1) % thin == 0:
            samples.append({"step": step + 1, "params": params.copy().tolist(), "energy": E, "chain_id": chain_id})

    acceptance_rate = n_accepted / n_steps
    log.info("Chain %d: acceptance rate = %.3f (final proposal_std = %.2f km)", chain_id, acceptance_rate, current_std)
    return samples


# ---------------------------------------------------------------------------
# Parallel chain runner (top-level function for pickling by multiprocessing)
# ---------------------------------------------------------------------------

def _run_single_chain(args: tuple) -> List[dict]:
    """Top-level function so multiprocessing can pickle it."""
    (model_dict, latent_names, init_params, n_steps,
     proposal_std, beta, thin, seed, chain_id, burn_in) = args

    model = ConstraintModel.model_validate(model_dict)
    energy_fn = build_energy_fn(model, latent_names)
    rng = np.random.default_rng(seed + chain_id * 1000)

    chain_init = init_params + rng.normal(0, proposal_std, len(init_params))
    raw = _metropolis_chain(energy_fn, chain_init, n_steps, proposal_std, beta, thin, rng, chain_id)

    burn_thinned = burn_in // thin
    return raw[burn_thinned:]


# ---------------------------------------------------------------------------
# emcee sampler with multiprocessing support
# ---------------------------------------------------------------------------

# Module-level state for the parallel log_prob worker function.
# Stored here so multiprocessing can pickle a reference to the top-level function.
_EMCEE_CTYPES: Optional[np.ndarray] = None
_EMCEE_IDX1: Optional[np.ndarray] = None
_EMCEE_IDX2: Optional[np.ndarray] = None
_EMCEE_WEIGHTS: Optional[np.ndarray] = None
_EMCEE_PARAMS_ARR: Optional[np.ndarray] = None
_EMCEE_BETA: float = 1.0
_EMCEE_BOUNDS: Optional[float] = None


def _init_emcee_worker(ct, i1, i2, w, pa, beta, bounds=None):
    """Initializer for pool workers — stores compiled arrays in each process."""
    global _EMCEE_CTYPES, _EMCEE_IDX1, _EMCEE_IDX2, _EMCEE_WEIGHTS, _EMCEE_PARAMS_ARR, _EMCEE_BETA, _EMCEE_BOUNDS
    _EMCEE_CTYPES = ct
    _EMCEE_IDX1 = i1
    _EMCEE_IDX2 = i2
    _EMCEE_WEIGHTS = w
    _EMCEE_PARAMS_ARR = pa
    _EMCEE_BETA = beta
    _EMCEE_BOUNDS = bounds


def _emcee_log_prob(params: np.ndarray) -> float:
    """Top-level log_prob that reads compiled arrays from module globals."""
    if _EMCEE_BOUNDS is not None:
        if np.any(np.abs(params) > _EMCEE_BOUNDS):
            return -np.inf
    total = 0.0
    ct = _EMCEE_CTYPES
    i1a = _EMCEE_IDX1
    i2a = _EMCEE_IDX2
    wa = _EMCEE_WEIGHTS
    pa = _EMCEE_PARAMS_ARR
    n = len(ct)

    for k in range(n):
        tc = ct[k]
        w = wa[k]
        i1 = i1a[k]
        i2 = i2a[k]

        if i1 >= 0:
            x1 = params[i1]; y1 = params[i1 + 1]
        else:
            x1 = pa[k, 3]; y1 = pa[k, 4]
        if i2 >= 0:
            x2 = params[i2]; y2 = params[i2 + 1]
        else:
            x2 = pa[k, 5]; y2 = pa[k, 6]

        if tc == _T_NORTH_OF:
            v = (y2 + pa[k, 0]) - y1
            if v > 0.0: total += w * v * v
        elif tc == _T_SOUTH_OF:
            v = y1 - (y2 - pa[k, 0])
            if v > 0.0: total += w * v * v
        elif tc == _T_EAST_OF:
            v = (x2 + pa[k, 0]) - x1
            if v > 0.0: total += w * v * v
        elif tc == _T_WEST_OF:
            v = x1 - (x2 - pa[k, 0])
            if v > 0.0: total += w * v * v
        elif tc == _T_NEAR or tc == _T_CO_OCCURRENCE:
            dx = x2 - x1; dy = y2 - y1
            dist = (dx*dx + dy*dy) ** 0.5
            v = dist - pa[k, 0]
            if v > 0.0: total += w * v * v
        elif tc == _T_FAR:
            dx = x2 - x1; dy = y2 - y1
            dist = (dx*dx + dy*dy) ** 0.5
            v = pa[k, 0] - dist
            if v > 0.0: total += w * v * v
        elif tc == _T_DIST_APPROX:
            dx = x2 - x1; dy = y2 - y1
            dist = (dx*dx + dy*dy) ** 0.5
            diff = dist - pa[k, 0]
            sigma = pa[k, 1]
            total += w * diff * diff / (2.0 * sigma * sigma)
        elif tc == _T_ACROSS:
            dx = x2 - x1; dy = y2 - y1
            dist = (dx*dx + dy*dy) ** 0.5
            v = dist - pa[k, 0]
            if v > 0.0: total += w * v * v
            ve = (x2 + pa[k, 1]) - x1
            if ve > 0.0: total += w * 0.3 * ve * ve
        elif tc == _T_IN_REGION:
            cx = pa[k, 0]; cy = pa[k, 1]; r = pa[k, 2]
            dx = x1 - cx; dy = y1 - cy
            dist = (dx*dx + dy*dy) ** 0.5
            v = dist - r
            if v > 0.0: total += w * v * v

    if not np.isfinite(total):
        return -np.inf
    return -_EMCEE_BETA * total


def _run_emcee(
    model: ConstraintModel,
    latent_names: List[str],
    ndim: int,
    num_walkers: int,
    total_steps: int,
    burn_in: int,
    init_positions: np.ndarray,
    beta: float,
    thin: int,
    rng: np.random.Generator,
    bbox_radius: float = 1000.0,
) -> List[dict]:
    """Run emcee ensemble sampler with multiprocessing across CPU cores."""
    import emcee
    from multiprocessing import Pool

    ctypes, idx1, idx2, weights, params_arr = _compile_constraints(model, latent_names)

    num_walkers = max(num_walkers, 2 * ndim + (2 * ndim) % 2)
    if num_walkers % 2 != 0:
        num_walkers += 1

    burn_stored = burn_in // thin
    n_workers = max(1, os.cpu_count() or 1)

    log.info(
        "emcee: %d walkers, %d dims, %d total steps (thin=%d, discard %d burn-in), %d CPU workers",
        num_walkers, ndim, total_steps, thin, burn_stored, n_workers,
    )

    p0 = init_positions[np.newaxis, :] + rng.normal(0, 1.0, (num_walkers, ndim))

    bounds = bbox_radius * 3.0

    # Set module-level state for the main process too (used if n_workers==1)
    _init_emcee_worker(ctypes, idx1, idx2, weights, params_arr, beta, bounds)

    if n_workers > 1:
        pool = Pool(
            processes=n_workers,
            initializer=_init_emcee_worker,
            initargs=(ctypes, idx1, idx2, weights, params_arr, beta, bounds),
        )
        sampler = emcee.EnsembleSampler(num_walkers, ndim, _emcee_log_prob, pool=pool)
    else:
        pool = None
        sampler = emcee.EnsembleSampler(num_walkers, ndim, _emcee_log_prob)

    try:
        sampler.run_mcmc(p0, total_steps, progress=True, thin_by=thin)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    flat_chain = sampler.get_chain(discard=burn_stored, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn_stored, flat=True)

    samples = []
    for i, (params, lp) in enumerate(zip(flat_chain, flat_log_prob)):
        if not np.isfinite(lp):
            continue
        E = -lp / beta if beta != 0 else 0.0
        samples.append({"step": i, "params": params.tolist(), "energy": float(E), "chain_id": 0})

    log.info("emcee: collected %d post-burn-in samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 6: Probabilistic Inference ===")

    dd = data_dir(cfg)
    constraints_path = dd / "constraints.json"
    samples_dir = dd / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if any(samples_dir.glob("*.jsonl")) and not force:
        log.info("Samples directory not empty, skipping (use --force to overwrite).")
        return

    # Clean old sample files when forcing
    for old in samples_dir.glob("*.jsonl"):
        old.unlink()

    model: ConstraintModel = read_json(constraints_path, model=ConstraintModel)

    inf_cfg = cfg.get("inference", {})
    method: str = inf_cfg.get("method", "metropolis")
    n_samples: int = inf_cfg.get("num_samples", 30_000)
    burn_in: int = inf_cfg.get("burn_in", 5_000)
    thin: int = inf_cfg.get("thin", 10)
    beta: float = inf_cfg.get("beta", 1.0)
    proposal_std: float = inf_cfg.get("proposal_std_km", 10.0)
    num_walkers: int = inf_cfg.get("num_walkers", 32)
    num_chains: int = inf_cfg.get("num_chains", 4)
    seed: int = inf_cfg.get("random_seed", 42)

    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    latent_entities = model.latent_entities
    if not latent_entities:
        log.warning("No latent entities found — nothing to infer.")
        return

    latent_names = [e.entity_id for e in latent_entities]
    ndim = 2 * len(latent_names)

    nearby_radius_km = inf_cfg.get("init_bbox_radius_km", 500.0)
    import math as _math
    nearby_fixed = [
        f for f in model.fixed_entities
        if _math.sqrt(f.x**2 + f.y**2) <= nearby_radius_km
    ]
    if not nearby_fixed:
        nearby_fixed = model.fixed_entities
    fixed_xs = [f.x for f in nearby_fixed]
    fixed_ys = [f.y for f in nearby_fixed]
    x_min, x_max, y_min, y_max = bounding_box_km(fixed_xs, fixed_ys, padding_km=30.0)

    init_params = np.zeros(ndim)
    for i in range(len(latent_names)):
        init_params[2 * i] = rng.uniform(x_min, x_max)
        init_params[2 * i + 1] = rng.uniform(y_min, y_max)

    log.info("Latent entities (%d): %s", len(latent_entities), [e.name for e in latent_entities])
    log.info("Method: %s, %d samples, burn_in=%d, thin=%d, ndim=%d", method, n_samples, burn_in, thin, ndim)

    total_steps = n_samples + burn_in
    all_raw_samples: List[dict] = []

    if method == "metropolis":
        model_dict = model.model_dump()
        chain_args = [
            (model_dict, latent_names, init_params.copy(), total_steps,
             proposal_std, beta, thin, seed, cid, burn_in)
            for cid in range(num_chains)
        ]

        n_workers = min(num_chains, os.cpu_count() or 1)
        log.info("Running %d Metropolis chains in parallel (%d workers)", num_chains, n_workers)

        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_run_single_chain, chain_args))
        else:
            results = [_run_single_chain(a) for a in chain_args]

        for chain_raw in results:
            all_raw_samples.extend(chain_raw)

    elif method == "emcee":
        raw = _run_emcee(
            model, latent_names, ndim, num_walkers, total_steps, burn_in,
            init_params, beta, thin, rng,
            bbox_radius=nearby_radius_km,
        )
        all_raw_samples = raw

    else:
        raise ValueError(f"Unknown inference method: {method!r}.  Choose 'metropolis' or 'emcee'.")

    log.info("Converting %d raw samples to structured format", len(all_raw_samples))

    chain_samples: Dict[int, List[Sample]] = {}
    for global_idx, raw in enumerate(all_raw_samples):
        chain_id = raw.get("chain_id", 0)
        params = raw["params"]
        entities_pos: Dict[str, EntityPosition] = {}
        for i, eid in enumerate(latent_names):
            entities_pos[eid] = EntityPosition(x=round(params[2 * i], 4), y=round(params[2 * i + 1], 4))

        s = Sample(
            sample_id=global_idx,
            entities=entities_pos,
            energy=round(raw["energy"], 6),
            chain_id=chain_id,
        )
        chain_samples.setdefault(chain_id, []).append(s)

    for chain_id, samples in chain_samples.items():
        out_path = samples_dir / f"chain_{chain_id:02d}.jsonl"
        with out_path.open("w") as fh:
            for s in samples:
                fh.write(s.model_dump_json() + "\n")
        log.info("Wrote %d samples to %s", len(samples), out_path)

    log.info("Phase 6 complete. Total samples: %d", sum(len(v) for v in chain_samples.values()))
