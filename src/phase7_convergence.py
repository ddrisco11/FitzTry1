"""Phase 7 — Convergence Diagnostics & Posterior Summaries.

Loads MCMC samples, computes:
- Gelman-Rubin R-hat per latent variable
- Effective sample size (ESS)
- Posterior mean, std, 95% credible ellipse
- Spatial entropy
- Multimodality detection (k-means BIC)
- Constraint satisfaction score
- Trace plots (saved as images)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

from src.utils.io import read_json, write_json, read_jsonl, data_dir
from src.utils.schemas import (
    ConstraintModel,
    CredibleRegion,
    GroundedEntity,
    PosteriorSummary,
    Sample,
)
from src.utils.geo import bounding_box_km

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Convergence statistics
# ---------------------------------------------------------------------------

def gelman_rubin(chains: List[np.ndarray]) -> float:
    """
    Gelman-Rubin R-hat for a single variable.
    *chains*: list of 1-D arrays, one per chain.  All must have the same length.
    """
    m = len(chains)
    n = len(chains[0])
    if m < 2 or n < 2:
        return float("nan")

    chain_means = np.array([c.mean() for c in chains])
    chain_vars = np.array([c.var(ddof=1) for c in chains])

    overall_mean = chain_means.mean()
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)
    W = chain_vars.mean()

    var_hat = (n - 1) / n * W + B / n
    r_hat = np.sqrt(var_hat / W) if W > 0 else float("nan")
    return float(r_hat)


def effective_sample_size(x: np.ndarray) -> float:
    """Estimate ESS via autocorrelation (Geyer's initial monotone sequence)."""
    n = len(x)
    if n < 4:
        return float(n)
    x = x - x.mean()
    acf = np.correlate(x, x, mode="full")[n - 1:]
    acf /= acf[0]
    # Sum pairs until the sum is negative
    rho_sum = 0.0
    for i in range(0, n - 1, 2):
        pair = acf[i] + acf[i + 1]
        if pair < 0:
            break
        rho_sum += pair
    tau = -1 + 2 * rho_sum
    return float(n / max(tau, 1.0))


# ---------------------------------------------------------------------------
# Credible region (95% ellipse)
# ---------------------------------------------------------------------------

def fit_credible_ellipse(xs: np.ndarray, ys: np.ndarray, level: float = 0.95) -> CredibleRegion:
    """
    Fit a 2-D ellipse to the (x, y) samples at the given credible level.
    Uses the eigendecomposition of the sample covariance.
    """
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    cov = np.cov(xs, ys)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Chi-squared quantile for 2 DOF
    chi2_q = stats.chi2.ppf(level, df=2)

    # Semi-axes lengths
    semi_major = float(np.sqrt(chi2_q * eigenvalues[-1]))
    semi_minor = float(np.sqrt(chi2_q * eigenvalues[0]))

    # Angle of major axis
    angle_rad = float(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    angle_deg = float(np.degrees(angle_rad))

    return CredibleRegion(
        type="ellipse",
        semi_major=round(semi_major, 4),
        semi_minor=round(semi_minor, 4),
        angle_deg=round(angle_deg, 2),
        center_x=round(center_x, 4),
        center_y=round(center_y, 4),
    )


# ---------------------------------------------------------------------------
# Spatial entropy
# ---------------------------------------------------------------------------

def spatial_entropy(xs: np.ndarray, ys: np.ndarray, n_bins: int = 20) -> float:
    """
    Estimate spatial entropy H = -Σ p log p via a 2-D histogram.
    Higher entropy → more uncertain placement.
    """
    H, _, _ = np.histogram2d(xs, ys, bins=n_bins)
    p = H.ravel()
    p = p[p > 0] / p.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Multimodality detection
# ---------------------------------------------------------------------------

def detect_modes(xs: np.ndarray, ys: np.ndarray, max_k: int = 5) -> int:
    """
    Run k-means for k = 1..max_k and pick k by BIC.
    Returns the detected number of modes.
    """
    data = np.column_stack([xs, ys])
    n = len(data)
    if n < max_k * 2:
        return 1

    best_k = 1
    best_bic = np.inf

    for k in range(1, max_k + 1):
        try:
            km = KMeans(n_clusters=k, n_init=5, random_state=0)
            km.fit(data)
            sse = km.inertia_
            # BIC approximation for k-means
            bic = sse / n + k * np.log(n) * 2 / n
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            break

    return best_k


# ---------------------------------------------------------------------------
# Constraint satisfaction
# ---------------------------------------------------------------------------

def constraint_satisfaction(
    samples: List[Sample],
    model: ConstraintModel,
    threshold: float = 0.1,
) -> Tuple[float, float]:
    """
    For each sample, compute the fraction of constraints with penalty < threshold.
    Returns (mean, std) across samples.
    """
    from src.utils.geo import (
        energy_north_of, energy_south_of, energy_east_of, energy_west_of,
        energy_near, energy_far, energy_distance_approx, energy_in_region,
        energy_co_occurrence,
    )

    fixed_xy: Dict[str, Tuple[float, float]] = {
        f.entity_id: (f.x, f.y) for f in model.fixed_entities
    }

    scores = []
    for sample in samples:
        n_satisfied = 0
        n_total = len(model.constraints)
        if n_total == 0:
            scores.append(1.0)
            continue

        for c in model.constraints:
            p = c.params
            ctype = c.type

            def get_xy(eid):
                if eid in fixed_xy:
                    return fixed_xy[eid]
                if eid in sample.entities:
                    pos = sample.entities[eid]
                    return pos.x, pos.y
                return None

            if len(c.entities) < 2 and ctype != "on_coast":
                n_total -= 1
                continue

            xy1 = get_xy(c.entities[0])
            xy2 = get_xy(c.entities[1]) if len(c.entities) > 1 else None

            if xy1 is None:
                n_total -= 1
                continue

            x1, y1 = xy1
            x2, y2 = (xy2 if xy2 else (x1, y1))

            if ctype == "north_of":
                pen = energy_north_of(y1, y2, p.get("epsilon_km", 1.0))
            elif ctype == "south_of":
                pen = energy_south_of(y1, y2, p.get("epsilon_km", 1.0))
            elif ctype == "east_of":
                pen = energy_east_of(x1, x2, p.get("epsilon_km", 1.0))
            elif ctype == "west_of":
                pen = energy_west_of(x1, x2, p.get("epsilon_km", 1.0))
            elif ctype == "near":
                pen = energy_near(x1, y1, x2, y2, p.get("d_near_km", 10.0))
            elif ctype == "far":
                pen = energy_far(x1, y1, x2, y2, p.get("d_far_km", 50.0))
            elif ctype == "distance_approx":
                pen = energy_distance_approx(x1, y1, x2, y2, p.get("target_d_km", 10.0), p.get("sigma_km", 5.0))
            elif ctype in ("across", "co_occurrence"):
                pen = energy_near(x1, y1, x2, y2, p.get("d_near_km", 10.0))
            else:
                pen = 0.0

            if pen < threshold:
                n_satisfied += 1

        scores.append(n_satisfied / n_total if n_total > 0 else 1.0)

    arr = np.array(scores)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------------

def save_trace_plots(
    chain_arrays: Dict[str, Dict[int, np.ndarray]],
    entity_names: Dict[str, str],
    out_dir: Path,
) -> None:
    """Save trace plots for each latent variable × coordinate."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        log.warning("matplotlib not available — skipping trace plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for entity_id, coord_chains in chain_arrays.items():
        name = entity_names.get(entity_id, entity_id)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        for coord_idx, coord_label in [(0, "x"), (1, "y")]:
            ax = axes[coord_idx]
            for chain_id, samples_1d in coord_chains.items():
                ax.plot(samples_1d, alpha=0.6, label=f"Chain {chain_id}", linewidth=0.5)
            ax.set_title(f"{name} — {coord_label}")
            ax.set_xlabel("Sample index")
            ax.set_ylabel(f"{coord_label} (km)")
            ax.legend(fontsize=6)
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("/", "_")
        fig.savefig(out_dir / f"trace_{safe_name}.png", dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 7: Convergence Diagnostics & Posterior Summaries ===")

    dd = data_dir(cfg)
    samples_dir = dd / "samples"
    convergence_dir = dd / "convergence"
    convergence_dir.mkdir(parents=True, exist_ok=True)
    constraints_path = dd / "constraints.json"

    summary_path = convergence_dir / "posterior_summaries.jsonl"
    if summary_path.exists() and not force:
        log.info("posterior_summaries.jsonl exists, skipping (use --force to overwrite).")
        return

    model: ConstraintModel = read_json(constraints_path, model=ConstraintModel)
    latent_ids = [e.entity_id for e in model.latent_entities]
    latent_names = {e.entity_id: e.name for e in model.latent_entities}

    if not latent_ids:
        log.warning("No latent entities — nothing to summarize.")
        return

    # --- Load all samples ---
    sample_files = sorted(samples_dir.glob("*.jsonl"))
    if not sample_files:
        raise FileNotFoundError(
            f"No sample files found in {samples_dir}.  "
            "Make sure Phase 6 has been run."
        )

    all_samples: List[Sample] = []
    chain_samples: Dict[int, List[Sample]] = {}

    for sf in sample_files:
        for s in read_jsonl(sf, model=Sample):
            all_samples.append(s)
            cid = s.chain_id or 0
            chain_samples.setdefault(cid, []).append(s)

    log.info("Loaded %d samples from %d chains", len(all_samples), len(chain_samples))

    # --- Build per-entity per-chain arrays ---
    # chain_arrays[entity_id][chain_id] = {"x": np.ndarray, "y": np.ndarray}
    chain_coord_arrays: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {
        eid: {} for eid in latent_ids
    }
    for cid, samples in chain_samples.items():
        for eid in latent_ids:
            xs = np.array([s.entities[eid].x for s in samples if eid in s.entities])
            ys = np.array([s.entities[eid].y for s in samples if eid in s.entities])
            chain_coord_arrays[eid][cid] = {"x": xs, "y": ys}

    # --- For trace plots ---
    trace_arrays: Dict[str, Dict[int, np.ndarray]] = {}
    for eid in latent_ids:
        trace_arrays[eid] = {
            cid: chain_coord_arrays[eid][cid]["x"]
            for cid in chain_coord_arrays[eid]
        }
    save_trace_plots(trace_arrays, latent_names, convergence_dir / "traces")

    # --- Compute constraint satisfaction ONCE (it's the same for all entities) ---
    log.info("Computing constraint satisfaction (sampling %d of %d samples)...",
             min(200, len(all_samples)), len(all_samples))
    cs_subset = all_samples[::max(1, len(all_samples) // 200)]
    cs_mean, cs_std = constraint_satisfaction(cs_subset, model)
    log.info("Constraint satisfaction: mean=%.4f std=%.4f", cs_mean, cs_std)

    # --- Posterior summaries ---
    summaries: List[PosteriorSummary] = []

    for eid in latent_ids:
        all_xs = np.concatenate(
            [chain_coord_arrays[eid][cid]["x"] for cid in chain_coord_arrays[eid]]
        )
        all_ys = np.concatenate(
            [chain_coord_arrays[eid][cid]["y"] for cid in chain_coord_arrays[eid]]
        )

        if len(all_xs) < 4:
            log.warning("Too few samples for entity %s", eid)
            continue

        mean_x = float(all_xs.mean())
        mean_y = float(all_ys.mean())
        std_x = float(all_xs.std())
        std_y = float(all_ys.std())

        credible = fit_credible_ellipse(all_xs, all_ys)
        entropy = spatial_entropy(all_xs, all_ys)
        n_modes = detect_modes(all_xs, all_ys)

        r_hat = None
        ess = None
        if len(chain_coord_arrays[eid]) >= 2:
            x_chains = [chain_coord_arrays[eid][cid]["x"] for cid in sorted(chain_coord_arrays[eid])]
            y_chains = [chain_coord_arrays[eid][cid]["y"] for cid in sorted(chain_coord_arrays[eid])]
            min_len = min(len(c) for c in x_chains)
            x_chains = [c[:min_len] for c in x_chains]
            y_chains = [c[:min_len] for c in y_chains]

            r_hat = {
                "x": round(gelman_rubin(x_chains), 4),
                "y": round(gelman_rubin(y_chains), 4),
            }
            ess = {
                "x": round(effective_sample_size(all_xs), 1),
                "y": round(effective_sample_size(all_ys), 1),
            }

        summaries.append(
            PosteriorSummary(
                entity_id=eid,
                name=latent_names[eid],
                posterior_mean={"x": round(mean_x, 4), "y": round(mean_y, 4)},
                posterior_std={"x": round(std_x, 4), "y": round(std_y, 4)},
                credible_region_95=credible,
                spatial_entropy=round(entropy, 4),
                num_modes=n_modes,
                r_hat=r_hat,
                ess=ess,
                constraint_satisfaction_mean=round(cs_mean, 4),
                constraint_satisfaction_std=round(cs_std, 4),
            )
        )
        log.info(
            "Entity '%s': mean=(%.2f, %.2f) std=(%.2f, %.2f) modes=%d entropy=%.2f",
            latent_names[eid], mean_x, mean_y, std_x, std_y, n_modes, entropy,
        )

    # Write summaries
    with summary_path.open("w") as fh:
        for s in summaries:
            fh.write(s.model_dump_json() + "\n")
    log.info("Phase 7 complete. Written %d summaries to %s", len(summaries), summary_path)
