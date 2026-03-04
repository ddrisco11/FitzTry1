"""Tests for Phase 6 — Probabilistic Inference."""

import pytest
import numpy as np

from src.phase6_inference import build_energy_fn, _metropolis_chain
from src.utils.schemas import (
    ConstraintModel,
    ConstraintSpec,
    CoordinateSystem,
    FixedEntitySpec,
    LatentEntitySpec,
)


# ---------------------------------------------------------------------------
# Toy problem: 3 entities, 2 constraints
#
# Fixed: A at (0, 0)
# Latent: B and C
# Constraints:
#   1. B is north of A  (y_B > 0)
#   2. C is near A      (||C - A|| < 10 km)
#
# Expected posterior: B concentrated at y > 0; C concentrated near (0, 0).
# ---------------------------------------------------------------------------

def _make_toy_model() -> ConstraintModel:
    return ConstraintModel(
        fixed_entities=[
            FixedEntitySpec(entity_id="a", name="A", x=0.0, y=0.0),
        ],
        latent_entities=[
            LatentEntitySpec(entity_id="b", name="B"),
            LatentEntitySpec(entity_id="c", name="C"),
        ],
        constraints=[
            ConstraintSpec(
                constraint_id="c_0001",
                type="north_of",
                entities=["b", "a"],
                params={"epsilon_km": 1.0},
                weight=1.0,
                source_relation_id="r_0001",
            ),
            ConstraintSpec(
                constraint_id="c_0002",
                type="near",
                entities=["c", "a"],
                params={"d_near_km": 10.0},
                weight=1.0,
                source_relation_id="r_0002",
            ),
        ],
        coordinate_system=CoordinateSystem(
            origin_lat=40.7128,
            origin_lon=-74.0060,
        ),
    )


class TestEnergyFunction:
    def setup_method(self):
        self.model = _make_toy_model()
        self.latent_names = ["b", "c"]
        self.energy_fn = build_energy_fn(self.model, self.latent_names)

    def test_zero_energy_when_satisfied(self):
        # B at (0, 5) — north of A(0,0) by 5 km  ✓
        # C at (0, 0) — same as A  ✓ (distance 0 < 10)
        params = np.array([0.0, 5.0, 0.0, 0.0])
        E = self.energy_fn(params)
        assert E == pytest.approx(0.0, abs=1e-9)

    def test_positive_energy_when_violated(self):
        # B at (0, -5) — south of A → north_of violated
        # C at (0, 0) — OK
        params = np.array([0.0, -5.0, 0.0, 0.0])
        E = self.energy_fn(params)
        assert E > 0.0

    def test_energy_is_nonnegative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            params = rng.uniform(-50, 50, 4)
            assert self.energy_fn(params) >= 0.0

    def test_energy_decreases_toward_constraint_satisfaction(self):
        # Start far from satisfied, move toward it — energy should drop
        params_bad = np.array([0.0, -20.0, 30.0, 30.0])
        params_good = np.array([0.0, 10.0, 2.0, 2.0])
        assert self.energy_fn(params_good) < self.energy_fn(params_bad)


class TestMetropolisSampler:
    def setup_method(self):
        self.model = _make_toy_model()
        self.latent_names = ["b", "c"]
        self.energy_fn = build_energy_fn(self.model, self.latent_names)

    def test_sampler_runs_and_returns_samples(self):
        rng = np.random.default_rng(42)
        init = np.array([0.0, 0.0, 0.0, 0.0])
        samples = _metropolis_chain(
            self.energy_fn, init, n_steps=200, proposal_std=2.0,
            beta=1.0, thin=10, rng=rng, chain_id=0,
        )
        assert len(samples) == 20  # 200 steps / thin=10
        for s in samples:
            assert "params" in s
            assert "energy" in s
            assert len(s["params"]) == 4

    def test_energy_tends_to_decrease_over_run(self):
        """After burn-in, mean energy of later samples should be <= early samples."""
        rng = np.random.default_rng(7)
        init = np.array([0.0, -30.0, 40.0, 40.0])  # start far from satisfied
        samples = _metropolis_chain(
            self.energy_fn, init, n_steps=5000, proposal_std=3.0,
            beta=2.0, thin=50, rng=rng, chain_id=0,
        )
        energies = [s["energy"] for s in samples]
        early = np.mean(energies[:10])
        late = np.mean(energies[-10:])
        assert late <= early + 5.0  # allow some slack for stochasticity

    def test_toy_converges_to_correct_posterior(self):
        """
        After sufficient sampling, B should mostly have y > 0,
        and C should be within 20 km of origin.
        """
        rng = np.random.default_rng(99)
        init = np.array([0.0, 10.0, 0.0, 0.0])
        samples = _metropolis_chain(
            self.energy_fn, init, n_steps=10_000, proposal_std=2.0,
            beta=2.0, thin=10, rng=rng, chain_id=0,
        )
        # Discard first 200 as burn-in
        samples = samples[200:]

        b_y_vals = [s["params"][1] for s in samples]   # y of B (index 1)
        c_x_vals = [s["params"][2] for s in samples]   # x of C (index 2)
        c_y_vals = [s["params"][3] for s in samples]   # y of C (index 3)

        # B is mostly north of origin (y > 0)
        frac_north = np.mean([y > 0 for y in b_y_vals])
        assert frac_north > 0.7, f"B is north only {frac_north:.1%} of the time"

        # C is mostly near origin (within 20 km)
        c_dists = np.sqrt(np.array(c_x_vals)**2 + np.array(c_y_vals)**2)
        frac_near = np.mean(c_dists < 20.0)
        assert frac_near > 0.7, f"C is near origin only {frac_near:.1%} of the time"
