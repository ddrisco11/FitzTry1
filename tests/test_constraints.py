"""Tests for Phase 5 — Formal Spatial Constraint Model (energy functions)."""

import math
import pytest
import numpy as np

from src.utils.geo import (
    energy_north_of,
    energy_south_of,
    energy_east_of,
    energy_west_of,
    energy_near,
    energy_far,
    energy_distance_approx,
    energy_in_region,
    energy_co_occurrence,
    latlon_to_km,
    km_to_latlon,
    haversine_km,
    euclidean_km,
    bounding_box_km,
)


# ---------------------------------------------------------------------------
# Directional constraints
# ---------------------------------------------------------------------------

class TestDirectionalConstraints:
    """Energy functions return 0 for satisfied constraints, >0 for violations."""

    def test_north_of_satisfied(self):
        # A at y=10, B at y=0 → A is north of B → satisfied
        assert energy_north_of(y_a=10.0, y_b=0.0, epsilon=1.0) == 0.0

    def test_north_of_violated(self):
        # A at y=0, B at y=10 → A is NOT north of B → violated
        assert energy_north_of(y_a=0.0, y_b=10.0, epsilon=1.0) > 0.0

    def test_north_of_boundary(self):
        # A exactly at y_B + epsilon → just satisfied
        assert energy_north_of(y_a=2.0, y_b=1.0, epsilon=1.0) == 0.0

    def test_south_of_satisfied(self):
        assert energy_south_of(y_a=0.0, y_b=10.0, epsilon=1.0) == 0.0

    def test_south_of_violated(self):
        assert energy_south_of(y_a=10.0, y_b=0.0, epsilon=1.0) > 0.0

    def test_east_of_satisfied(self):
        assert energy_east_of(x_a=10.0, x_b=0.0, epsilon=1.0) == 0.0

    def test_east_of_violated(self):
        assert energy_east_of(x_a=0.0, x_b=10.0, epsilon=1.0) > 0.0

    def test_west_of_satisfied(self):
        assert energy_west_of(x_a=0.0, x_b=10.0, epsilon=1.0) == 0.0

    def test_west_of_violated(self):
        assert energy_west_of(x_a=10.0, x_b=0.0, epsilon=1.0) > 0.0

    def test_directional_energy_grows_with_violation(self):
        # Larger violation → larger energy
        e_small = energy_north_of(y_a=0.0, y_b=5.0, epsilon=1.0)
        e_large = energy_north_of(y_a=0.0, y_b=20.0, epsilon=1.0)
        assert e_large > e_small

    def test_weight_scales_energy(self):
        e1 = energy_north_of(y_a=0.0, y_b=10.0, epsilon=1.0, weight=1.0)
        e2 = energy_north_of(y_a=0.0, y_b=10.0, epsilon=1.0, weight=2.0)
        assert pytest.approx(e2) == 2.0 * e1


# ---------------------------------------------------------------------------
# Proximity constraints
# ---------------------------------------------------------------------------

class TestProximityConstraints:
    def test_near_satisfied(self):
        # Distance = 5 km, threshold = 10 km → satisfied
        assert energy_near(0.0, 0.0, 5.0, 0.0, d_near=10.0) == 0.0

    def test_near_violated(self):
        # Distance = 20 km, threshold = 10 km → violated
        assert energy_near(0.0, 0.0, 20.0, 0.0, d_near=10.0) > 0.0

    def test_far_satisfied(self):
        # Distance = 60 km, threshold = 50 km → satisfied (they ARE far)
        assert energy_far(0.0, 0.0, 60.0, 0.0, d_far=50.0) == 0.0

    def test_far_violated(self):
        # Distance = 10 km, threshold = 50 km → violated (they're NOT far)
        assert energy_far(0.0, 0.0, 10.0, 0.0, d_far=50.0) > 0.0

    def test_distance_approx_satisfied(self):
        # Entities are exactly target_d apart → minimal energy
        e = energy_distance_approx(0.0, 0.0, 10.0, 0.0, target_d=10.0, sigma=5.0)
        assert pytest.approx(e, abs=1e-9) == 0.0

    def test_distance_approx_off_target(self):
        e = energy_distance_approx(0.0, 0.0, 20.0, 0.0, target_d=10.0, sigma=5.0)
        assert e > 0.0

    def test_in_region_inside(self):
        # Entity at (1, 1), centroid at (0, 0), radius 5 km → inside → 0
        e = energy_in_region(1.0, 1.0, 0.0, 0.0, radius=5.0)
        assert e == 0.0

    def test_in_region_outside(self):
        # Entity at (20, 0), centroid at (0, 0), radius 5 km → outside
        e = energy_in_region(20.0, 0.0, 0.0, 0.0, radius=5.0)
        assert e > 0.0


# ---------------------------------------------------------------------------
# Coordinate conversion round-trips
# ---------------------------------------------------------------------------

class TestCoordinateConversion:
    def test_roundtrip_latlon_km(self):
        origin_lat, origin_lon = 40.7128, -74.0060
        for lat, lon in [(40.8, -73.9), (40.5, -74.2), (40.7128, -74.0060)]:
            x, y = latlon_to_km(lat, lon, origin_lat, origin_lon)
            lat2, lon2 = km_to_latlon(x, y, origin_lat, origin_lon)
            assert pytest.approx(lat, abs=0.001) == lat2
            assert pytest.approx(lon, abs=0.001) == lon2

    def test_origin_maps_to_zero(self):
        origin_lat, origin_lon = 40.7128, -74.0060
        x, y = latlon_to_km(origin_lat, origin_lon, origin_lat, origin_lon)
        assert pytest.approx(x, abs=1e-9) == 0.0
        assert pytest.approx(y, abs=1e-9) == 0.0

    def test_east_is_positive_x(self):
        x, y = latlon_to_km(40.7128, -73.0, 40.7128, -74.0060)
        assert x > 0

    def test_north_is_positive_y(self):
        x, y = latlon_to_km(41.0, -74.0060, 40.7128, -74.0060)
        assert y > 0

    def test_haversine_nyc_la(self):
        # NYC → LA ≈ 3940 km
        d = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3900 < d < 4000

    def test_bounding_box_with_padding(self):
        xs, ys = [0.0, 10.0], [0.0, 10.0]
        x_min, x_max, y_min, y_max = bounding_box_km(xs, ys, padding_km=5.0)
        assert x_min == pytest.approx(-5.0)
        assert x_max == pytest.approx(15.0)
        assert y_min == pytest.approx(-5.0)
        assert y_max == pytest.approx(15.0)
