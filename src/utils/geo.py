"""Geocoding + coordinate utilities.

All pipeline math operates in a local planar (km) system to avoid mixing
lat/lon with Euclidean distances.  The projection origin is defined in
config.yaml (constraints.projection_origin_lat / lon).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# Earth radius in km
_R_KM = 6371.0


# ---------------------------------------------------------------------------
# Projection: lat/lon  ↔  local planar (x, y) in km
# ---------------------------------------------------------------------------

def latlon_to_km(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """
    Equirectangular projection: convert (lat, lon) → (x, y) in kilometres.

    x increases eastward, y increases northward.
    The origin is at (origin_lat, origin_lon).
    """
    cos_lat = math.cos(math.radians(origin_lat))
    x = (lon - origin_lon) * cos_lat * _R_KM * math.pi / 180.0
    y = (lat - origin_lat) * _R_KM * math.pi / 180.0
    return x, y


def km_to_latlon(
    x: float,
    y: float,
    origin_lat: float,
    origin_lon: float,
) -> Tuple[float, float]:
    """Inverse of latlon_to_km: (x, y) in km → (lat, lon)."""
    cos_lat = math.cos(math.radians(origin_lat))
    lat = origin_lat + y * 180.0 / (math.pi * _R_KM)
    lon = origin_lon + x * 180.0 / (math.pi * _R_KM * cos_lat)
    return lat, lon


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return _R_KM * 2 * math.asin(math.sqrt(a))


def euclidean_km(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in the local planar km system."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------------------------------
# Constraint energy functions
# ---------------------------------------------------------------------------

def energy_north_of(
    y_a: float, y_b: float, epsilon: float = 1.0, weight: float = 1.0
) -> float:
    """A is north of B  ⟹  y_A > y_B + ε."""
    violation = max(0.0, (y_b + epsilon) - y_a)
    return weight * violation ** 2


def energy_south_of(
    y_a: float, y_b: float, epsilon: float = 1.0, weight: float = 1.0
) -> float:
    """A is south of B  ⟹  y_A < y_B - ε."""
    violation = max(0.0, y_a - (y_b - epsilon))
    return weight * violation ** 2


def energy_east_of(
    x_a: float, x_b: float, epsilon: float = 1.0, weight: float = 1.0
) -> float:
    """A is east of B  ⟹  x_A > x_B + ε."""
    violation = max(0.0, (x_b + epsilon) - x_a)
    return weight * violation ** 2


def energy_west_of(
    x_a: float, x_b: float, epsilon: float = 1.0, weight: float = 1.0
) -> float:
    """A is west of B  ⟹  x_A < x_B - ε."""
    violation = max(0.0, x_a - (x_b - epsilon))
    return weight * violation ** 2


def energy_near(
    x_a: float, y_a: float, x_b: float, y_b: float, d_near: float = 10.0, weight: float = 1.0
) -> float:
    """A is near B  ⟹  ||A - B|| < d_near."""
    dist = euclidean_km(x_a, y_a, x_b, y_b)
    violation = max(0.0, dist - d_near)
    return weight * violation ** 2


def energy_far(
    x_a: float, y_a: float, x_b: float, y_b: float, d_far: float = 50.0, weight: float = 1.0
) -> float:
    """A is far from B  ⟹  ||A - B|| > d_far."""
    dist = euclidean_km(x_a, y_a, x_b, y_b)
    violation = max(0.0, d_far - dist)
    return weight * violation ** 2


def energy_distance_approx(
    x_a: float, y_a: float, x_b: float, y_b: float,
    target_d: float, sigma: float = 5.0, weight: float = 1.0,
) -> float:
    """||A - B|| ≈ target_d  (Gaussian potential)."""
    dist = euclidean_km(x_a, y_a, x_b, y_b)
    return weight * (dist - target_d) ** 2 / (2 * sigma ** 2)


def energy_in_region(
    x_a: float, y_a: float,
    cx: float, cy: float, radius: float, weight: float = 1.0,
) -> float:
    """A is inside a circular region centred at (cx, cy) with given radius."""
    dist = euclidean_km(x_a, y_a, cx, cy)
    violation = max(0.0, dist - radius)
    return weight * violation ** 2


def energy_co_occurrence(
    x_a: float, y_a: float, x_b: float, y_b: float,
    d_near: float = 10.0, weight: float = 0.1,
) -> float:
    """Weak proximity pull for co-occurring entities (treated as soft near)."""
    return energy_near(x_a, y_a, x_b, y_b, d_near=d_near, weight=weight)


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def bounding_box_km(
    fixed_xs: list[float],
    fixed_ys: list[float],
    padding_km: float = 30.0,
) -> Tuple[float, float, float, float]:
    """
    Return (x_min, x_max, y_min, y_max) bounding box around fixed entities
    with *padding_km* added on each side.
    """
    if not fixed_xs:
        return -50.0, 50.0, -50.0, 50.0
    x_min = min(fixed_xs) - padding_km
    x_max = max(fixed_xs) + padding_km
    y_min = min(fixed_ys) - padding_km
    y_max = max(fixed_ys) + padding_km
    return x_min, x_max, y_min, y_max
