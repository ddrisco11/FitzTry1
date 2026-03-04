"""Phase 3 — Geographic Grounding.

Geocodes 'real' and 'uncertain' entities via Nominatim.  Results are cached
to avoid redundant API calls.  Entities that fail geocoding are reclassified
as 'fictional'.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from src.utils.io import read_jsonl, write_jsonl, write_json, read_json, data_dir
from src.utils.schemas import Entity, GroundedEntity, EntityMention
from src.utils.geo import haversine_km

log = logging.getLogger(__name__)

# Known fictional names that should never be geocoded
KNOWN_FICTIONAL = {
    "East Egg",
    "West Egg",
    "Valley of Ashes",
    "Eggs",
}

# Manual coordinate overrides for entities that Nominatim systematically misgrounds.
# Format: canonical_name -> (lat, lon, confidence)
COORDINATE_OVERRIDES: Dict[str, tuple] = {
    "Normandy":    (49.1829, -0.3707, 0.95),    # Normandy, France
    "Marseilles":  (43.2965, 5.3698, 0.95),      # Marseille, France
    "Versailles":  (48.8014, 2.1301, 0.95),      # Versailles, France
    "Castile":     (39.4, -3.0, 0.90),            # Castile, Spain
    "Frisco":      (37.7749, -122.4194, 0.85),    # San Francisco, CA
    "the Pennsylvania Station": (40.7506, -73.9935, 0.95),
    "Pennsylvania Station":     (40.7506, -73.9935, 0.95),
}


# ---------------------------------------------------------------------------
# Geocoding cache
# ---------------------------------------------------------------------------

class GeocodeCache:
    """Simple JSON-backed cache for Nominatim results."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Optional[dict]] = {}
        if path.exists():
            with path.open() as fh:
                self._data = json.load(fh)
            log.info("Loaded geocode cache: %d entries from %s", len(self._data), path)

    def get(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    def set(self, key: str, value: Optional[dict]) -> None:
        self._data[key] = value
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as fh:
            json.dump(self._data, fh, indent=2)

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Disambiguation
# ---------------------------------------------------------------------------

def _disambiguate(
    candidates,
    country_bias: str,
    co_occurring_coords: List[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Pick the best candidate from multiple geocoder results.

    Priority:
    1. If co-occurring grounded entities exist, pick the candidate closest
       to their centroid (geographic context is the strongest signal).
    2. Otherwise, prefer candidates matching the country bias if set.
    Returns (lat, lon, confidence).
    """
    if not candidates:
        return None, None, 0.0

    pool = list(candidates)

    if co_occurring_coords:
        mean_lat = sum(x[0] for x in co_occurring_coords) / len(co_occurring_coords)
        mean_lon = sum(x[1] for x in co_occurring_coords) / len(co_occurring_coords)
        best = min(pool, key=lambda c: haversine_km(c.latitude, c.longitude, mean_lat, mean_lon))
        return best.latitude, best.longitude, 0.85

    if country_bias:
        biased = [c for c in pool if _matches_country(c, country_bias)]
        if biased:
            c = biased[0]
            return c.latitude, c.longitude, 0.85

    # No context — take the first (most relevant per Nominatim ranking)
    c = pool[0]
    return c.latitude, c.longitude, 0.6


def _matches_country(candidate, bias: str) -> bool:
    addr = getattr(candidate, "raw", {}).get("display_name", "")
    if bias == "US":
        return "United States" in addr or "USA" in addr
    return bias.lower() in addr.lower()


# ---------------------------------------------------------------------------
# Main phase function
# ---------------------------------------------------------------------------

def run(cfg: dict, force: bool = False) -> None:
    log.info("=== Phase 3: Geographic Grounding ===")

    dd = data_dir(cfg)
    entities_path = dd / "entities.jsonl"
    grounded_path = dd / "grounded_entities.jsonl"

    if grounded_path.exists() and not force:
        log.info("grounded_entities.jsonl exists, skipping (use --force to overwrite).")
        return

    grounding_cfg = cfg.get("grounding", {})
    user_agent = grounding_cfg.get("user_agent", "fitzgerald_geo_project")
    cache_file = Path(grounding_cfg.get("cache_file", str(dd / "geocode_cache.json")))
    sleep_sec = grounding_cfg.get("rate_limit_sleep", 1.1)
    country_bias = grounding_cfg.get("default_country_bias", "US")
    fictional_overrides = set(cfg.get("ner", {}).get("fictional_overrides", [])) | KNOWN_FICTIONAL

    entities: List[Entity] = read_jsonl(entities_path, model=Entity)
    log.info("Loaded %d entities for grounding", len(entities))

    geocoder = Nominatim(user_agent=user_agent)
    cache = GeocodeCache(cache_file)

    # Build a map of already-grounded real coords for disambiguation
    grounded_coords: Dict[str, Tuple[float, float]] = {}  # name → (lat, lon)

    grounded: List[GroundedEntity] = []

    # Load any extra overrides from config
    config_overrides = cfg.get("grounding", {}).get("coordinate_overrides", {})
    all_overrides = dict(COORDINATE_OVERRIDES)
    for name_key, coords in config_overrides.items():
        all_overrides[name_key] = (coords["lat"], coords["lon"], coords.get("confidence", 0.9))

    for entity in entities:
        name = entity.canonical_name

        # Fictional overrides -> always ungrounded
        if name in fictional_overrides or entity.type == "fictional":
            grounded.append(_to_grounded(entity, None, None, None))
            continue

        # Check for manual coordinate overrides first
        if name in all_overrides:
            lat, lon, confidence = all_overrides[name]
            grounded_coords[name] = (lat, lon)
            log.info("Override grounding for '%s' → (%.4f, %.4f)", name, lat, lon)
            ge = GroundedEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                canonical_name=entity.canonical_name,
                type="real",
                ner_label=entity.ner_label,
                latitude=lat,
                longitude=lon,
                confidence=confidence,
                mentions=entity.mentions,
                mention_count=entity.mention_count,
                doc_ids=entity.doc_ids,
            )
            grounded.append(ge)
            continue

        # Try to geocode
        lat, lon, confidence = _geocode(
            name, geocoder, cache, sleep_sec, country_bias,
            list(grounded_coords.values()),
        )

        if lat is not None:
            entity_type = "real"
            grounded_coords[name] = (lat, lon)
            log.debug("Grounded '%s' → (%.4f, %.4f) conf=%.2f", name, lat, lon, confidence)
        else:
            entity_type = "fictional"
            confidence = None
            log.info("Could not geocode '%s' → classifying as fictional", name)

        ge = GroundedEntity(
            entity_id=entity.entity_id,
            name=entity.name,
            canonical_name=entity.canonical_name,
            type=entity_type,
            ner_label=entity.ner_label,
            latitude=lat,
            longitude=lon,
            confidence=confidence,
            mentions=entity.mentions,
            mention_count=entity.mention_count,
            doc_ids=entity.doc_ids,
        )
        grounded.append(ge)

    n_real = sum(1 for g in grounded if g.type == "real")
    n_fict = sum(1 for g in grounded if g.type == "fictional")
    log.info("Grounding complete: %d real, %d fictional", n_real, n_fict)

    write_jsonl(grounded_path, grounded, overwrite=True)
    log.info("Phase 3 complete. Written to %s", grounded_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _geocode(
    name: str,
    geocoder: Nominatim,
    cache: GeocodeCache,
    sleep_sec: float,
    country_bias: str,
    co_occurring_coords: List[Tuple[float, float]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (lat, lon, confidence) for *name*, using cache when possible."""
    cache_key = name.lower()

    if cache_key in cache:
        cached = cache.get(cache_key)
        if cached is None:
            return None, None, None
        return cached["lat"], cached["lon"], cached["confidence"]

    # Rate-limited API call
    time.sleep(sleep_sec)
    try:
        results = geocoder.geocode(name, exactly_one=False, addressdetails=True, limit=5)
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        log.warning("Geocoder error for '%s': %s", name, exc)
        cache.set(cache_key, None)
        return None, None, None

    if not results:
        cache.set(cache_key, None)
        return None, None, None

    lat, lon, confidence = _disambiguate(results, country_bias, co_occurring_coords)
    if lat is not None:
        cache.set(cache_key, {"lat": lat, "lon": lon, "confidence": confidence})
    else:
        cache.set(cache_key, None)
    return lat, lon, confidence


def _to_grounded(
    entity: Entity,
    lat: Optional[float],
    lon: Optional[float],
    confidence: Optional[float],
) -> GroundedEntity:
    return GroundedEntity(
        entity_id=entity.entity_id,
        name=entity.name,
        canonical_name=entity.canonical_name,
        type="fictional" if lat is None else "real",
        ner_label=entity.ner_label,
        latitude=lat,
        longitude=lon,
        confidence=confidence,
        mentions=entity.mentions,
        mention_count=entity.mention_count,
        doc_ids=entity.doc_ids,
    )
