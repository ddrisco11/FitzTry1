"""Pydantic v2 models defining the data contracts between pipeline phases."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 — Corpus Preparation
# ---------------------------------------------------------------------------

class SentenceRecord(BaseModel):
    """One sentence from the cleaned corpus."""
    doc_id: str
    sentence_id: str
    text: str


# ---------------------------------------------------------------------------
# Phase 2 — Named Entity Recognition
# ---------------------------------------------------------------------------

class EntityMention(BaseModel):
    """A single occurrence of an entity in the text."""
    sentence_id: str
    char_start: int
    char_end: int
    source_text: str


class Entity(BaseModel):
    """A deduplicated geographic entity extracted from the corpus."""
    entity_id: str
    name: str
    canonical_name: str
    type: str                    # "real" | "fictional" | "uncertain"
    ner_label: str               # GPE | LOC | FAC
    mentions: List[EntityMention]
    mention_count: int
    doc_ids: List[str]


# ---------------------------------------------------------------------------
# Phase 3 — Geographic Grounding
# ---------------------------------------------------------------------------

class GroundedEntity(BaseModel):
    """Entity enriched with geocoordinates (None for fictional entities)."""
    entity_id: str
    name: str
    canonical_name: str
    type: str
    ner_label: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: Optional[float] = None
    mentions: List[EntityMention]
    mention_count: int
    doc_ids: List[str]


# ---------------------------------------------------------------------------
# Phase 4 — Spatial Relation Extraction
# ---------------------------------------------------------------------------

class SpatialRelation(BaseModel):
    """A pairwise spatial constraint extracted from the text."""
    relation_id: str
    type: str                        # near | far | north_of | south_of | east_of |
                                     # west_of | across | on_coast | in_region |
                                     # distance_approx | co_occurrence
    entity_1: str                    # canonical_name of first entity
    entity_2: Optional[str] = None  # None for unary relations (on_coast)
    direction: Optional[str] = None
    distance_value: Optional[float] = None
    distance_unit: Optional[str] = None
    weight: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    source_sentence_id: str
    source_text: str
    extraction_method: str           # pattern_match | co_occurrence | hf_zero_shot


# ---------------------------------------------------------------------------
# Phase 4 (overhauled) — Location–Location Spatial Relation Extraction
#
# Schema follows the Spatial Role Labeling annotation tradition; cf.
#   Kordjamshidi, P., van Otterlo, M., & Moens, M.-F. (2017).
#   "Spatial Role Labeling Annotation Scheme."
#   In N. Ide & J. Pustejovsky (Eds.), Handbook of Linguistic Annotation.
#   Springer.
# Each LocationRelation captures a (trajector, spatial_indicator, landmark)
# triple where both trajector and landmark are LOCATION spans.
# ---------------------------------------------------------------------------

class LocationRelation(BaseModel):
    """A spatial relation whose two arguments are both location spans."""
    location_1: str                          # trajector — the located entity
    location_2: str                          # landmark — the reference entity
    spatial_indicator: str                   # verbatim spatial cue from the text
    semantic_type: List[str]                 # subset of {"REGION", "DIRECTION", "DISTANCE"}


class SentenceLocationRelations(BaseModel):
    """All location–location relations grounded in a single sentence.

    Only emitted when the sentence contains at least one valid relation
    (i.e. two location spans linked by a spatial indicator).
    """
    doc_id: str
    sentence_id: str
    sentence: str
    location_relations: List[LocationRelation]


# ---------------------------------------------------------------------------
# Phase 5 — Formal Constraint Model
# ---------------------------------------------------------------------------

class FixedEntitySpec(BaseModel):
    entity_id: str
    name: str
    x: float   # km east of origin
    y: float   # km north of origin


class LatentEntitySpec(BaseModel):
    entity_id: str
    name: str
    init_x: Optional[float] = None
    init_y: Optional[float] = None


class ConstraintSpec(BaseModel):
    constraint_id: str
    type: str
    entities: List[str]              # entity_ids
    params: Dict[str, Any]
    weight: float
    source_relation_id: str


class CoordinateSystem(BaseModel):
    projection: str = "equirectangular"
    origin_lat: float
    origin_lon: float
    units: str = "km"


class ConstraintModel(BaseModel):
    fixed_entities: List[FixedEntitySpec]
    latent_entities: List[LatentEntitySpec]
    constraints: List[ConstraintSpec]
    coordinate_system: CoordinateSystem


# ---------------------------------------------------------------------------
# Phase 6 — Probabilistic Inference
# ---------------------------------------------------------------------------

class EntityPosition(BaseModel):
    x: float
    y: float


class Sample(BaseModel):
    sample_id: int
    entities: Dict[str, EntityPosition]   # entity_id -> position
    energy: float
    chain_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Phase 7 — Convergence Diagnostics
# ---------------------------------------------------------------------------

class CredibleRegion(BaseModel):
    type: str = "ellipse"
    semi_major: float
    semi_minor: float
    angle_deg: float
    center_x: float
    center_y: float


class PosteriorSummary(BaseModel):
    entity_id: str
    name: str
    posterior_mean: Dict[str, float]       # {"x": ..., "y": ...}
    posterior_std: Dict[str, float]        # {"x": ..., "y": ...}
    credible_region_95: CredibleRegion
    spatial_entropy: float
    num_modes: int
    r_hat: Optional[Dict[str, float]] = None   # None if single chain
    ess: Optional[Dict[str, float]] = None
    constraint_satisfaction_mean: Optional[float] = None
    constraint_satisfaction_std: Optional[float] = None
