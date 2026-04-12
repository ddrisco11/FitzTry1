"""Tests for Strategy B modules: phase_mistral_joint and phase_graph_build."""

import json
import pytest

from src.phase_mistral_joint import (
    _normalize,
    _canonical_name,
    _classify_entity,
    _find_mentions,
    _parse_model_response,
    _validate_relation,
    _resolve_entity,
    _build_chunks,
    VALID_RELATION_TYPES,
    NULLABLE_ENTITY2_TYPES,
    KNOWN_FICTIONAL,
    SURFACE_NORM,
)
from src.phase_graph_build import (
    _is_quality_entity,
    _build_grounded_lookup,
    _resolve_to_grounded,
    _deduplicate,
)
from src.utils.schemas import (
    Entity,
    EntityMention,
    GroundedEntity,
    SentenceRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sentences():
    return [
        SentenceRecord(doc_id="gatsby", sentence_id="gatsby_s0001", text="I lived at West Egg."),
        SentenceRecord(doc_id="gatsby", sentence_id="gatsby_s0002", text="Across the bay was East Egg."),
        SentenceRecord(doc_id="gatsby", sentence_id="gatsby_s0003", text="We drove to New York."),
    ]


@pytest.fixture
def sample_grounded_entities():
    return [
        GroundedEntity(
            entity_id="e_0000", name="West Egg", canonical_name="West Egg",
            type="fictional", ner_label="GPE", mentions=[
                EntityMention(sentence_id="gatsby_s0001", char_start=14, char_end=22, source_text="...West Egg..."),
            ], mention_count=1, doc_ids=["gatsby"],
        ),
        GroundedEntity(
            entity_id="e_0001", name="East Egg", canonical_name="East Egg",
            type="fictional", ner_label="GPE", mentions=[
                EntityMention(sentence_id="gatsby_s0002", char_start=20, char_end=28, source_text="...East Egg..."),
            ], mention_count=1, doc_ids=["gatsby"],
        ),
        GroundedEntity(
            entity_id="e_0002", name="New York", canonical_name="New York",
            type="real", ner_label="GPE", latitude=40.7128, longitude=-74.006,
            confidence=0.95, mentions=[
                EntityMention(sentence_id="gatsby_s0003", char_start=13, char_end=21, source_text="...New York..."),
                EntityMention(sentence_id="gatsby_s0010", char_start=5, char_end=13, source_text="...New York..."),
            ], mention_count=2, doc_ids=["gatsby"],
        ),
        GroundedEntity(
            entity_id="e_0003", name="Long Island", canonical_name="Long Island",
            type="real", ner_label="GPE", latitude=40.789, longitude=-73.135,
            confidence=0.90, mentions=[
                EntityMention(sentence_id="gatsby_s0005", char_start=0, char_end=11, source_text="Long Island..."),
                EntityMention(sentence_id="gatsby_s0006", char_start=0, char_end=11, source_text="Long Island..."),
                EntityMention(sentence_id="gatsby_s0007", char_start=0, char_end=11, source_text="Long Island..."),
            ], mention_count=3, doc_ids=["gatsby"],
        ),
        GroundedEntity(
            entity_id="e_0004", name="Xanadu", canonical_name="Xanadu",
            type="real", ner_label="GPE", latitude=None, longitude=None,
            confidence=None, mentions=[
                EntityMention(sentence_id="gatsby_s0099", char_start=0, char_end=6, source_text="Xanadu..."),
            ], mention_count=1, doc_ids=["gatsby"],
        ),
    ]


# ---------------------------------------------------------------------------
# phase_mistral_joint tests
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_basic(self):
        assert _normalize("  East  Egg  ") == "east egg"

    def test_multiline(self):
        assert _normalize("New\n  York") == "new york"


class TestCanonicalName:
    def test_surface_norm(self):
        assert _canonical_name("N.Y.") == "New York"
        assert _canonical_name("L.I.") == "Long Island"

    def test_title_case(self):
        assert _canonical_name("east egg") == "East Egg"

    def test_already_cased(self):
        assert _canonical_name("New York") == "New York"


class TestClassifyEntity:
    def test_known_fictional(self):
        assert _classify_entity("East Egg", "real", set()) == "fictional"
        assert _classify_entity("West Egg", "uncertain", set()) == "fictional"

    def test_fictional_override(self):
        assert _classify_entity("Valley of Ashes", "real", {"Valley of Ashes"}) == "fictional"

    def test_model_hint_real(self):
        assert _classify_entity("Some Place", "real", set()) == "real"

    def test_model_hint_fictional(self):
        assert _classify_entity("Some Place", "fictional", set()) == "fictional"

    def test_heuristic_real_indicator(self):
        assert _classify_entity("New York", "uncertain", set()) == "real"
        assert _classify_entity("Chicago", "uncertain", set()) == "real"

    def test_heuristic_suffix(self):
        assert _classify_entity("Hudson River", "uncertain", set()) == "real"
        assert _classify_entity("Long Island", "uncertain", set()) == "real"

    def test_uncertain_default(self):
        assert _classify_entity("Obscuretown", "uncertain", set()) == "uncertain"


class TestFindMentions:
    def test_finds_mentions(self, sample_sentences):
        mentions = _find_mentions("West Egg", "I lived at West Egg.", sample_sentences[:1])
        assert len(mentions) == 1
        assert mentions[0].sentence_id == "gatsby_s0001"
        assert mentions[0].char_start == 11

    def test_no_match(self, sample_sentences):
        mentions = _find_mentions("Valley of Ashes", "I lived at West Egg.", sample_sentences[:1])
        assert len(mentions) == 0


class TestBuildChunks:
    def test_basic_chunking(self, sample_sentences):
        chunks = _build_chunks(sample_sentences, chunk_size=2, overlap=1)
        assert len(chunks) >= 2
        # Each chunk should have a chunk_id and sentences
        for chunk_id, sents in chunks:
            assert chunk_id.startswith("gatsby_chunk_")
            assert 1 <= len(sents) <= 2

    def test_empty_input(self):
        assert _build_chunks([]) == []

    def test_single_sentence(self, sample_sentences):
        chunks = _build_chunks(sample_sentences[:1], chunk_size=6)
        assert len(chunks) == 1


class TestResolveEntity:
    def test_exact_match(self):
        lookup = {"east egg": "East Egg", "new york": "New York"}
        assert _resolve_entity("East Egg", lookup) == "East Egg"

    def test_case_insensitive(self):
        lookup = {"east egg": "East Egg"}
        assert _resolve_entity("east egg", lookup) == "East Egg"

    def test_substring_fallback(self):
        lookup = {"new york": "New York"}
        assert _resolve_entity("New York City", lookup) == "New York"

    def test_no_match(self):
        lookup = {"east egg": "East Egg"}
        assert _resolve_entity("Chicago", lookup) is None

    def test_empty(self):
        assert _resolve_entity("", {}) is None
        assert _resolve_entity(None, {}) is None


class TestValidateRelation:
    def test_valid_relation(self):
        lookup = {"east egg": "East Egg", "west egg": "West Egg"}
        raw = {
            "entity_1": "East Egg",
            "relation_type": "across",
            "entity_2": "West Egg",
            "confidence": 0.9,
            "evidence": "across the bay",
        }
        result = _validate_relation(raw, lookup, "East Egg was across the bay from West Egg")
        assert result is not None
        assert result["entity_1"] == "East Egg"
        assert result["entity_2"] == "West Egg"
        assert result["relation_type"] == "across"

    def test_invalid_type(self):
        lookup = {"east egg": "East Egg"}
        raw = {"entity_1": "East Egg", "relation_type": "unknown_type", "entity_2": "West Egg"}
        assert _validate_relation(raw, lookup, "") is None

    def test_missing_entity_1(self):
        lookup = {"east egg": "East Egg"}
        raw = {"entity_1": "Unknown Place", "relation_type": "near", "entity_2": "East Egg"}
        assert _validate_relation(raw, lookup, "") is None

    def test_self_relation(self):
        lookup = {"east egg": "East Egg"}
        raw = {"entity_1": "East Egg", "relation_type": "near", "entity_2": "East Egg"}
        assert _validate_relation(raw, lookup, "") is None

    def test_null_entity_2_for_on_coast(self):
        lookup = {"east egg": "East Egg"}
        raw = {"entity_1": "East Egg", "relation_type": "on_coast", "entity_2": None}
        result = _validate_relation(raw, lookup, "")
        assert result is not None
        assert result["entity_2"] is None

    def test_null_entity_2_for_near_rejected(self):
        lookup = {"east egg": "East Egg"}
        raw = {"entity_1": "East Egg", "relation_type": "near", "entity_2": None}
        assert _validate_relation(raw, lookup, "") is None

    def test_confidence_clamped(self):
        lookup = {"east egg": "East Egg", "west egg": "West Egg"}
        raw = {
            "entity_1": "East Egg", "relation_type": "near",
            "entity_2": "West Egg", "confidence": 5.0,
        }
        result = _validate_relation(raw, lookup, "")
        assert result["confidence"] <= 1.0


class TestParseModelResponse:
    def test_valid_joint_response(self, sample_sentences):
        response = json.dumps({
            "entities": [
                {"name": "West Egg", "ner_label": "GPE", "classification": "fictional"},
                {"name": "East Egg", "ner_label": "GPE", "classification": "fictional"},
            ],
            "relations": [
                {
                    "entity_1": "West Egg", "relation_type": "across",
                    "entity_2": "East Egg", "confidence": 0.9,
                    "evidence": "across the bay",
                }
            ],
        })
        chunk_text = "I lived at West Egg. Across the bay was East Egg."
        entities, relations = _parse_model_response(
            response, chunk_text, sample_sentences[:2], "gatsby",
        )
        assert "West Egg" in entities
        assert "East Egg" in entities
        assert len(relations) == 1
        assert relations[0]["relation_type"] == "across"

    def test_empty_response(self, sample_sentences):
        response = json.dumps({"entities": [], "relations": []})
        entities, relations = _parse_model_response(response, "", sample_sentences[:1], "gatsby")
        assert len(entities) == 0
        assert len(relations) == 0

    def test_malformed_json(self, sample_sentences):
        entities, relations = _parse_model_response(
            "not json at all", "", sample_sentences[:1], "gatsby",
        )
        assert len(entities) == 0
        assert len(relations) == 0

    def test_salvage_json(self, sample_sentences):
        response = 'Here is the result: {"entities": [{"name": "New York", "ner_label": "GPE", "classification": "real"}], "relations": []}'
        entities, relations = _parse_model_response(
            response, "We drove to New York.", sample_sentences[2:3], "gatsby",
        )
        assert "New York" in entities


# ---------------------------------------------------------------------------
# phase_graph_build tests
# ---------------------------------------------------------------------------

class TestIsQualityEntity:
    def test_fictional_passes(self, sample_grounded_entities):
        west_egg = sample_grounded_entities[0]
        assert _is_quality_entity(west_egg, {"West Egg"})

    def test_real_with_coords_and_mentions(self, sample_grounded_entities):
        new_york = sample_grounded_entities[2]
        assert _is_quality_entity(new_york, set())

    def test_real_with_many_mentions(self, sample_grounded_entities):
        long_island = sample_grounded_entities[3]
        assert _is_quality_entity(long_island, set())

    def test_real_no_coords_rejected(self, sample_grounded_entities):
        xanadu = sample_grounded_entities[4]
        assert not _is_quality_entity(xanadu, set())

    def test_short_name_rejected(self):
        ent = GroundedEntity(
            entity_id="e_x", name="NY", canonical_name="NY",
            type="real", ner_label="GPE", latitude=40.7, longitude=-74.0,
            confidence=0.9, mentions=[
                EntityMention(sentence_id="s1", char_start=0, char_end=2, source_text="NY"),
                EntityMention(sentence_id="s2", char_start=0, char_end=2, source_text="NY"),
            ], mention_count=2, doc_ids=["gatsby"],
        )
        assert not _is_quality_entity(ent, set())


class TestBuildGroundedLookup:
    def test_builds_lookup(self, sample_grounded_entities):
        lookup = _build_grounded_lookup(sample_grounded_entities)
        assert lookup["west egg"] == "West Egg"
        assert lookup["new york"] == "New York"
        assert lookup["long island"] == "Long Island"


class TestResolveToGrounded:
    def test_exact(self):
        lookup = {"east egg": "East Egg", "new york": "New York"}
        assert _resolve_to_grounded("East Egg", lookup) == "East Egg"

    def test_substring(self):
        lookup = {"new york": "New York"}
        assert _resolve_to_grounded("New York City", lookup) == "New York"

    def test_no_match(self):
        lookup = {"east egg": "East Egg"}
        assert _resolve_to_grounded("Chicago", lookup) is None


class TestDeduplicate:
    def test_keeps_highest_confidence(self):
        relations = [
            {"entity_1": "East Egg", "relation_type": "near", "entity_2": "West Egg", "confidence": 0.7},
            {"entity_1": "East Egg", "relation_type": "near", "entity_2": "West Egg", "confidence": 0.9},
        ]
        result = _deduplicate(relations)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_different_types_preserved(self):
        relations = [
            {"entity_1": "East Egg", "relation_type": "near", "entity_2": "West Egg", "confidence": 0.7},
            {"entity_1": "East Egg", "relation_type": "across", "entity_2": "West Egg", "confidence": 0.8},
        ]
        result = _deduplicate(relations)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_valid_relation_types(self):
        assert "near" in VALID_RELATION_TYPES
        assert "north_of" in VALID_RELATION_TYPES
        assert "distance_approx" in VALID_RELATION_TYPES

    def test_nullable_types(self):
        assert "on_coast" in NULLABLE_ENTITY2_TYPES
        assert "near" not in NULLABLE_ENTITY2_TYPES

    def test_known_fictional(self):
        assert "East Egg" in KNOWN_FICTIONAL
        assert "West Egg" in KNOWN_FICTIONAL
        assert "New York" not in KNOWN_FICTIONAL

    def test_surface_norm(self):
        assert SURFACE_NORM["N.Y."] == "New York"
        assert SURFACE_NORM["L.I."] == "Long Island"
