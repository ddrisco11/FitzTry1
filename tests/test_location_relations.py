"""Tests for the overhauled Phase 4 location–location spatial relation extractor."""

import json

import pytest

from src.phase4_relations import (
    INDICATOR_LEXICON,
    VALID_SEMANTIC_TYPES,
    _infer_semantic_types,
    _is_likely_metaphorical,
    _parse_response,
    _span_in_sentence,
    _validate_relation,
)
from src.utils.schemas import LocationRelation, SentenceLocationRelations


# ---------------------------------------------------------------------------
# Lexicon / helpers
# ---------------------------------------------------------------------------

class TestLexicon:
    def test_all_lexicon_values_are_valid_types(self):
        for indicator, t in INDICATOR_LEXICON.items():
            assert t in VALID_SEMANTIC_TYPES, f"{indicator}→{t} not in valid set"

    def test_infer_region(self):
        assert _infer_semantic_types("in") == ["REGION"]
        assert _infer_semantic_types("on") == ["REGION"]
        assert _infer_semantic_types("inside") == ["REGION"]

    def test_infer_distance(self):
        assert _infer_semantic_types("near") == ["DISTANCE"]
        assert _infer_semantic_types("close to") == ["DISTANCE"]
        assert _infer_semantic_types("far from") == ["DISTANCE"]

    def test_infer_direction(self):
        assert _infer_semantic_types("toward") == ["DIRECTION"]
        assert _infer_semantic_types("into") == ["DIRECTION"]

    def test_unknown_indicator(self):
        assert _infer_semantic_types("xyzzy") == []

    def test_span_check_case_insensitive(self):
        assert _span_in_sentence("the Park", "The park is here.")
        assert not _span_in_sentence("the lake", "The park is here.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidate:
    def test_valid_relation(self):
        sentence = "The park is near the river."
        rel = _validate_relation(
            {
                "location_1": "the park",
                "location_2": "the river",
                "spatial_indicator": "near",
                "semantic_type": ["DISTANCE"],
            },
            sentence,
        )
        assert rel is not None
        assert rel.location_1 == "the park"
        assert rel.location_2 == "the river"
        assert rel.spatial_indicator == "near"
        assert rel.semantic_type == ["DISTANCE"]

    def test_missing_field_rejected(self):
        sentence = "The park is near the river."
        assert _validate_relation(
            {"location_1": "the park", "spatial_indicator": "near"},
            sentence,
        ) is None

    def test_self_relation_rejected(self):
        sentence = "The park is near the park."
        assert _validate_relation(
            {
                "location_1": "the park",
                "location_2": "the park",
                "spatial_indicator": "near",
                "semantic_type": ["DISTANCE"],
            },
            sentence,
        ) is None

    def test_span_not_in_sentence_rejected(self):
        sentence = "The park is near the river."
        assert _validate_relation(
            {
                "location_1": "the lake",   # not in sentence
                "location_2": "the river",
                "spatial_indicator": "near",
                "semantic_type": ["DISTANCE"],
            },
            sentence,
        ) is None

    def test_semantic_type_inferred_when_missing(self):
        sentence = "The cup sits on the table."
        rel = _validate_relation(
            {
                "location_1": "the cup",
                "location_2": "the table",
                "spatial_indicator": "on",
                # no semantic_type provided
            },
            sentence,
        )
        assert rel is not None
        assert rel.semantic_type == ["REGION"]

    def test_invalid_semantic_types_filtered(self):
        sentence = "The cup sits on the table."
        rel = _validate_relation(
            {
                "location_1": "the cup",
                "location_2": "the table",
                "spatial_indicator": "on",
                "semantic_type": ["BOGUS", "REGION"],
            },
            sentence,
        )
        assert rel is not None
        assert rel.semantic_type == ["REGION"]

    def test_metaphorical_in_love_rejected(self):
        sentence = "He was in love with her."
        # The model should never propose this, but if it does we drop it.
        assert _is_likely_metaphorical(sentence, "in", "he", "love")


# ---------------------------------------------------------------------------
# Spec example / non-example
# ---------------------------------------------------------------------------

class TestSpecExamples:
    def test_spec_example_park_river(self):
        sentence = "The park is near the river."
        raw = json.dumps({
            "location_relations": [
                {
                    "location_1": "the park",
                    "location_2": "the river",
                    "spatial_indicator": "near",
                    "semantic_type": ["DISTANCE"],
                }
            ]
        })
        relations = _parse_response(raw, sentence)
        assert len(relations) == 1
        r = relations[0]
        assert r.location_1 == "the park"
        assert r.location_2 == "the river"
        assert r.spatial_indicator == "near"
        assert r.semantic_type == ["DISTANCE"]

    def test_spec_nonexample_cat_table(self):
        # When the model correctly emits an empty list, we get nothing.
        sentence = "The cat is on the table."
        raw = json.dumps({"location_relations": []})
        assert _parse_response(raw, sentence) == []

    def test_multi_relation_sentence(self):
        sentence = "The book is on the table in the room."
        raw = json.dumps({
            "location_relations": [
                {
                    "location_1": "the book",
                    "location_2": "the table",
                    "spatial_indicator": "on",
                    "semantic_type": ["REGION"],
                },
                {
                    "location_1": "the table",
                    "location_2": "the room",
                    "spatial_indicator": "in",
                    "semantic_type": ["REGION"],
                },
            ]
        })
        relations = _parse_response(raw, sentence)
        assert len(relations) == 2
        assert {r.location_2 for r in relations} == {"the table", "the room"}

    def test_duplicate_relation_collapsed(self):
        sentence = "The park is near the river."
        raw = json.dumps({
            "location_relations": [
                {
                    "location_1": "the park",
                    "location_2": "the river",
                    "spatial_indicator": "near",
                    "semantic_type": ["DISTANCE"],
                },
                {
                    "location_1": "The Park",
                    "location_2": "The River",
                    "spatial_indicator": "Near",
                    "semantic_type": ["DISTANCE"],
                },
            ]
        })
        relations = _parse_response(raw, sentence)
        assert len(relations) == 1


# ---------------------------------------------------------------------------
# Parser robustness
# ---------------------------------------------------------------------------

class TestParser:
    def test_garbage_returns_empty(self):
        assert _parse_response("not json at all", "anything") == []

    def test_salvage_object_in_text(self):
        sentence = "The park is near the river."
        wrapped = (
            "Here you go: "
            + json.dumps({"location_relations": [
                {
                    "location_1": "the park",
                    "location_2": "the river",
                    "spatial_indicator": "near",
                    "semantic_type": ["DISTANCE"],
                }
            ]})
            + " hope that helps!"
        )
        relations = _parse_response(wrapped, sentence)
        assert len(relations) == 1


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

class TestSchema:
    def test_sentence_record_serialization(self):
        rec = SentenceLocationRelations(
            doc_id="doc1",
            sentence_id="doc1_s0001",
            sentence="The park is near the river.",
            location_relations=[
                LocationRelation(
                    location_1="the park",
                    location_2="the river",
                    spatial_indicator="near",
                    semantic_type=["DISTANCE"],
                )
            ],
        )
        d = rec.model_dump()
        assert d["doc_id"] == "doc1"
        assert d["location_relations"][0]["semantic_type"] == ["DISTANCE"]
        # Round-trip
        rec2 = SentenceLocationRelations.model_validate(d)
        assert rec2 == rec
