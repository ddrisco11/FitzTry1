"""Tests for Phase 4 — Spatial Relation Extraction."""

import pytest

from src.phase4_relations import (
    _extract_distance,
    _hedging_penalty,
    _is_likely_metaphorical,
    HEDGING_WORDS,
)


class TestDistancePattern:
    def test_miles(self):
        result = _extract_distance("It was about 20 miles from New York.")
        assert result is not None
        value, unit = result
        assert value == 20.0
        assert unit == "miles"

    def test_km(self):
        result = _extract_distance("Only 5 km from the city.")
        assert result is not None
        value, unit = result
        assert value == 5.0
        assert unit == "km"

    def test_decimal(self):
        result = _extract_distance("The house was 3.5 miles from town.")
        assert result is not None
        assert result[0] == 3.5

    def test_no_match(self):
        result = _extract_distance("He drove quickly to the station.")
        assert result is None

    def test_kilometers_spelled_out(self):
        result = _extract_distance("Located 12 kilometers from the border.")
        assert result is not None
        assert result[1] == "km"


class TestHedgingPenalty:
    def test_no_hedging(self):
        penalty = _hedging_penalty("east of the village")
        assert penalty == 0.0

    def test_single_hedge_word(self):
        penalty = _hedging_penalty("perhaps east of the village")
        assert penalty > 0.0

    def test_multiple_hedge_words(self):
        p1 = _hedging_penalty("perhaps east of the village")
        p2 = _hedging_penalty("maybe possibly somewhere east of the village")
        assert p2 > p1

    def test_capped_at_max(self):
        many_hedges = " ".join(HEDGING_WORDS)
        penalty = _hedging_penalty(many_hedges)
        assert penalty <= 0.4


class TestMetaphorDetection:
    def test_literal_is_not_metaphorical(self):
        assert not _is_likely_metaphorical("east of new york")

    def test_a_world_away(self):
        assert _is_likely_metaphorical("it seemed a world away from home")

    def test_worlds_apart(self):
        assert _is_likely_metaphorical("they were worlds apart")

    def test_in_shadow(self):
        assert _is_likely_metaphorical("the town lay in the shadow of the mountain")


class TestPatternExtraction:
    """Integration test — requires spaCy."""

    @pytest.fixture(scope="class")
    def nlp_and_matcher(self):
        pytest.importorskip("spacy")
        import spacy
        from src.phase4_relations import _build_patterns
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                pytest.skip("No spaCy model installed")
        matcher = _build_patterns(nlp)
        return nlp, matcher

    def _extract(self, text, nlp, matcher, method="pattern_match"):
        from src.phase4_relations import _extract_from_sentence
        entities = ["East Egg", "West Egg", "New York"]
        return _extract_from_sentence(
            text, "test_sent_0", entities, nlp, matcher,
            [method, "co_occurrence"],
            {"co_occurrence_weight": 0.3},
        )

    def test_near_pattern(self, nlp_and_matcher):
        nlp, matcher = nlp_and_matcher
        rels = self._extract("East Egg is close to West Egg.", nlp, matcher)
        types = [r.type for r in rels]
        assert "near" in types

    def test_north_of_pattern(self, nlp_and_matcher):
        nlp, matcher = nlp_and_matcher
        rels = self._extract("East Egg lies north of New York.", nlp, matcher)
        types = [r.type for r in rels]
        assert "north_of" in types

    def test_across_pattern(self, nlp_and_matcher):
        nlp, matcher = nlp_and_matcher
        rels = self._extract("East Egg was across the bay from West Egg.", nlp, matcher)
        types = [r.type for r in rels]
        assert "across" in types

    def test_uncertainty_in_range(self, nlp_and_matcher):
        nlp, matcher = nlp_and_matcher
        rels = self._extract("East Egg is near West Egg.", nlp, matcher)
        for r in rels:
            assert 0.0 <= r.uncertainty <= 1.0
            assert 0.0 <= r.weight <= 1.0

    def test_co_occurrence_weight(self, nlp_and_matcher):
        nlp, matcher = nlp_and_matcher
        rels = self._extract("Both East Egg and West Egg are described.", nlp, matcher)
        co = [r for r in rels if r.extraction_method == "co_occurrence"]
        for r in co:
            assert r.weight == pytest.approx(0.3, abs=0.01)
