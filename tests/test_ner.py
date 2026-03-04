"""Tests for Phase 2 — Named Entity Recognition."""

import pytest

from src.phase2_ner import _classify_entity, normalize_surface, KNOWN_FICTIONAL


SAMPLE_PARA = (
    "He drove from West Egg to New York City, passing through the Valley of Ashes "
    "on his way to the party in Manhattan.  East Egg glittered across the bay."
)


class TestNormalization:
    def test_ny_abbreviation(self):
        assert normalize_surface("N.Y.") == "New York"

    def test_long_island_abbreviation(self):
        assert normalize_surface("L.I.") == "Long Island"

    def test_no_change_for_normal(self):
        assert normalize_surface("New York") == "New York"

    def test_strips_whitespace(self):
        assert normalize_surface("  Manhattan  ") == "Manhattan"


class TestClassification:
    def test_east_egg_is_fictional(self):
        assert _classify_entity("East Egg", KNOWN_FICTIONAL, 0.7) == "fictional"

    def test_west_egg_is_fictional(self):
        assert _classify_entity("West Egg", KNOWN_FICTIONAL, 0.7) == "fictional"

    def test_valley_of_ashes_is_fictional(self):
        assert _classify_entity("Valley of Ashes", KNOWN_FICTIONAL, 0.7) == "fictional"

    def test_new_york_is_real(self):
        assert _classify_entity("New York", KNOWN_FICTIONAL, 0.7) == "real"

    def test_manhattan_is_real(self):
        assert _classify_entity("Manhattan", KNOWN_FICTIONAL, 0.7) == "real"

    def test_long_island_is_real(self):
        assert _classify_entity("Long Island", KNOWN_FICTIONAL, 0.7) == "real"

    def test_unknown_entity_is_uncertain(self):
        result = _classify_entity("Bleecker Street", KNOWN_FICTIONAL, 0.7)
        # Streets → real (suffix match)
        assert result in ("real", "uncertain")

    def test_custom_fictional_override(self):
        custom = KNOWN_FICTIONAL | {"Fake Town"}
        assert _classify_entity("Fake Town", custom, 0.7) == "fictional"

    def test_uncertain_when_no_indicator(self):
        result = _classify_entity("Gatz Manor", KNOWN_FICTIONAL, 0.7)
        assert result == "uncertain"


class TestSpacyNER:
    """Integration test — requires spaCy model installed."""

    @pytest.fixture(scope="class")
    def nlp(self):
        pytest.importorskip("spacy")
        import spacy
        try:
            return spacy.load("en_core_web_lg")
        except OSError:
            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                pytest.skip("No spaCy model installed")

    def test_extracts_new_york(self, nlp):
        doc = nlp("He took the train to New York.")
        ents = [e.text for e in doc.ents if e.label_ in {"GPE", "LOC", "FAC"}]
        assert any("New York" in e for e in ents)

    def test_extracts_manhattan(self, nlp):
        doc = nlp("The party was held in Manhattan.")
        ents = [e.text for e in doc.ents if e.label_ in {"GPE", "LOC", "FAC"}]
        assert any("Manhattan" in e for e in ents)

    def test_geo_labels_present(self, nlp):
        doc = nlp(SAMPLE_PARA)
        labels = {e.label_ for e in doc.ents}
        assert labels & {"GPE", "LOC", "FAC"}
