import pytest

# Import functions under test from main.py (module is at project root)
from main import (
    _normalize,
    _extract_location_cues,
    _suggest_from_question,
)


def test_normalize_basic_and_accents():
    # Lowercasing and whitespace collapse
    assert _normalize("  Cabo   Pulmo  ") == "cabo pulmo"
    # Accent stripping
    assert _normalize("Pulmó") == "pulmo"
    # Punctuation simplification
    assert _normalize("Islote/Islotes – La Paz!") == "islote islotes la paz"
    # Keep hyphens/apostrophes but normalize
    assert _normalize("Isla Espíritu-Santo's Reef") == "isla espiritu-santo's reef"


def test_extract_location_cues():
    text = "Holacanthus passer in the Los Islotes reef in La Paz, near Espíritu Santo."
    cues = _extract_location_cues(text)
    # Expect locality names normalized and present
    assert "la paz" in cues
    # 2-gram capture should include 'los islotes'
    assert any(cue == "los islotes" for cue in cues)


def test_suggest_from_question_reefs():
    candidates = [
        "Los Islotes",
        "La Reina",
        "Cabo Pulmo",
        "El Bajo",
        "San Rafaelito",
    ]
    # Intentionally misspell/partial to test fuzzy matching and locality filter
    query = "holacanthus passer in the Los Islote reef in La Paz"
    sugg = _suggest_from_question(query, candidates, topn=5)

    # Should suggest 'Los Islotes' within the top suggestions
    assert "Los Islotes" in sugg
    # It should be ranked near the top (be lenient regardless of backend scorer)
    assert sugg.index("Los Islotes") <= 2
