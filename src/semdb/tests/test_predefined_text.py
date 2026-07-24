import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vadar import predefined_text as pt


def test_normalized_phrase_matching_is_local_and_deterministic():
    text = "A SCI-FI / comedy film, directed by Chloé Zhao."
    assert pt.normalize(text) == "a sci fi comedy film directed by chloe zhao"
    assert pt.contains_phrase(text, "comedy film")
    assert pt.contains_any(text, ["horror", "sci fi"])
    assert pt.contains_all(text, ["film", "chloe zhao"])
    assert not pt.contains_phrase(text, "comed")


def test_lexical_matching_regex_and_lists():
    assert pt.lexical_score("direct flights to Frankfurt", "flights Frankfurt") == 1.0
    assert pt.best_lexical_match("A funny romantic comedy", ["horror", "comedy"]) == "comedy"
    assert pt.best_lexical_match("unknown", ["horror", "comedy"]) == "none"
    assert pt.regex_extract("Director: Jane Doe; 2024", r"Director:\s*([^;]+)") == "Jane Doe"
    assert pt.split_values("Drama, sci-fi | unknown", allowed=["Drama", "Sci-Fi"]) == [
        "Drama", "Sci-Fi"
    ]


def test_api_surface_has_no_endpoint_or_semantic_judge():
    public = set(pt.MODULES_SIGNATURES_TEXT.casefold().split())
    assert "judge" not in public
    assert "endpoint" not in public
    assert not hasattr(pt, "judge")
