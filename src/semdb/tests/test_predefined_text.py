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


def test_bounded_classification_and_candidate_extraction_expose_confidence():
    label, confidence = pt.classify_detail(
        "soft fabric clothing worn on the feet",
        ["Topwear", "Socks"],
        descriptions={"Socks": "clothing worn on the feet made of soft fabric"})
    assert label == "Socks"
    assert confidence > 0
    assert pt.extract_from_candidates(
        "The new Acme Sports running shoe", ["Acme", "Acme Sports"])[0] == "Acme Sports"


def test_multi_label_classification_keeps_each_supported_option():
    labels, scores = pt.classify_multi_detail(
        "A funny romantic story with jokes and a love affair.",
        ["comedy", "romance", "horror"],
        aliases={
            "comedy": ["funny jokes"],
            "romance": ["romantic love affair"],
            "horror": ["ghost slasher"],
        },
        threshold=0.5,
    )
    assert labels == ["comedy", "romance"]
    assert scores["horror"] == 0


def test_destination_region_handles_airport_suffixes_and_fails_closed():
    assert pt.destination_in_region("Atlanta, London–Heathrow", "Europe")
    assert pt.destination_in_region("Frankfurt", "Germany")
    assert not pt.destination_in_region("Jeddah, Atlanta", "Europe")
    assert not pt.destination_in_region("Frankfurt", "Oceania")


def test_movie_genres_are_multi_label_and_use_general_taxonomy():
    assert pt.classify_movie_genres(
        "A romantic comedy film and love story.") == ["comedy", "romance"]
    assert pt.classify_movie_genres(
        "A science fiction horror story about an alien monster.") == [
            "horror", "science fiction"]
    assert pt.has_movie_genres(
        "A romantic comedy and love story.", ["romance", "comedy"])
    assert not pt.has_movie_genres(
        "A romantic comedy and love story.", ["romance", "horror"])


def test_person_name_candidates_are_deduplicated_and_preserve_spelling():
    assert pt.extract_person_names(
        "Lizzy Caplan stars with Jesse Bradford. Later Lizzy Caplan returns.") == [
            "Lizzy Caplan", "Jesse Bradford"]


def test_api_surface_has_no_endpoint_or_semantic_judge():
    public = set(pt.MODULES_SIGNATURES_TEXT.casefold().split())
    assert "judge" not in public
    assert "endpoint" not in public
    assert not hasattr(pt, "judge")
