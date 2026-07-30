"""The TEXT half of the operator library (merged into `vadar/predefined.py`).

Every one of these takes its value space as an ARGUMENT — the library carries no
built-in taxonomy, so the same function serves a genre predicate, a region predicate
and a brand predicate.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vadar import predefined as pt


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
    label, confidence = pt.text_classify_detail(
        "soft fabric clothing worn on the feet",
        ["Topwear", "Socks"],
        descriptions={"Socks": "clothing worn on the feet made of soft fabric"})
    assert label == "Socks"
    assert confidence > 0
    assert pt.extract_from_candidates(
        "The new Acme Sports running shoe", ["Acme", "Acme Sports"])[0] == "Acme Sports"


def test_multi_label_classification_keeps_each_supported_option():
    labels, scores = pt.text_classify_multi_detail(
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


def test_phrase_cues_compose_into_a_multi_label_predicate_without_a_builtin_taxonomy():
    """曾经的 `classify_movie_genres` 现在是「调用方传 value space + cue 短语」。

    注意它用的是 `contains_any`（整短语命中）而不是
    `text_classify_multi_detail`（token 重叠打分）—— 见下面那条测试。
    """
    cues = {
        "comedy": ("comedy", "comic", "funny", "humorous"),
        "romance": ("romantic", "romance", "love story"),
        "horror": ("horror", "monster", "haunted"),
        "science fiction": ("science fiction", "sci fi", "alien"),
    }

    def genres(text):
        return [g for g, c in cues.items() if pt.contains_any(text, c)]

    assert genres("A romantic comedy film and love story.") == ["comedy", "romance"]
    assert genres("A science fiction horror story about an alien monster.") == [
        "horror", "science fiction"]


def test_token_overlap_scoring_over_fires_on_multi_word_cues():
    """`text_classify_multi_detail` 按 token 重叠打分，多词 cue 会被部分命中。

    "love story" 在 "...horror story..." 里命中 1/2 个 token = 0.5，刚好过阈值。
    需要整短语语义时用 `contains_any`，这条边界必须是显式的、可测的。"""
    aliases = {"romance": ["love story"], "horror": ["horror"]}
    labels, scores = pt.text_classify_multi_detail(
        "A science fiction horror story about an alien monster.",
        ["romance", "horror"], aliases=aliases, threshold=0.5)
    assert scores["romance"] == 0.5 and "romance" in labels     # the partial hit
    assert not pt.contains_any(
        "A science fiction horror story about an alien monster.", ["love story"])


def test_any_value_in_set_handles_delimiters_suffixes_and_fails_closed():
    """曾经的 `destination_in_region` 现在是「调用方传成员集合」的通用集合测试。"""
    europe = ["London", "Frankfurt", "Paris", "Amsterdam"]
    germany = ["Germany", "Frankfurt", "Berlin", "Munich"]
    assert pt.any_value_in_set("Atlanta, London–Heathrow", europe)
    assert pt.any_value_in_set("Frankfurt", germany)
    assert not pt.any_value_in_set("Jeddah, Atlanta", europe)
    assert not pt.any_value_in_set("Frankfurt", [])          # empty value space: closed
    assert pt.any_value_in_set("London (LHR), Doha", europe)  # airport-code suffix stripped


def test_person_name_candidates_are_deduplicated_and_preserve_spelling():
    assert pt.extract_person_names(
        "Lizzy Caplan stars with Jesse Bradford. Later Lizzy Caplan returns.") == [
            "Lizzy Caplan", "Jesse Bradford"]


def test_api_surface_has_no_endpoint_or_semantic_judge():
    public = set(pt.MODULES_SIGNATURES.casefold().split())
    assert "judge" not in public
    assert "endpoint" not in public
    assert not hasattr(pt, "judge")
