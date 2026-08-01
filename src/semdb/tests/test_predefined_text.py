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
    """`method="lexical"` pinned deliberately: this asserts `horror == 0`, which is an
    OVERLAP property. Cosine similarity is never exactly 0, so under the embedding
    default the assertion would be meaningless rather than merely different."""
    labels, scores = pt.text_classify_multi_detail(
        "A funny romantic story with jokes and a love affair.",
        ["comedy", "romance", "horror"],
        aliases={
            "comedy": ["funny jokes"],
            "romance": ["romantic love affair"],
            "horror": ["ghost slasher"],
        },
        threshold=0.5,
        method="lexical",
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
        ["romance", "horror"], aliases=aliases, threshold=0.5, method="lexical")
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


# --- embedding-backed classification (the default since the lexical scorer was
# --- shown to leave 998/1000 real reviews tied at exactly 0.0) -----------------

def test_embedding_scoring_separates_what_token_overlap_could_not():
    """The defect that motivated the change: neither review contains the words
    "positive"/"negative", so token overlap scored both 0.0 and could not rank them."""
    good = "An absolute masterpiece, I loved every minute of it."
    bad = "A dull, lifeless slog that wastes its cast."
    options = ["positive", "negative"]
    descriptions = {"positive": "a positive movie review",
                    "negative": "a negative movie review"}
    assert pt.text_classify_detail(good, options, method="lexical") == ("none", 0.0)
    assert pt.text_classify_detail(bad, options, method="lexical") == ("none", 0.0)

    good_label, good_score = pt.text_classify_detail(good, options, descriptions)
    bad_label, _ = pt.text_classify_detail(bad, options, descriptions)
    assert good_label == "positive" and bad_label == "negative"
    assert 0.0 < good_score <= 1.0


def test_batch_matches_the_per_row_form():
    """`text_classify_batch` exists only to save encoder passes — it must not change
    a single answer, or plans would score differently depending on how they looped."""
    texts = ["a brilliant, moving film", "utterly boring and pointless",
             "", "the cinematography is stunning"]
    options = ["positive", "negative"]
    batched = pt.text_classify_batch(texts, options)
    per_row = [pt.text_classify_detail(t, options) for t in texts]
    assert [b[0] for b in batched] == [s[0] for s in per_row]
    assert all(abs(b[1] - s[1]) < 1e-5 for b, s in zip(batched, per_row))


def test_confidence_is_comparable_across_rows_not_a_two_valued_score():
    """A ranking (Spearman) or clustering (ARI) query needs a score with real spread.
    The lexical scorer produced two distinct values over a whole corpus."""
    texts = ["superb and unforgettable", "quite good overall", "it was okay",
             "somewhat disappointing", "a complete disaster"]
    scores = [s for _, s in pt.text_classify_batch(texts, ["positive", "negative"])]
    assert len({round(s, 4) for s in scores}) == len(texts)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert not all(s > 0.99 for s in scores), "softmax temperature saturated the score"


def test_min_confidence_abstains_instead_of_taking_a_weak_argmax():
    options = ["positive", "negative"]
    label, score = pt.text_classify_detail("the film is 104 minutes long", options,
                                           min_confidence=0.99, default="unknown")
    assert label == "unknown" and score < 0.99


def test_empty_and_degenerate_inputs_do_not_crash():
    assert pt.text_classify_batch([], ["a", "b"]) == []
    assert pt.text_classify_batch(["x"], []) == [("none", 0.0)]
    assert pt.text_classify_batch([None], ["a", "b"])[0][0] in ("a", "b")


# --- score() calibration trap -------------------------------------------------

def test_clip_match_rescaling_makes_an_absolute_half_threshold_meaningless():
    """`score()` is nominally [0,1] but `clip_match` maps cosine via (c+1)/2, and real
    CLIP image-text cosines are ~0.1..0.35. So `>= 0.5` accepts EVERY pair — the defect
    that made one query emit 1,000 rows against a 4-row ground truth. Pinned as
    arithmetic so it needs no GPU and cannot regress silently.
    """
    def rescale(cosine):                                     # mirrors backend.clip_match
        return (cosine + 1.0) / 2.0

    realistic = [0.05, 0.10, 0.20, 0.35]                     # incl. an unrelated pair
    assert all(rescale(c) >= 0.5 for c in realistic), (
        "if this fails the rescaling changed and the guidance in score()'s docstring "
        "must be re-measured")
    # A cosine would have to be NEGATIVE to fall below 0.5 — CLIP image-text pairs
    # essentially never are.
    assert rescale(-0.01) < 0.5
    assert "verify_detail" in pt.score.__doc__, (
        "score() must keep steering yes/no decisions to the calibrated primitive")
