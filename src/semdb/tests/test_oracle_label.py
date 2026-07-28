"""Oracle labeling: abstention discipline, caching, and cost accounting.

The single most important behaviour here is that an untrustworthy verdict produces
NO label rather than a guessed one. The refinement loop optimizes against these
labels, so a fabricated negative is not a small error — it silently redefines the
objective. Every path that could invent a label is pinned below.

The endpoint is stubbed at the `semvqa` boundary; semvqa's own logprob→score
derivation is tested with the vision stack, not here.
"""
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import oracle_label as OL  # noqa: E402
import semvqa  # noqa: E402


def detail(answer="true", score=0.9, source="logprobs", error=None):
    return {"answer": answer, "score": score, "score_source": source,
            "raw": "", "error": error}


@pytest.fixture
def stub(monkeypatch):
    """Replace the oracle call with a scripted one; records what it was asked."""
    calls = []

    class Stub:
        def __init__(self):
            self.reply = lambda key: detail()

        def __call__(self, *args, **kwargs):
            # per-row caller passes (image_paths|text, question, choices)
            calls.append((args, kwargs))
            return self.reply(args[0] if args else None)

    s = Stub()
    s.calls = calls
    monkeypatch.setattr(semvqa, "imgs_vqa_detail", s)
    monkeypatch.setattr(semvqa, "txt_vqa_detail", s)
    return s


CORPUS = {str(i): {"id": str(i), "text": f"row {i}", "filename": f"{i}.jpg"}
          for i in range(10)}


def caller(stub_ok=True, **over):
    kwargs = dict(question="is it a shoe?", cfg=object(), choices=["true", "false"],
                  boolean=True, image_col=None, image_dir=None, text_cols=["text"])
    kwargs.update(over)
    return OL.make_row_caller(CORPUS, **kwargs)


# --- normalize_bool --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("true", "true"), ("TRUE", "true"), ("True.", "true"), ("yes", "true"), ("1", "true"),
    ("false", "false"), ("No", "false"), ("0", "false"), (" false ", "false"),
])
def test_normalize_bool_accepts_the_usual_spellings(raw, expected):
    assert OL.normalize_bool(raw) == expected


@pytest.mark.parametrize("raw", ["maybe", "", "I think so", "shoe", "none"])
def test_normalize_bool_refuses_to_coerce_junk_to_false(raw):
    """Coercing an unparseable answer to 'false' would read as a confident negative."""
    assert OL.normalize_bool(raw) is None


# --- abstention ------------------------------------------------------------

def test_a_calibrated_answer_becomes_a_label(stub):
    run = OL.label_batch(["1"], caller(), model="m", question="q")
    assert run.labels == {"1": "true"}
    assert run.abstained == []


def test_row_values_are_substituted_into_concatenated_question(stub):
    call = caller(question="Does {text} satisfy the predicate?")
    run = OL.label_batch(["1"], call, model="m", question="template")
    assert run.labels == {"1": "true"}
    assert stub.calls[0][0][1] == "Does row 1 satisfy the predicate?"


def test_bigquery_format_placeholders_receive_row_values(stub):
    call = caller(question="Complaint: %s")
    run = OL.label_batch(["1"], call, model="m", question="template")
    assert run.labels == {"1": "true"}
    assert stub.calls[0][0][1] == "Complaint: row 1"


@pytest.mark.parametrize("source", ["self", "default", "cache-miss"])
def test_an_uncalibrated_score_abstains(stub, source):
    """Without logprobs there is no way to tell a confident answer from a coin flip."""
    stub.reply = lambda k: detail(source=source)
    run = OL.label_batch(["1"], caller(), model="m", question="q")
    assert run.labels == {}
    assert run.abstained == ["1"]
    assert source in run.detail["1"]["error"]


def test_an_errored_call_abstains_and_keeps_the_message(stub):
    stub.reply = lambda k: detail(error="connection refused")
    run = OL.label_batch(["1"], caller(), model="m", question="q")
    assert run.abstained == ["1"]
    assert "connection refused" in run.detail["1"]["error"]


def test_an_off_vocabulary_answer_abstains(stub):
    stub.reply = lambda k: detail(answer="probably a sneaker")
    run = OL.label_batch(["1"], caller(), model="m", question="q")
    assert run.abstained == ["1"]
    assert "unusable answer" in run.detail["1"]["error"]


def test_an_id_absent_from_the_corpus_abstains_rather_than_calling(stub):
    run = OL.label_batch(["999"], caller(), model="m", question="q")
    assert run.abstained == ["999"]
    assert run.detail["999"]["error"] == "id not in corpus"
    assert stub.calls == [], "a missing row must not cost an oracle call"


def test_a_missing_image_file_abstains_rather_than_calling(stub, tmp_path):
    run = OL.label_batch(["1"], caller(image_col="filename", image_dir=str(tmp_path),
                                       text_cols=None),
                         model="m", question="q")
    assert run.abstained == ["1"]
    assert "image not found" in run.detail["1"]["error"]
    assert stub.calls == []


def test_multimodal_pair_passes_structured_text_with_the_image(stub, tmp_path):
    image = tmp_path / "logo.png"
    image.write_bytes(b"stub")
    rows = {"0-logo.png": {
        "pair_id": "0-logo.png", "text1": "British Airways", "file2": str(image),
    }}
    call = OL.make_pair_caller(
        rows, question="Does the logo match the airline?", cfg=object(),
        choices=["true", "false"], boolean=True, image_cols=["file2"],
        image_dir=None, text_cols=["text1"])
    label = call("0-logo.png")
    assert label.value == "true"
    assert "Provided structured fields:\ntext1: British Airways" in stub.calls[0][0][1]


def test_abstentions_do_not_shrink_the_denominator_silently(stub):
    """detail covers every requested key; labels covers only the trustworthy ones."""
    stub.reply = lambda k: detail() if k and "row 1" in str(k) else detail(source="self")
    run = OL.label_batch([str(i) for i in range(5)], caller(), model="m", question="q")
    assert len(run.detail) == 5
    assert len(run.labels) + len(run.abstained) == 5


# --- free-form (AI.GENERATE) ----------------------------------------------

def test_non_boolean_labels_keep_the_raw_answer(stub):
    stub.reply = lambda k: detail(answer="Navy Blue")
    run = OL.label_batch(["1"], caller(boolean=False, choices=None),
                         model="m", question="q")
    assert run.labels == {"1": "Navy Blue"}


def test_a_blank_free_form_answer_abstains(stub):
    stub.reply = lambda k: detail(answer="   ")
    run = OL.label_batch(["1"], caller(boolean=False, choices=None),
                         model="m", question="q")
    assert run.abstained == ["1"]


# --- cache -----------------------------------------------------------------

def test_a_second_run_pays_nothing(stub, tmp_path):
    path = str(tmp_path / "cache.json")
    ids = ["1", "2", "3"]
    first = OL.label_batch(ids, caller(), model="m", question="q",
                           cache=OL.LabelCache(path))
    assert first.calls == 3 and first.cache_hits == 0

    second = OL.label_batch(ids, caller(), model="m", question="q",
                            cache=OL.LabelCache(path))
    assert second.calls == 0 and second.cache_hits == 3
    assert second.labels == first.labels


def test_raising_the_rate_pays_only_for_the_new_rows(stub, tmp_path):
    path = str(tmp_path / "cache.json")
    OL.label_batch(["1", "2"], caller(), model="m", question="q", cache=OL.LabelCache(path))
    grown = OL.label_batch(["1", "2", "3", "4"], caller(), model="m", question="q",
                           cache=OL.LabelCache(path))
    assert grown.cache_hits == 2 and grown.calls == 2


def test_a_changed_question_invalidates_the_labels(stub, tmp_path):
    """A different question is a different label — reusing the old verdict would
    answer the wrong query."""
    path = str(tmp_path / "cache.json")
    OL.label_batch(["1"], caller(), model="m", question="is it a shoe?",
                   cache=OL.LabelCache(path))
    other = OL.label_batch(["1"], caller(), model="m", question="is it a hat?",
                           cache=OL.LabelCache(path))
    assert other.calls == 1 and other.cache_hits == 0


def test_a_changed_model_invalidates_the_labels(stub, tmp_path):
    path = str(tmp_path / "cache.json")
    OL.label_batch(["1"], caller(), model="small", question="q", cache=OL.LabelCache(path))
    other = OL.label_batch(["1"], caller(), model="large", question="q",
                           cache=OL.LabelCache(path))
    assert other.calls == 1


def test_changed_choices_invalidate_the_labels(stub, tmp_path):
    path = str(tmp_path / "cache.json")
    OL.label_batch(["1"], caller(), model="m", question="q", choices=["a", "b"],
                   cache=OL.LabelCache(path))
    other = OL.label_batch(["1"], caller(), model="m", question="q", choices=["a", "c"],
                           cache=OL.LabelCache(path))
    assert other.calls == 1


def test_a_broken_row_abstention_is_cached(tmp_path, monkeypatch):
    """A row whose image does not exist is permanently broken -- re-asking cannot
    change the answer, so it must not be re-queried on every run."""
    path = str(tmp_path / "cache.json")
    call = OL.make_row_caller({"1": {"img": "/nonexistent.jpg", "text": "t"}},
                              question="q", cfg=object(), choices=["true", "false"],
                              boolean=True, image_col="img", image_dir=None,
                              text_cols=None)
    first = OL.label_batch(["1"], call, model="m", question="q", cache=OL.LabelCache(path))
    assert first.abstained == ["1"]
    again = OL.label_batch(["1"], call, model="m", question="q", cache=OL.LabelCache(path))
    assert again.calls == 0 and again.abstained == ["1"], "a missing image stays cached"


def test_a_call_level_abstention_is_NOT_cached(stub, tmp_path):
    """The row is fine; the CALL failed (here: an uncalibrated score). Caching that
    freezes a configuration fault in place and hides it behind "cached" on the retry --
    exactly what happened when a reasoning model's thinking ate the token budget and
    every row came back with no content. The retry must re-ask.
    """
    path = str(tmp_path / "cache.json")
    stub.reply = lambda k: detail(source="self")          # not in TRUSTED_SCORE_SOURCES
    first = OL.label_batch(["1"], caller(), model="m", question="q",
                           cache=OL.LabelCache(path))
    assert first.abstained == ["1"]
    stub.reply = lambda k: detail()                        # config fixed -> logprobs
    again = OL.label_batch(["1"], caller(), model="m", question="q",
                           cache=OL.LabelCache(path))
    assert again.calls == 1, "the fixed configuration must be re-asked, not served stale"
    assert again.labels == {"1": "true"} and again.abstained == []


def test_an_unreadable_cache_is_ignored_not_fatal(stub, tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("{ this is not json")
    run = OL.label_batch(["1"], caller(), model="m", question="q",
                         cache=OL.LabelCache(str(path)))
    assert run.labels == {"1": "true"}


def test_the_cache_file_is_valid_json_after_a_flush(stub, tmp_path):
    path = str(tmp_path / "cache.json")
    OL.label_batch(["1", "2"], caller(), model="m", question="q", cache=OL.LabelCache(path))
    with open(path) as handle:
        assert len(json.load(handle)) == 2


def test_no_cache_path_still_works(stub):
    run = OL.label_batch(["1"], caller(), model="m", question="q", cache=None)
    assert run.labels == {"1": "true"}


# --- concurrency -----------------------------------------------------------

def test_results_stay_matched_to_their_keys_under_concurrency(stub):
    """A thread pool that mismatched a verdict to a key would corrupt every label
    while leaving the counts looking perfect."""
    stub.reply = lambda k: detail(answer="true" if "row 3" in str(k) else "false")
    ids = [str(i) for i in range(10)]
    run = OL.label_batch(ids, caller(), model="m", question="q", concurrency=8)
    assert run.labels["3"] == "true"
    assert all(run.labels[i] == "false" for i in ids if i != "3")


def test_concurrency_is_actually_used(stub):
    seen_threads = set()

    def reply(_):
        seen_threads.add(threading.get_ident())
        return detail()

    stub.reply = reply
    OL.label_batch([str(i) for i in range(8)], caller(), model="m", question="q",
                   concurrency=4)
    assert len(seen_threads) > 1


def test_concurrency_never_exceeds_the_number_of_rows(stub):
    run = OL.label_batch(["1"], caller(), model="m", question="q", concurrency=64)
    assert run.calls == 1


# --- cost ------------------------------------------------------------------

def test_the_summary_reports_calls_cache_hits_and_positives(stub, tmp_path):
    path = str(tmp_path / "c.json")
    stub.reply = lambda k: detail(answer="true" if "row 0" in str(k) else "false")
    OL.label_batch(["0", "1"], caller(), model="m", question="q", cache=OL.LabelCache(path))
    run = OL.label_batch(["0", "1", "2"], caller(), model="m", question="q",
                         cache=OL.LabelCache(path))
    text = run.summary()
    assert "1 calls" in text and "2 cached" in text
    assert "1 positive" in text


def test_mean_latency_ignores_cached_rows(stub, tmp_path):
    """Cached rows cost 0ms; averaging them in would understate the real oracle cost."""
    path = str(tmp_path / "c.json")
    OL.label_batch(["1"], caller(), model="m", question="q", cache=OL.LabelCache(path))
    second = OL.label_batch(["1"], caller(), model="m", question="q",
                            cache=OL.LabelCache(path))
    assert second.mean_latency_ms == 0
    assert second.detail["1"]["cached"] is True
