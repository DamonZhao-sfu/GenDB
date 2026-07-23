import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vadar import predefined_text as PT


class _FakePatch:
    def judge(self, q): return True
    def classify(self, opts): return opts[0]
    def extract(self, field): return "x"
    def generate(self, instr): return "y"
    def score(self, q): return 0.5


def test_free_functions_delegate_to_patch():
    p = _FakePatch()
    assert PT.judge(p, "q?") is True
    assert PT.classify(p, ["a", "b"]) == "a"
    assert PT.extract(p, "genre") == "x"
    assert PT.generate(p, "summarize") == "y"
    assert PT.score(p, "romance") == 0.5


def test_signatures_doc_present():
    assert "judge(" in PT.MODULES_SIGNATURES_TEXT
    assert "classify(" in PT.MODULES_SIGNATURES_TEXT
