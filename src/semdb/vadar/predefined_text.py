#!/usr/bin/env python3
"""
vadar/predefined_text.py — the PREDEFINED TEXT API (the text analog of predefined.py).
Free functions taking a `TextPatch` `text` (semtext-backed), mirroring the vision API's
`classify(image, ...)` call style. The generated solve_<q>.py and helper functions
compose ONLY these. `MODULES_SIGNATURES_TEXT` is the docstring block shown to the
Signature/API/Program agents.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/semdb


def judge(text, question):
    """True/False LLM judgement over the text (AI.IF semantics)."""
    return text.judge(question)


def classify(text, options):
    """Pick the single best VALUE from a CLOSED value space (enum / DB column values)."""
    return text.classify(options)


def extract(text, field):
    """Extract one attribute value named `field` from the text (AI.GENERATE field)."""
    return text.extract(field)


def generate(text, instruction):
    """Free-form generation over the text following `instruction`."""
    return text.generate(instruction)


def score(text, query):
    """Relevance of the text to `query` in [0, 1] (soft filter / ranking)."""
    return text.score(query)


MODULES_SIGNATURES_TEXT = '''
"""
Answers a yes/no question about the text and returns a bool. Use for AI.IF predicates.
Args:
    text (TextPatch): the row's text.
    question (string): a yes/no question.
Returns:
    bool: True iff the answer is yes.
"""
def judge(text, question) -> bool

"""
Classifies the text into the single best option from a CLOSED value space and returns
that VALUE (a real field). Use for enum categories or a DB column's values. Read the
value space from the structured CSV column AT RUNTIME — never hardcode it.
Args:
    text (TextPatch): the row's text.
    options (list): candidate string values.
Returns:
    string: the best-matching option value.
"""
def classify(text, options) -> str

"""
Extracts the value of a named attribute from the text (AI.GENERATE of one field).
Returns 'none' when absent.
Args:
    text (TextPatch): the row's text.
    field (string): the attribute name to extract.
Returns:
    string: the extracted value (or 'none').
"""
def extract(text, field) -> str

"""
Free-form generation over the text following an instruction (AI.GENERATE text).
Args:
    text (TextPatch): the row's text.
    instruction (string): what to produce.
Returns:
    string: the generated text.
"""
def generate(text, instruction) -> str

"""
Relevance of the text to a short query in [0,1]. Use for ranking / soft filters.
Args:
    text (TextPatch): the row's text.
    query (string): a short phrase to match.
Returns:
    float: similarity in [0,1].
"""
def score(text, query) -> float
'''
