#!/usr/bin/env python3
"""
Offline text primitives for VADAR-generated programs.

These functions operate on ordinary strings.  They are intentionally deterministic:
they do not create model clients, make network requests, or delegate to ``TextPatch``.
Query-specific VADAR helpers may compose them with Python's standard library.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping


def _plain_text(value) -> str:
    """Accept a string or a legacy object exposing a `.text` string."""
    return str(getattr(value, "text", value) or "")


def normalize(text) -> str:
    """Case-fold text and collapse punctuation/whitespace for stable matching."""
    value = unicodedata.normalize("NFKD", _plain_text(text)).casefold()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def tokens(text) -> set[str]:
    """Return the normalized word-token set."""
    return set(normalize(text).split())


def contains_phrase(text, phrase) -> bool:
    """True when a normalized phrase occurs on token boundaries."""
    haystack = f" {normalize(text)} "
    needle = normalize(phrase)
    return bool(needle) and f" {needle} " in haystack


def contains_any(text, phrases: Iterable[str]) -> bool:
    """True when at least one normalized phrase occurs in the text."""
    return any(contains_phrase(text, phrase) for phrase in phrases)


def contains_all(text, phrases: Iterable[str]) -> bool:
    """True when every normalized phrase occurs in the text."""
    return all(contains_phrase(text, phrase) for phrase in phrases)


def lexical_score(text, query) -> float:
    """Token-overlap score in [0, 1], with an exact-phrase match scoring 1."""
    if contains_phrase(text, query):
        return 1.0
    left, right = tokens(text), tokens(query)
    if not left or not right:
        return 0.0
    return len(left & right) / len(right)


def best_lexical_match(text, options, aliases: Mapping[str, Iterable[str]] | None = None,
                       default="none"):
    """Return the best locally matched option, or ``default`` when none overlaps."""
    best, best_score = default, 0.0
    for option in options:
        candidates = [str(option)]
        if aliases:
            candidates.extend(str(v) for v in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates), default=0.0)
        if score > best_score:
            best, best_score = option, score
    return best


def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none"):
    """Return a regex capture from text, or ``default`` when it is absent."""
    match = re.search(pattern, _plain_text(text), flags)
    if not match:
        return default
    try:
        value = match.group(group)
    except (IndexError, KeyError):
        return default
    value = str(value).strip()
    return value if value else default


def split_values(text, separators=r"[,;/|]", allowed=None):
    """Split a delimited field and optionally retain only allowed normalized values."""
    values = [part.strip() for part in re.split(separators, _plain_text(text)) if part.strip()]
    if allowed is None:
        return values
    lookup = {normalize(value): value for value in allowed}
    return [lookup[normalize(value)] for value in values if normalize(value) in lookup]


MODULES_SIGNATURES_TEXT = '''
All functions are deterministic, offline, and take ordinary strings. They never call a
model or a network service. Query-specific helpers may also use Python standard-library
string, regex, numeric, and date operations.

def normalize(text) -> str
def tokens(text) -> set[str]
def contains_phrase(text, phrase) -> bool
def contains_any(text, phrases) -> bool
def contains_all(text, phrases) -> bool
def lexical_score(text, query) -> float
def best_lexical_match(text, options, aliases=None, default="none")
def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none")
def split_values(text, separators=r"[,;/|]", allowed=None) -> list[str]
'''
