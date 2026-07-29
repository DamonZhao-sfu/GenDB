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


def classify_detail(text, options,
                    descriptions: Mapping[str, str] | None = None,
                    aliases: Mapping[str, Iterable[str]] | None = None,
                    default="none") -> tuple[str, float]:
    """Bounded offline text classification over an explicit database/SQL value space.

    Scores label names, caller-supplied descriptions, and aliases by token overlap.
    This is deliberately a bounded approximation—not a claim to reproduce a remote
    model's behavior—and returns its confidence so a plan can expose that boundary.
    """
    best, best_score = default, 0.0
    for option in options:
        candidates = [str(option)]
        if descriptions and option in descriptions:
            candidates.append(str(descriptions[option]))
        if aliases:
            candidates.extend(str(value) for value in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates),
                    default=0.0)
        if score > best_score:
            best, best_score = option, score
    return best, float(best_score)


def extract_from_candidates(text, candidates, default="none") -> tuple[str, float]:
    """Extract the longest candidate occurring in text, else best lexical match.

    Candidate values must come from an ordinary database column or SQL literal value
    space, never validation labels. This supports bounded entity extraction such as a
    brand name without network access.
    """
    values = [str(value).strip() for value in candidates if str(value).strip()]
    contained = [value for value in values if contains_phrase(text, value)]
    if contained:
        value = max(contained, key=lambda item: (len(normalize(item)), item))
        return value, 1.0
    value = best_lexical_match(text, values, default=default)
    return value, lexical_score(text, value) if value != default else 0.0


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
def classify_detail(text, options, descriptions=None, aliases=None, default="none") -> tuple[str,float]
def extract_from_candidates(text, candidates, default="none") -> tuple[str,float]
def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none")
def split_values(text, separators=r"[,;/|]", allowed=None) -> list[str]
'''
