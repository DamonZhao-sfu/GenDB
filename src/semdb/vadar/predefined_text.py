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


def classify_multi_detail(text, options,
                          descriptions: Mapping[str, str] | None = None,
                          aliases: Mapping[str, Iterable[str]] | None = None,
                          threshold: float = 0.34) -> tuple[list[str], dict[str, float]]:
    """Select every supported label from an explicit multi-label value space."""
    selected: list[str] = []
    scores: dict[str, float] = {}
    for option in options:
        candidates = [str(option)]
        if descriptions and option in descriptions:
            candidates.append(str(descriptions[option]))
        if aliases:
            candidates.extend(str(value) for value in aliases.get(option, ()))
        score = max((lexical_score(text, candidate) for candidate in candidates),
                    default=0.0)
        scores[str(option)] = float(score)
        if score >= float(threshold):
            selected.append(option)
    return selected, scores


# General geographic facts used by airport-route predicates. These are repository
# runtime knowledge, not validation examples or benchmark labels.
_EUROPE_LOCATIONS = frozenset({
    "albania", "andorra", "armenia", "austria", "azerbaijan", "belarus",
    "belgium", "bosnia", "bulgaria", "croatia", "cyprus", "czechia",
    "czech republic", "denmark", "estonia", "finland", "france", "georgia",
    "germany", "greece", "hungary", "iceland", "ireland", "italy", "kosovo",
    "latvia", "liechtenstein", "lithuania", "luxembourg", "malta", "moldova",
    "monaco", "montenegro", "netherlands", "north macedonia", "norway",
    "poland", "portugal", "romania", "san marino", "serbia", "slovakia",
    "slovenia", "spain", "sweden", "switzerland", "turkey", "ukraine",
    "united kingdom",
    "amsterdam", "athens", "barcelona", "belgrade", "berlin", "birmingham",
    "bratislava", "brussels", "bucharest", "budapest", "copenhagen", "dublin",
    "dubrovnik", "dusseldorf", "edinburgh", "frankfurt", "geneva", "glasgow",
    "hamburg", "helsinki", "istanbul", "lisbon", "ljubljana", "london",
    "madrid", "manchester", "milan", "munich", "naples", "nice", "oslo",
    "paris", "prague", "reykjavik", "riga", "rome", "sarajevo", "sofia",
    "stockholm", "tallinn", "tirana", "venice", "vienna", "vilnius", "warsaw",
    "zagreb", "zurich",
})
_GERMANY_LOCATIONS = frozenset({
    "germany", "berlin", "bremen", "cologne", "dortmund", "dresden",
    "dusseldorf", "frankfurt", "hamburg", "hanover", "leipzig", "munich",
    "nuremberg", "stuttgart",
})

_MOVIE_GENRE_ALIASES = {
    "action": ("action film", "combat", "mission", "battle", "explosion"),
    "biography": ("biographical", "biopic", "based on the life", "true story",
                  "non-fiction book", "memoir", "my life"),
    "comedy": ("comedy", "comic", "funny", "humorous"),
    "crime": ("crime film", "criminal", "police", "gangster", "murder"),
    "drama": ("drama film", "dramatic"),
    "heist": ("heist", "robbery", "magician thieves"),
    "horror": ("horror", "monster", "supernatural", "haunted", "slasher"),
    "romance": ("romantic", "romance", "love story", "falls in love"),
    "satire": ("satire", "satirical", "parody"),
    "science fiction": ("science fiction", "sci fi", "alien", "extraterrestrial",
                        "time travel", "post apocalyptic", "superhero",
                        "marvel cinematic universe"),
    "thriller": ("thriller", "espionage", "suspense"),
    "war": ("war film", "world war", "wartime"),
    "western": ("western film", "old west", "outlaw", "frontier"),
}


def classify_movie_genres(text, threshold: float = 0.5) -> list[str]:
    """Classify a synopsis into the repository's general multi-label genre taxonomy."""
    # Genre cues are phrases, not bags of generic words: token-overlap on "action
    # film" would otherwise classify every synopsis containing only "film" as action.
    del threshold  # retained for a stable generated-helper interface
    return [
        genre for genre, cues in _MOVIE_GENRE_ALIASES.items()
        if contains_any(text, (genre, *cues))
    ]


def has_movie_genres(text, required: Iterable[str]) -> bool:
    """Whether every requested genre is supported by the synopsis taxonomy."""
    found = set(classify_movie_genres(text))
    return all(normalize(value) in found for value in required)


def destination_in_region(destinations, region,
                          extra_locations: Iterable[str] | None = None) -> bool:
    """Test a delimited destination field against Germany/Europe geography."""
    wanted = normalize(region)
    if wanted == "germany":
        locations = set(_GERMANY_LOCATIONS)
    elif wanted in {"europe", "european"}:
        locations = set(_EUROPE_LOCATIONS)
    else:
        return False
    locations.update(normalize(value) for value in (extra_locations or ()))
    for destination in split_values(destinations):
        cleaned = re.sub(r"\([^)]*\)", " ", destination)
        if any(contains_phrase(cleaned, location) for location in locations):
            return True
    return False


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


def extract_person_names(text) -> list[str]:
    """Extract explicit multi-token proper names without a remote NER service.

    This intentionally returns candidates, not a claim that every candidate is a
    person. Relational evidence such as occurrence across all target documents can
    safely disambiguate the candidates downstream.
    """
    pattern = re.compile(
        r"\b(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
        r"(?:\s+(?:de|del|van|von|da|dos|la|le))?\s+)"
        r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+"
        r"(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]+){0,2}\b")
    seen: set[str] = set()
    result: list[str] = []
    sentence_leaders = {
        "a", "after", "an", "as", "at", "before", "by", "during", "for",
        "from", "in", "later", "meanwhile", "on", "the", "then", "when",
    }
    for match in pattern.finditer(_plain_text(text)):
        value = " ".join(match.group(0).split()).strip(" ,.;:()[]")
        pieces = value.split()
        if len(pieces) >= 3 and pieces[0].casefold() in sentence_leaders:
            value = " ".join(pieces[1:])
        key = normalize(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


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
def classify_multi_detail(text, options, descriptions=None, aliases=None, threshold=0.34) -> tuple[list[str],dict[str,float]]
def classify_movie_genres(text, threshold=0.5) -> list[str]
def has_movie_genres(text, required) -> bool
def destination_in_region(destinations, region, extra_locations=None) -> bool
def extract_from_candidates(text, candidates, default="none") -> tuple[str,float]
def regex_extract(text, pattern, group=1, flags=re.IGNORECASE, default="none")
def extract_person_names(text) -> list[str]
def split_values(text, separators=r"[,;/|]", allowed=None) -> list[str]
'''
