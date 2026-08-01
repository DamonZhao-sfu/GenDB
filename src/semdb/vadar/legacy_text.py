#!/usr/bin/env python3
"""Compatibility shims for generated programs written BEFORE the vadar reorg.

`vadar/predefined_text.py` was merged into `vadar/predefined.py`. Three
scenario-specific helpers were dropped in that merge — the general text API keeps
no benchmark geography or genre tables, and that is the right call for the surface
the agents are shown. But solvers generated before the merge still call them, and
those artifacts are measurement records: rewriting their semantics to re-run them
would silently change what the recorded numbers mean.

So the tables and the three functions live on HERE, verbatim from
predefined_text.py@4cccf43^, reachable only by an explicit
`from vadar.legacy_text import ...`. Deliberately NOT re-exported from
`vadar/__init__.py` and NOT listed in API.md or MODULES_SIGNATURES, so no newly
generated program can reach for them.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from .predefined import contains_any, contains_phrase, normalize, split_values


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
