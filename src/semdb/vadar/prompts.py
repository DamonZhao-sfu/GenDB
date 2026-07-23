#!/usr/bin/env python3
"""vadar/prompts.py — the 3 agent prompts, adapted from VADAR (signature/api/program) for
our predefined API and SemBench IMAGE extraction. The "question" is a query's visual
predicate + the fields to extract; the final Program is `def extract(image) -> {fields}`."""

# ---- Signature agent: propose NEW helper signatures over the predefined API ----
SIGNATURE_PROMPT = """Propose ONLY new helper method signatures to add to the API for
extracting the fields these SemBench image queries need. Build minorly on the existing
API; do NOT re-add existing methods, and do NOT add a method if a combination of existing
ones suffices.

Existing API:
{signatures}

Queries (each references an image; solvable by composing the API):
{question}

For each proposed method output the docstring in <docstring></docstring> immediately
followed by the signature in <signature></signature>. Methods take `image` (the image)
first, return a real field VALUE / bool / list. Keep them general (do not assume specific
objects). Examples of GOOD helpers: `logo_name(image, names)` (classify over the value
space), `is_sports_shoe(image)`, `is_racetrack_logo(image)` (a gate)."""

# ---- API agent: implement one proposed signature by composing the API ----
API_PROMPT = """You implement a method body from its docstring and signature, composing
ONLY the API below (call the functions directly; `image` is already an ImagePatch).

API:
{predef_signatures}

{generated_signatures}

Examples:
<docstring>
\"\"\"Returns the airline whose logo the image shows, from the given names.\"\"\"
</docstring>
<signature>def logo_name(image, names):</signature>
<implementation>
return best_ocr_match(image, names)
</implementation>

<docstring>
\"\"\"True if the image is a sports shoe featuring both yellow and silver.\"\"\"
</docstring>
<signature>def is_yellow_silver_sports_shoe(image):</signature>
<implementation>
if classify(image, ["sports_shoes","sandal","boot","other_footwear","not_footwear"]) != "sports_shoes":
    return False
cols = set(dominant_colors(image, 0.03))
return "yellow" in cols and "silver" in cols
</implementation>

Now implement this one — output ONLY the implementation body inside <implementation></implementation>:
<docstring>
{docstring}
</docstring>
<signature>{signature}</signature>"""

# ---- Program agent: write extract(image) using the full API ----
PROGRAM_PROMPT = """You write a Python function `extract(image)` that returns a dict with
one entry per requested field, composing the API to answer each field for ONE image. Use
the lightest composition; the returned label IS the field value (joins/filters downstream).

API (predefined + generated helpers):
{api}

Value spaces (closed sets, from the DB) available as module constants:
{value_spaces}

Fields to extract and the query predicate:
{question}

Rules: SMALL/visual value space or enum -> classify; LARGE value space of legible wordmark
names -> best_ocr_match; colors -> dominant_colors; presence/count -> detect. Gate when a
lone target hides among many images (classify/verify a "kind" first). Output ONLY:
<program>
def extract(image):
    return { ... }
</program>"""
