"""
semruntime.py — runtime helpers that compiled SemDB queries import for the
RESIDUAL path (the rows the cheap relational plan was unsure about).

RESIDUAL FALLBACK IS CURRENTLY DISABLED. `vlm_judge` / `llm_judge` are no-ops that
return False without calling any model — so a compiled query's result comes PURELY
from the extracted attributes + relational code. This is deliberate: we want to
measure the generated code's own ability, not a live-model crutch. Rows the plan
was unsure about are simply left unmatched and counted in `METER.skipped`.

To re-enable the live residual judge later, restore the endpoint-backed body (see
git history) — the call sites in generated code do not need to change.

Example (inside a generated compiled_<q>.py):
    from semruntime import vlm_judge, METER
    if vlm_judge(f"Determine if the image shows the logo of {airline}.",
                 endpoint=args.endpoint, model=args.model, api_key=args.api_key,
                 image_path=image_file):
        pairs.append((airline, uri))     # never taken while disabled
    ...
    residual_calls = METER.judge_calls   # stays 0; METER.skipped counts the unsure rows
"""


class _Meter:
    def __init__(self):
        self.judge_calls = 0     # real live-model residual calls (0 while disabled)
        self.skipped = 0         # rows that WOULD have gone to the residual judge


METER = _Meter()


def vlm_judge(prompt, endpoint=None, model=None, api_key="EMPTY", image_path=None, timeout=120):
    """DISABLED residual judge: never calls a model, always returns False, requires
    no endpoint. Counts the row in METER.skipped so the compiled query can report how
    many unsure rows were left unresolved. Signature kept stable so generated code is
    unchanged."""
    METER.skipped += 1
    return False


def llm_judge(prompt, endpoint=None, model=None, api_key="EMPTY", timeout=120):
    """Text-only alias — also disabled."""
    METER.skipped += 1
    return False
