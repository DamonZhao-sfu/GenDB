"""
semruntime.py — runtime helpers that compiled SemDB queries import for the
RESIDUAL path (the rows the cheap relational plan was unsure about).

The residual is a TYPED OpImgVQA call (`semvqa`): the answer is constrained to a value
space by guided decoding, and the score is derived from token logprobs, so it is
comparable across rows and a `theta` cut on it is meaningful.

WHEN IT RUNS. Only when the compiled query is given an `--endpoint`. Without one, every
judge is a no-op returning False, exactly as before — a run measures the generated
code's own ability, and the unsure rows are counted in `METER.skipped`. Set
`SEMDB_RESIDUAL=0` to force that measurement mode even when an endpoint is present.

Example (inside a generated compiled_<q>.py):
    from semruntime import vlm_judge, vlm_answer, METER
    if vlm_judge(f"Does the image show the logo of {airline}?",
                 endpoint=args.endpoint, model=args.model, api_key=args.api_key,
                 image_path=image_file):
        pairs.append((airline, uri))
    # or, to recover a real FIELD value instead of a boolean:
    brand, score = vlm_answer("Which airline's logo is this?", AIRLINES,
                              endpoint=args.endpoint, model=args.model,
                              api_key=args.api_key, image_path=image_file)
    ...
    residual_calls = METER.judge_calls   # METER.skipped counts unresolved rows
"""

import os


class _Meter:
    def __init__(self):
        self.judge_calls = 0     # live-model residual calls actually made
        self.skipped = 0         # unsure rows left unresolved (no endpoint / disabled)
        self.scores = []         # the residual's calibrated scores, for threshold tuning

    def summary(self):
        n = len(self.scores)
        return {"judge_calls": self.judge_calls, "skipped": self.skipped,
                "mean_score": (sum(self.scores) / n) if n else None}


METER = _Meter()


def residual_enabled():
    """The residual needs an endpoint; SEMDB_RESIDUAL=0 disables it even with one."""
    return os.environ.get("SEMDB_RESIDUAL", "1").strip().lower() not in ("0", "false", "no")


def vlm_answer(question, choices=None, endpoint=None, model=None, api_key="EMPTY",
               image_path=None, text=None, timeout=120):
    """The typed residual: returns (answer, score). `choices` constrains the answer to a
    value space so the result is a real field value that joins. Returns ("none", 0.0)
    and counts a skip when the residual is unavailable."""
    if not endpoint or not residual_enabled():
        METER.skipped += 1
        return "none", 0.0
    import semvqa
    cfg = semvqa.build_cfg(model, endpoint, api_key, timeout=timeout)
    d = (semvqa.img_vqa_detail(image_path, question, choices, cfg=cfg)
         if image_path is not None else
         semvqa.txt_vqa_detail(text or "", question, choices, cfg=cfg))
    METER.judge_calls += 1
    if d["error"] is not None:
        METER.skipped += 1
        return "none", 0.0
    METER.scores.append(d["score"])
    return d["answer"], d["score"]


def vlm_judge(prompt, endpoint=None, model=None, api_key="EMPTY", image_path=None,
              timeout=120, theta=None):
    """Boolean residual: re-asks the ORIGINAL semantic predicate. True only when the
    model answers yes AND (when `theta` is given) its score clears the threshold.
    Signature is unchanged from the disabled version, so generated call sites work."""
    answer, score = vlm_answer(prompt, ["yes", "no"], endpoint=endpoint, model=model,
                               api_key=api_key, image_path=image_path, timeout=timeout)
    return answer == "yes" and (theta is None or score >= theta)


def llm_judge(prompt, endpoint=None, model=None, api_key="EMPTY", timeout=120,
              text=None, theta=None):
    """Text-only twin of `vlm_judge`."""
    answer, score = vlm_answer(prompt, ["yes", "no"], endpoint=endpoint, model=model,
                               api_key=api_key, text=text, timeout=timeout)
    return answer == "yes" and (theta is None or score >= theta)
