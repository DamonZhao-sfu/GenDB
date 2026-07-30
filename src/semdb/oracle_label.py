"""Label sampled rows with an oracle model over an OpenAI-compatible endpoint.

This is the "+ LLM labeling" half of the validation-set design: `sampling.py` decides
WHICH rows to spend labels on, this module spends them. The oracle is the expensive,
trusted model; the program being validated is the cheap proxy that must reproduce it.
That separation is real here rather than nominal, because generated VADAR solvers are
offline by construction (`validateOfflineVadarFile` in orchestrator.mjs rejects any
endpoint or API-key use), so a solver physically cannot call the model that graded it.

Three properties matter more than throughput:

  * **Abstention over guessing.** `semvqa` returns a CALIBRATED score derived from the
    answer span's token logprobs, plus `score_source` recording where the score came
    from. When the score did not come from logprobs, or the call errored, or the model
    answered off-vocabulary, this module records an ABSTAIN and emits no label. A
    fabricated label is worse than a missing one: it silently moves the accuracy the
    refinement loop optimizes against.
  * **The question comes from the SQL.** `predicate.py` extracts the exact `AI.IF` /
    `AI.GENERATE` / `AI.CLASSIFY` prompt, so the oracle is asked the query's own
    question rather than a paraphrase of it.
  * **Labels are cached and content-addressed.** The cache key is
    (model, sha1(question), sha1(choices), row key), so raising `--rate`, re-running a
    query, or adding a CERT split re-pays only for rows never labeled before. Changing
    the question changes the key, which is the point: a different question is a
    different label.

CLI::

    python3 oracle_label.py --ids ids.txt --corpus IMAGES.csv --id-col id \\
      --image-col filename --image-dir /path/images \\
      --sql q2.sql --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-VL-2B-Instruct \\
      --out labels.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predicate  # noqa: E402
from vadar.paths import resolve_image_path  # noqa: E402
import semextract  # noqa: E402
import semvqa  # noqa: E402

TRUE, FALSE = "true", "false"
BOOL_CHOICES = (TRUE, FALSE)

# A label is only trustworthy when its score came from the model's own token
# logprobs. "self" is the model rating itself (poorly calibrated) and "default" is
# semvqa's placeholder, meaning no score was recoverable at all.
TRUSTED_SCORE_SOURCES = frozenset({"logprobs"})

# Abstentions worth remembering: the ROW is what is broken, so re-asking cannot change
# the outcome and a permanently bad row would otherwise be re-queried every run.
#
# Every other abstention reason -- an endpoint error, an unparseable response, an
# uncalibrated score -- is a property of the CALL, not the row, and is usually a
# configuration fault affecting the whole batch. Caching those makes one bad run
# permanent and, worse, silent: the retry reports the rows as cached and never surfaces
# the original error. This bit us with a reasoning model whose chain of thought
# consumed the token budget, leaving no `content`: 25/25 rows abstained and cached.
CACHEABLE_ABSTENTIONS = frozenset({"missing-image"})


@dataclass(frozen=True)
class Label:
    """One oracle verdict. `value is None` means ABSTAIN — recorded, never labeled."""
    key: str
    value: str | None
    score: float
    score_source: str
    latency_ms: int
    error: str | None = None
    cached: bool = False

    @property
    def abstained(self) -> bool:
        return self.value is None


@dataclass
class LabelRun:
    """The outcome of labeling a batch, including what it cost."""
    labels: dict[str, str] = field(default_factory=dict)
    detail: dict[str, dict[str, Any]] = field(default_factory=dict)
    abstained: list[str] = field(default_factory=list)
    calls: int = 0
    cache_hits: int = 0
    wall_ms: int = 0

    @property
    def mean_latency_ms(self) -> int:
        lat = [d["latency_ms"] for d in self.detail.values() if not d["cached"]]
        return int(sum(lat) / len(lat)) if lat else 0

    def summary(self) -> str:
        n = len(self.detail)
        pos = sum(1 for v in self.labels.values() if str(v).lower() == TRUE)
        return (f"[oracle] {len(self.labels)}/{n} labeled "
                f"({len(self.abstained)} abstained), {pos} positive; "
                f"{self.calls} calls, {self.cache_hits} cached, "
                f"{self.wall_ms / 1000:.1f}s wall, {self.mean_latency_ms}ms mean")


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------

LABEL_PROTOCOL_VERSION = "multimodal-fields-v2"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def cache_key(model: str, question: str, choices: Sequence[str] | None, row_key: str) -> str:
    """Content-addressed so a changed question invalidates exactly the labels it
    should, and nothing else."""
    return "|".join((LABEL_PROTOCOL_VERSION, model, _sha1(question),
                     _sha1(",".join(choices or ())), row_key))


class LabelCache:
    """A flat JSON cache. Small enough that atomicity beats cleverness: the file is
    rewritten whole on flush, via a temp file + rename so an interrupted run cannot
    leave a truncated cache behind."""

    def __init__(self, path: str | None):
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    self.data = json.load(handle)
            except (OSError, ValueError) as exc:
                print(f"[oracle] ignoring unreadable cache {path}: {exc}", file=sys.stderr)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.data.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.data[key] = record

    def flush(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2)
        os.replace(tmp, self.path)


# --------------------------------------------------------------------------
# labeling
# --------------------------------------------------------------------------

def normalize_bool(answer: str) -> str | None:
    """Map an oracle answer onto the true/false vocabulary the scorer compares against.

    Guided decoding constrains the answer to the enum, but a server that ignores
    `guided_json` can still return prose; anything unrecognised abstains rather than
    being coerced to false, which would look like a confident negative label.
    """
    text = str(answer).strip().strip(".").lower()
    if text in ("true", "yes", "y", "1"):
        return TRUE
    if text in ("false", "no", "n", "0"):
        return FALSE
    return None


def label_one(row_key: str, *, question: str, cfg: Any,
              choices: Sequence[str] | None = None,
              image_paths: Sequence[str] | None = None,
              text: str | None = None,
              boolean: bool = True) -> Label:
    """One oracle call. Never raises — a failure becomes an abstention."""
    started = time.monotonic()
    if image_paths:
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            return Label(row_key, None, 0.0, "missing-image", 0,
                         error=f"image not found: {missing[0]}")
        # A multimodal join needs BOTH sides. Previously q7 attached the logo image
        # but silently discarded `text1` (the airline), so the VLM marked the same
        # logo true for many unrelated airlines. Put structured fields in the VLM
        # question while retaining the image attachments.
        multimodal_question = question
        if text:
            multimodal_question += "\nProvided structured fields:\n" + text
        detail = semvqa.imgs_vqa_detail(list(image_paths), multimodal_question,
                                        list(choices) if choices else None, cfg=cfg)
    else:
        detail = semvqa.txt_vqa_detail(text or "", question,
                                       list(choices) if choices else None, cfg=cfg)
    latency = int((time.monotonic() - started) * 1000)

    source = detail.get("score_source", "default")
    if detail.get("error"):
        return Label(row_key, None, float(detail.get("score", 0.0)), source, latency,
                     error=str(detail["error"]))
    if source not in TRUSTED_SCORE_SOURCES:
        # No calibrated score means no way to tell a confident answer from a coin
        # flip. Record the answer in `error` for debugging, but do not label.
        return Label(row_key, None, float(detail.get("score", 0.0)), source, latency,
                     error=f"uncalibrated score_source={source!r}, "
                           f"answer={detail.get('answer')!r}")

    answer = detail.get("answer", "")
    value = normalize_bool(answer) if boolean else str(answer)
    if value is None or (not boolean and not str(value).strip()):
        return Label(row_key, None, float(detail.get("score", 0.0)), source, latency,
                     error=f"unusable answer {answer!r}")
    return Label(row_key, value, float(detail.get("score", 0.0)), source, latency)


def label_batch(keys: Sequence[str], make_call: Callable[[str], Label], *,
                model: str, question: str, choices: Sequence[str] | None = None,
                cache: LabelCache | None = None, concurrency: int = 8,
                progress_every: int = 25) -> LabelRun:
    """Label every key, reusing cached verdicts and running the rest concurrently.

    `make_call` builds and performs the oracle call for one key; keeping it a callback
    is what lets per-row and pairwise labeling share this driver unchanged.
    """
    run = LabelRun()
    started = time.monotonic()
    cache = cache if cache is not None else LabelCache(None)

    pending: list[str] = []
    for key in keys:
        hit = cache.get(cache_key(model, question, choices, key))
        if hit is None:
            pending.append(key)
            continue
        run.cache_hits += 1
        _record(run, Label(key, hit.get("value"), float(hit.get("score", 0.0)),
                           str(hit.get("score_source", "cache")),
                           int(hit.get("latency_ms", 0)),
                           error=hit.get("error"), cached=True))

    if pending:
        workers = max(1, min(concurrency, len(pending)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for done, label in enumerate(pool.map(make_call, pending), start=1):
                run.calls += 1
                _record(run, label)
                if not label.abstained or label.score_source in CACHEABLE_ABSTENTIONS:
                    cache.put(cache_key(model, question, choices, label.key),
                              {k: v for k, v in asdict(label).items() if k != "cached"})
                if progress_every and done % progress_every == 0:
                    print(f"[oracle] {done}/{len(pending)} labeled", file=sys.stderr)
        cache.flush()

    run.wall_ms = int((time.monotonic() - started) * 1000)
    return run


def _record(run: LabelRun, label: Label) -> None:
    run.detail[label.key] = {k: v for k, v in asdict(label).items() if k != "key"}
    if label.abstained:
        run.abstained.append(label.key)
    else:
        run.labels[label.key] = label.value


# --------------------------------------------------------------------------
# row -> call wiring
# --------------------------------------------------------------------------

def make_row_caller(rows_by_id: dict[str, dict[str, str]], *, question: str, cfg: Any,
                    choices: Sequence[str] | None, boolean: bool,
                    image_col: str | None, image_dir: str | None,
                    text_cols: Sequence[str] | None) -> Callable[[str], Label]:
    """A `make_call` for per-row labeling: one row in, one oracle verdict out."""
    def call(row_id: str) -> Label:
        row = rows_by_id.get(row_id)
        if row is None:
            return Label(row_id, None, 0.0, "missing-row", 0, error="id not in corpus")
        images = None
        if image_col:
            images = [resolve_image_path(str(row.get(image_col, "")), image_dir)]
        text = None
        if text_cols:
            text = "\n".join(f"{c}: {row.get(c, '')}" for c in text_cols)
        # predicate.py preserves SQL concatenation operands as `{column}` tokens.
        # Substitute this row's actual values before asking the oracle: leaving the
        # placeholders literal made q6c interpret “flights to Europe” as a destination
        # named Europe instead of recognizing London/Frankfurt as European cities.
        row_question = question
        for column, value in row.items():
            row_question = row_question.replace("{" + column + "}", str(value or ""))
        # BigQuery FORMAT("... %s ...", row.column) is common outside mmqa. The
        # predicate parser retains the format marker, so substitute the semantic
        # input columns in order just as the SQL expression would.
        for column in text_cols or ():
            if "%s" not in row_question:
                break
            row_question = row_question.replace("%s", str(row.get(column, "") or ""), 1)
        return label_one(row_id, question=row_question, cfg=cfg, choices=choices,
                         image_paths=images, text=text, boolean=boolean)
    return call


def make_pair_caller(rows_by_id: dict[str, dict[str, str]], *, question: str, cfg: Any,
                     choices: Sequence[str] | None, boolean: bool,
                     image_cols: Sequence[str], image_dir: str | None,
                     text_cols: Sequence[str] | None) -> Callable[[str], Label]:
    """A `make_call` for PAIRWISE labeling: one frame row carries both sides.

    The frame row already holds `file1`/`file2` (see build_pairs.py), so the two
    images are attached to a single call in that order. Order matters even for a
    symmetric predicate, because the question refers to "the first"/"the second"
    product -- so the images are never sorted.
    """
    def call(key: str) -> Label:
        row = rows_by_id.get(key)
        if row is None:
            return Label(key, None, 0.0, "missing-row", 0, error="pair id not in frame")
        images = [resolve_image_path(str(row.get(c, "")), image_dir)
                  for c in image_cols]
        text = None
        if text_cols:
            text = "\n".join(f"{c}: {row.get(c, '')}" for c in text_cols)
        return label_one(key, question=question, cfg=cfg, choices=choices,
                         image_paths=images, text=text, boolean=boolean)
    return call


def read_corpus(path: str, id_col: str) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"corpus {path} is empty")
    if id_col not in rows[0]:
        raise SystemExit(f"--id-col {id_col!r} not in {path}; "
                         f"available: {', '.join(rows[0].keys())}")
    return {str(r[id_col]): r for r in rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Label rows with an oracle model.")
    ap.add_argument("--ids", required=True, help="File with one row id per line.")
    ap.add_argument("--corpus", required=True, help="Corpus CSV.")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--image-col", help="Column holding the image filename.")
    ap.add_argument("--image-dir", help="Directory the image filenames are relative to.")
    ap.add_argument("--text-col", action="append", default=[],
                    help="Text column to show the oracle; repeatable.")

    ap.add_argument("--sql", help="Query .sql — the question is read from its AI call.")
    ap.add_argument("--call-site", type=int, help="Which AI call site (see predicate.py).")
    ap.add_argument("--question", help="Question text, when not using --sql.")

    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--cache", help="Label cache JSON (reused across runs).")
    ap.add_argument("--out", required=True, help="Where to write the labels JSON.")
    args = ap.parse_args(argv)

    question, choices, boolean = args.question, None, True
    if args.sql:
        sites = predicate.parse_file(args.sql)
        try:
            site = predicate.pick_site(sites, args.call_site, "per_row")
        except ValueError as exc:
            raise SystemExit(f"[oracle] {exc}") from exc
        if site.is_pairwise:
            raise SystemExit(
                f"[oracle] call site {args.call_site} is PAIRWISE ({site.reason}). "
                f"A per-row label is undefined for it.")
        question = site.prompt
        choices = list(site.choices) or None
        boolean = site.kind == "if"
    if not question:
        raise SystemExit("[oracle] need --question or --sql")
    if not boolean and not choices:
        choices = None          # free-form AI.GENERATE

    rows_by_id = read_corpus(args.corpus, args.id_col)
    with open(args.ids, encoding="utf-8") as handle:
        ids = [line.strip() for line in handle if line.strip()]

    cfg = semvqa.build_cfg(args.model, args.endpoint, args.api_key,
                           max_new_tokens=args.max_new_tokens, timeout=args.timeout)
    caller = make_row_caller(rows_by_id, question=question, cfg=cfg, choices=choices,
                             boolean=boolean, image_col=args.image_col,
                             image_dir=args.image_dir, text_cols=args.text_col or None)
    run = label_batch(ids, caller, model=args.model, question=question, choices=choices,
                      cache=LabelCache(args.cache), concurrency=args.concurrency)

    payload = {
        "question": question, "choices": choices, "boolean": boolean,
        "model": args.model, "labels": run.labels, "detail": run.detail,
        "abstained": run.abstained,
        "cost": {"calls": run.calls, "cache_hits": run.cache_hits,
                 "wall_ms": run.wall_ms, "mean_latency_ms": run.mean_latency_ms},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(run.summary())
    print(f"[oracle] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
