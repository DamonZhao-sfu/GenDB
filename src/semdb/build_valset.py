"""Build a sampled validation set: draw rows under a probability design, label
them, and write a disjoint SELECT / CERT pair.

    SELECT  the refinement loop may read this freely; nothing it produces is an
            outward claim, so looking at it as often as you like costs nothing
            but the labels.
    CERT    sealed. The orchestrator never reads it. It is touched exactly once,
            after the run, to produce a certificate on a program that was chosen
            without ever seeing it.

The two are drawn as one sample and partitioned, so they are disjoint by
construction (see `sampling.split_sample`).

Sampling designs come from `sampling.py`; this module only decides WHAT to
stratify on and WHAT to score by, which must be signals available BEFORE any
program exists -- otherwise the design is defined in terms of the thing it is
supposed to measure:

    --strata-by length          text-length quantile buckets
    --strata-by kmeans          TF-IDF k-means cluster id (unsupervised)
    --strata-by score-decile    quantile buckets of the --importance-by score
    --strata-by column:<name>   an existing categorical column
    --importance-by query-similarity   TF-IDF cosine against the query text
    --importance-by clip-similarity    CLIP text-image cosine (image corpora)
    --importance-by column:<name>      an existing numeric column

Size may be given as a count (`--n`) or as a **rate** (`--rate 0.2`, meaning 20% of
the corpus). Rates are what callers usually want, because they survive a change of
scale factor unchanged.

Why the sampling design is not a free choice: positives are rare in these
benchmarks. On ecomm sf_250, Q2 has 5 positives in 250 rows (2.0%) and Q13 has 12
(4.8%). A uniform 20% sample of Q2 draws ~1 positive, and a validation set with one
positive cannot tell a good program apart from `return false` -- both score ~98%.
So for a selective predicate, prefer `--method stratified --strata-by score-decile`,
which spends labels where the answer is actually in doubt while keeping exact
inclusion probabilities. `--zero-class-guard` (on by default) refuses to write a
val set whose SELECT half is single-class, because such a file cannot separate any
program from a constant.

Labels come from `--label-source`:

    oracle   an LLM/VLM over an OpenAI-compatible endpoint (see oracle_label.py).
             The question is read from the query's own AI.IF/AI.GENERATE/AI.CLASSIFY
             call via predicate.py, so the oracle is asked what the query asks.
    gt       the benchmark answer key -- a plumbing fixture only (see GT_WARNING).
    none     emit ids without labels, to be labeled separately.

Example (oracle-labeled, stratified, image corpus)::

    python3 src/semdb/build_valset.py \
      --corpus runs/_materialized/ecomm_sf250/IMAGES.csv --id-col id \
      --image-col filename --image-dir /path/ecomm/data/sf_250/images \
      --query q2 --attr matches --sql /path/queries/bigquery/q2.sql \
      --method stratified --strata-by score-decile --importance-by clip-similarity \
      --rate 0.25 --cert-rate 0.1 --seed 7 \
      --label-source oracle --endpoint http://localhost:8000/v1 \
      --oracle-model Qwen/Qwen3-VL-2B-Instruct \
      --out src/semdb/runs/_val/ecomm-q2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sampling  # noqa: E402

# Labels a ground-truth-backed source emits. Kept as strings because
# evaluate.score_inference compares str(v).strip().lower().
TRUE, FALSE = "true", "false"

PPS_WARNING = (
    "This is an unequal-probability (pps) sample: rows were deliberately drawn at "
    "different rates, so the UNWEIGHTED accuracy over these rows is NOT an estimate "
    "of corpus accuracy -- it over-weights the oversampled region. The `weights` "
    "field carries what a correct estimator needs, but the current --val-file scorer "
    "ignores it. Until the weighted estimator lands, use --method uniform (or "
    "stratified with proportional allocation), where the unweighted mean is unbiased."
)

GT_WARNING = (
    "Labels came from the benchmark ground truth, NOT from an oracle model. "
    "This is a plumbing/meta-evaluation fixture: a validation set built this way "
    "cannot support any claim about oracle-free operation, and the refinement "
    "loop consuming it is reading the answer key. Use --label-source oracle for "
    "a real run."
)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def read_corpus(path: str, id_col: str,
                text_cols: Sequence[str] | None) -> list[dict[str, str]]:
    """Read a CSV into row dicts, validating that the named columns exist."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"cannot read corpus {path}: {exc}") from exc
    if not rows:
        raise SystemExit(f"corpus {path} is empty")
    header = rows[0].keys()
    for col in [id_col, *(text_cols or [])]:
        if col not in header:
            raise SystemExit(f"column {col!r} not in {path}; available: {', '.join(header)}")
    seen: set[str] = set()
    for row in rows:
        rid = str(row[id_col])
        if rid in seen:
            raise SystemExit(f"duplicate id {rid!r} in {path} -- ids must be unique")
        seen.add(rid)
    return rows


def _texts(rows: Sequence[dict[str, str]],
           text_cols: Sequence[str] | None) -> list[str]:
    """Render every predicate input column into one natural row description.

    Both the relevance scorer and the oracle must see the same semantic inputs.
    Keeping all columns matters for predicates such as mmqa q6, which is a function
    of both Airlines and Destinations rather than the old last-column heuristic
    (Airport). Period-separated field/value clauses also give text encoders enough
    context to connect Frankfurt→Germany→Europe.
    """
    cols = list(text_cols or [])
    if not cols:
        return [""] * len(rows)
    return [
        " ".join(f"{c}: {str(r.get(c, '') or '').strip()}." for c in cols)
        for r in rows
    ]


# --------------------------------------------------------------------------
# strata (pre-program signals only)
# --------------------------------------------------------------------------

def strata_by_length(texts: Sequence[str], k: int) -> list[str]:
    """Quantile buckets of text length. Ties collapse, so fewer than k buckets
    may come back -- that is fine, the allocator works off realized sizes."""
    lengths = np.array([len(t) for t in texts], dtype=float)
    edges = np.quantile(lengths, np.linspace(0, 1, k + 1)[1:-1]) if k > 1 else np.array([])
    return [f"len{int(np.searchsorted(edges, x, side='right'))}" for x in lengths]


def strata_by_kmeans(texts: Sequence[str], k: int, seed: int) -> list[str]:
    """TF-IDF + k-means cluster id. Unsupervised and query-independent, so it
    cannot leak the label into the design."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    matrix = TfidfVectorizer(max_features=4096, stop_words="english").fit_transform(texts)
    k = min(k, matrix.shape[0])
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(matrix)
    return [f"c{int(c)}" for c in labels]


def strata_by_score_decile(scores: Sequence[float], k: int) -> list[str]:
    """Equal-count buckets of a pre-program relevance score, lowest first.

    This is the design that makes a rare-positive predicate measurable. The score
    (CLIP text-image similarity, TF-IDF cosine, ...) correlates with the label
    without being the label, so the top buckets concentrate the candidate positives.
    Combined with Neyman allocation the sample spends its labels where the answer is
    in doubt, while inclusion probabilities stay EXACT -- unlike pps, where the
    weight identity only holds in expectation.

    Bucketing is by RANK, not by quantile cut points. Quantile edges collapse when
    the score has heavy ties -- a corpus of near-duplicate texts can put every row
    above all three quartile edges, yielding a single stratum. That failure is
    invisible: the design still calls itself "stratified" while having silently
    degraded to a uniform draw. Ranking always yields k buckets of near-equal size.
    Ties land in adjacent buckets in a deterministic (stable-sort) order, which
    costs a little within-stratum homogeneity and keeps the design valid.
    """
    values = np.nan_to_num(np.asarray(scores, dtype=float),
                           nan=0.0, posinf=0.0, neginf=0.0)
    n = len(values)
    if n == 0 or k <= 1:
        return ["s0"] * n
    k = min(k, n)
    order = np.argsort(values, kind="stable")       # ascending, ties keep input order
    bucket = np.empty(n, dtype=int)
    bucket[order] = (np.arange(n) * k) // n
    distinct = len(np.unique(values))
    if distinct < k:
        print(f"[build_valset] note: the score takes only {distinct} distinct value(s) "
              f"across {n} rows, so score strata carry little signal; consider a "
              f"different --importance-by source", file=sys.stderr)
    return [f"s{int(b)}" for b in bucket]


def tilt_from_scores(scores: Sequence[float], strata: Sequence[str],
                     gamma: float = 2.0) -> dict[str, float]:
    """Disproportionate allocation weights that tilt labels toward high-score strata.

    `sampling.allocate` sets n_h proportional to N_h * sigma_h, so returning
    sigma_h = (mean rank-percentile of stratum h) ** gamma buys a controlled tilt
    toward the strata where the positives live.

    Why not textbook Neyman (sigma_h = sqrt(p_h(1-p_h)))? Neyman minimizes the
    variance of the estimated MEAN. For a 2%-positive predicate that is the wrong
    estimand: what the validation set has to support is the program's error on the
    positives, and a sample with no positives in it cannot estimate that at any
    sample size. Measured on ecomm q2, CLIP similarity ranks all five positives in
    the top six of 250 rows, yet a p(1-p) sigma came out nearly flat (10/14/15/16/15
    across five strata) and drew just one of them -- worse than a uniform draw of the
    same size. This is standard practice for rare populations: sample the screened
    stratum hard, sample the rest sparsely, and let the (exact, for stratified)
    Horvitz-Thompson weights put the estimate back on the population scale.

    The trade-off `gamma` controls is real. A higher tilt captures more positives but
    spreads the HT weights further apart, which makes the weighted estimate noisier
    per labeled row. gamma=2 keeps every stratum observable while sampling the top
    stratum several times harder than the bottom one.

    Rank percentiles, not raw scores: CLIP cosines sit in a narrow band (~0.20-0.33
    here), so rescaling raw values compresses every stratum toward the same tilt.
    """
    values = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    n = len(values)
    if n == 0:
        return {}
    # Rank percentile in (0, 1]; ties resolved by stable order, as in the bucketing.
    percentile = np.empty(n, dtype=float)
    percentile[np.argsort(values, kind="stable")] = (np.arange(n) + 1) / n
    out: dict[str, float] = {}
    for stratum in set(strata):
        mask = np.array([s == stratum for s in strata])
        # Floor keeps a low-score stratum observable rather than allocated nothing;
        # every stratum must retain a positive inclusion probability.
        out[stratum] = max(float(percentile[mask].mean()) ** gamma, 1e-3)
    return out


def build_strata(rows: Sequence[dict[str, str]], texts: Sequence[str],
                 spec: str, k: int, seed: int,
                 scores: Sequence[float] | None = None) -> list[str]:
    if spec == "length":
        return strata_by_length(texts, k)
    if spec == "kmeans":
        return strata_by_kmeans(texts, k, seed)
    if spec == "score-decile":
        if scores is None:
            raise SystemExit("--strata-by score-decile needs an --importance-by source")
        return strata_by_score_decile(scores, k)
    if spec.startswith("column:"):
        col = spec.split(":", 1)[1]
        if col not in rows[0]:
            raise SystemExit(f"--strata-by column:{col} -- no such column")
        return [str(r.get(col, "") or "(blank)") for r in rows]
    raise SystemExit(f"unknown --strata-by {spec!r}")


# --------------------------------------------------------------------------
# importance scores (pre-program signals only)
# --------------------------------------------------------------------------

def scores_by_query_similarity(texts: Sequence[str], query_text: str) -> list[float]:
    """TF-IDF cosine similarity to the query text.

    The point of this proposal is class imbalance: when positives are a few
    percent of the corpus, a uniform sample spends nearly every label on
    negatives and any recall-side estimate is built on a handful of rows. Rows
    that look like the query are likelier to be positive, so oversampling them
    buys more information per label -- and the pps weights put the estimate back
    on the right scale.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not query_text.strip():
        raise SystemExit("--importance-by query-similarity needs --query-nl or --query-sql text")
    vec = TfidfVectorizer(max_features=4096, stop_words="english")
    matrix = vec.fit_transform(list(texts) + [query_text])
    sims = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    return [float(s) for s in sims]


def scores_by_clip_text_similarity(texts: Sequence[str], query_text: str,
                                   clip_model: str) -> list[float]:
    """Semantic text-text relevance using the already configured local CLIP encoder.

    TF-IDF cannot connect a destination such as Frankfurt with Germany or Europe.
    CLIP is already a dependency of DIRECT image queries and supplies a pre-program,
    label-free signal that can make those relational text predicates sampleable.
    """
    if not query_text.strip():
        raise SystemExit("--importance-by clip-text-similarity needs query text")
    import semvision

    encoder = semvision.get_encoder(clip_model)
    query_vec = np.asarray(semvision.embed_text(query_text, encoder), dtype=np.float32)
    matrix = np.vstack([
        np.asarray(semvision.embed_text(text, encoder), dtype=np.float32)
        for text in texts
    ])
    return [float(x) for x in matrix @ query_vec]


def scores_by_clip_similarity(image_paths: Sequence[str], query_text: str,
                              clip_model: str) -> list[float]:
    """CLIP text-image cosine between the predicate and each corpus image.

    The image analogue of `scores_by_query_similarity`, and the only pre-program
    signal available for an image corpus with no text column. One batched encode of
    the corpus plus one text encode -- no VLM calls, so it costs a fraction of a
    single label. Unreadable images score 0 rather than aborting the build.
    """
    import semvision

    if not query_text.strip():
        raise SystemExit("--importance-by clip-similarity needs --sql or --query-nl")
    encoder = semvision.get_encoder(clip_model)
    matrix = semvision.embed_corpus(list(image_paths), encoder, on_error="zero")
    text_vec = semvision.embed_text(query_text, encoder)
    return [float(s) for s in np.asarray(matrix) @ np.asarray(text_vec).ravel()]


def build_scores(rows: Sequence[dict[str, str]], texts: Sequence[str],
                 spec: str, query_text: str,
                 image_paths: Sequence[str] | None = None,
                 clip_model: str = "openai/clip-vit-base-patch32") -> list[float]:
    if spec == "query-similarity":
        return scores_by_query_similarity(texts, query_text)
    if spec == "clip-text-similarity":
        return scores_by_clip_text_similarity(texts, query_text, clip_model)
    if spec == "clip-similarity":
        if not image_paths:
            raise SystemExit("--importance-by clip-similarity needs --image-col/--image-dir")
        return scores_by_clip_similarity(image_paths, query_text, clip_model)
    if spec.startswith("column:"):
        col = spec.split(":", 1)[1]
        if col not in rows[0]:
            raise SystemExit(f"--importance-by column:{col} -- no such column")
        out = []
        for r in rows:
            try:
                out.append(float(r.get(col) or 0.0))
            except ValueError:
                out.append(0.0)
        return out
    raise SystemExit(f"unknown --importance-by {spec!r}")


def sample_stratified_with_score_anchors(
        ids: Sequence[str], strata: Sequence[str], scores: Sequence[float], n: int,
        seed: int, *, gamma: float, min_per_stratum: int) -> sampling.Sample:
    """Stratified sample with a small, probability-valid certainty top-score stratum.

    Merely allocating more draws to the highest score decile does not guarantee that
    any of its genuinely rare positives are observed: q7 has positives at ranks 18
    and 31 of a 400-pair screened frame, yet its seeded 10% SRS draw selected none.
    Survey designs handle this with *certainty units*. We take at most half the label
    budget (and at most sqrt(N) rows) from the score-ranked head with pi=1, then draw
    the remaining budget by ordinary stratified SRSWOR. Every remaining row retains
    positive inclusion probability, so Horvitz-Thompson weights remain exact.
    """
    ids = [str(i) for i in ids]
    strata = [str(h) for h in strata]
    N = len(ids)
    desired = min(n // 2, int(math.ceil(math.sqrt(N))))
    order = np.argsort(-np.nan_to_num(np.asarray(scores, dtype=float),
                                      nan=-np.inf), kind="stable")

    # Leave enough budget to observe every realized non-certainty stratum. One
    # residual draw per stratum is sufficient because the certainty head already
    # consumes part of the budget; requiring the ordinary two-row floor made a
    # two-row cars validation sample fall back to two random rows and lose the rare
    # high-score candidate entirely.
    residual_min = 1
    anchors = 0
    for candidate in range(desired, 0, -1):
        chosen = set(int(i) for i in order[:candidate])
        remaining_strata = {strata[i] for i in range(N) if i not in chosen}
        if n - candidate >= residual_min * len(remaining_strata):
            anchors = candidate
            break
    if anchors <= 0:
        sigma = tilt_from_scores(scores, strata, gamma)
        return sampling.sample_stratified(
            ids, dict(zip(ids, strata)), n, seed, allocation="neyman", sigma=sigma,
            min_per_stratum=min_per_stratum)

    anchor_idx = {int(i) for i in order[:anchors]}
    anchor_ids = [ids[i] for i in order[:anchors]]
    rem_ids = [ids[i] for i in range(N) if i not in anchor_idx]
    rem_strata = [strata[i] for i in range(N) if i not in anchor_idx]
    rem_scores = [scores[i] for i in range(N) if i not in anchor_idx]
    rem_sigma = tilt_from_scores(rem_scores, rem_strata, gamma)
    remainder = sampling.sample_stratified(
        rem_ids, dict(zip(rem_ids, rem_strata)), n - anchors, seed,
        allocation="neyman", sigma=rem_sigma,
        min_per_stratum=residual_min)

    pi = {i: 1.0 for i in anchor_ids}
    pi.update(remainder.pi)
    stratum = {i: "_certainty_top_score" for i in anchor_ids}
    stratum.update(remainder.stratum)
    meta = dict(remainder.meta)
    meta.update({
        "n": n,
        "certainty_n": anchors,
        "certainty_rule": "top min(floor(n/2), ceil(sqrt(N))) by pre-program score",
        "strata": {
            "_certainty_top_score": {"N_h": anchors, "n_h": anchors},
            **dict(remainder.meta.get("strata", {})),
        },
    })
    return sampling.Sample(
        ids=tuple(anchor_ids) + remainder.ids, pi=pi, stratum=stratum,
        method="stratified", seed=seed, N=N, meta=meta)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def labels_from_ground_truth(rows: Sequence[dict[str, str]], ids: Sequence[str],
                             id_col: str, gt_file: str, match_col: str) -> dict[str, str]:
    """Binary membership labels from a SemBench ground-truth JSON.

    The GT file lists the values of `match_col` that satisfy the query, so the
    label for a row is whether its `match_col` value is in that set.
    """
    with open(gt_file, encoding="utf-8") as handle:
        payload = json.load(handle)
    gold = payload.get("ground_truth", payload) if isinstance(payload, dict) else payload
    if not isinstance(gold, list):
        raise SystemExit(f"{gt_file}: expected a list under 'ground_truth'")
    gold_set = {str(g).strip().lower() for g in gold}
    if match_col not in rows[0]:
        raise SystemExit(f"--gt-match-col {match_col!r} not in the corpus")
    by_id = {str(r[id_col]): str(r.get(match_col, "") or "").strip().lower() for r in rows}
    return {i: (TRUE if by_id.get(i, "") in gold_set else FALSE) for i in ids}


def labels_from_oracle(rows: Sequence[dict[str, str]], ids: Sequence[str], id_col: str,
                       *, question: str, choices: Sequence[str] | None, boolean: bool,
                       model: str, endpoint: str, api_key: str, concurrency: int,
                       image_col: str | None, image_dir: str | None,
                       text_cols: Sequence[str] | None,
                       cache_path: str | None,
                       pair_image_cols: Sequence[str] | None = None,
                       ) -> tuple[dict[str, str], dict[str, Any]]:
    """Label the sampled ids with the oracle model. Returns (labels, report).

    Abstentions are NOT labeled — see oracle_label for why a guessed label is worse
    than a missing one. The report carries the cost and the abstention list so the
    caller can record both in the val file's provenance.
    """
    import oracle_label

    cfg = oracle_label.semvqa.build_cfg(model, endpoint, api_key)
    by_id = {str(r[id_col]): r for r in rows}
    if pair_image_cols:
        caller = oracle_label.make_pair_caller(
            by_id, question=question, cfg=cfg, choices=choices, boolean=boolean,
            image_cols=pair_image_cols, image_dir=image_dir, text_cols=text_cols)
    else:
        caller = oracle_label.make_row_caller(
            by_id, question=question, cfg=cfg, choices=choices, boolean=boolean,
            image_col=image_col, image_dir=image_dir, text_cols=text_cols)
    run = oracle_label.label_batch(
        list(ids), caller, model=model, question=question, choices=choices,
        cache=oracle_label.LabelCache(cache_path), concurrency=concurrency)
    print(run.summary(), file=sys.stderr)
    report = {
        "question": question, "choices": list(choices) if choices else None,
        "model": model, "abstained": run.abstained,
        "cost": {"calls": run.calls, "cache_hits": run.cache_hits,
                 "wall_ms": run.wall_ms, "mean_latency_ms": run.mean_latency_ms},
    }
    return dict(run.labels), report


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def class_balance(ids: Sequence[str], labels: dict[str, str] | None) -> dict[str, int]:
    """Label histogram over `ids`, counting unlabeled rows explicitly."""
    counts: dict[str, int] = {}
    for i in ids:
        value = labels.get(i) if labels else None
        key = "(unlabeled)" if value is None else str(value).strip().lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def zero_class_message(counts: dict[str, int]) -> str | None:
    """Why this val set cannot be used, or None when it can.

    A single-class SELECT half scores the constant predictor perfectly, so every
    program looks equally good and the refinement loop has nothing to climb. That is
    a design failure, not a small loss of power, and it must not be discovered later
    as a suspiciously perfect accuracy.
    """
    labeled = {k: v for k, v in counts.items() if k != "(unlabeled)"}
    if not labeled:
        return "no labeled rows at all"
    if len(labeled) < 2:
        only, n = next(iter(labeled.items()))
        return (f"every one of the {n} labeled rows is {only!r}. A single-class "
                f"validation set scores the constant predictor perfectly, so it "
                f"cannot rank programs. Raise --rate, or use "
                f"--method stratified --strata-by score-decile to concentrate "
                f"labels where the answer is in doubt.")
    return None


def resolve_size(count: int | None, rate: float | None, population: int,
                 label: str, *, required: bool) -> int:
    """A size given either as a count or as a fraction of the corpus.

    Rates are what callers usually want because they survive a change of scale
    factor; counts are what the sampler needs. Supplying both is refused rather than
    silently resolved, since the two would disagree the moment the corpus changes.
    """
    if count is not None and rate is not None:
        raise SystemExit(f"{label}: give one or the other, not both")
    if count is not None:
        if count < 0:
            raise SystemExit(f"{label}: negative size {count}")
        return count
    if rate is not None:
        if not 0.0 <= rate <= 1.0:
            raise SystemExit(f"{label}: rate {rate} is not in [0, 1]")
        size = int(math.ceil(rate * population))
        if rate > 0 and size == 0:
            size = 1
        return size
    if required:
        raise SystemExit(f"{label} is required")
    return 0


def resolve_question(sql_path: str | None, call_site: int | None, query_nl: str,
                     shape: str = "per_row",
                     ) -> tuple[str, list[str] | None, bool, dict[str, Any]]:
    """(question, choices, boolean, site_info) for the oracle.

    Reading the question out of the query's own AI call beats retyping it: the
    validation set then measures the predicate the query actually states, not a
    paraphrase. A PAIRWISE call site is refused here rather than labeled per-row,
    because a per-row label for a two-row predicate does not exist.
    """
    if not sql_path:
        if not query_nl.strip():
            raise SystemExit("--label-source oracle needs --sql or --query-nl")
        return query_nl, None, True, {"source": "--query-nl"}

    import predicate

    sites = predicate.parse_file(sql_path)
    try:
        site = predicate.pick_site(sites, call_site, shape)
    except ValueError as exc:
        raise SystemExit(f"[build_valset] {exc}") from exc
    if site.shape != shape:
        want = "PAIRWISE" if site.is_pairwise else "PER-ROW"
        raise SystemExit(
            f"[build_valset] call site {call_site} is {want} ({site.reason}), but a "
            f"{shape} validation set was requested. A pairwise val set keys on "
            f"'<id1>-<id2>' and needs --pairwise with a frame from build_pairs.py.")
    return (site.prompt, list(site.choices) or None, site.kind == "if",
            {"source": os.path.abspath(sql_path), "kind": site.kind, "line": site.line,
             "shape": site.shape, "aliases": list(site.aliases)})


def design_block(sample: sampling.Sample) -> dict[str, Any]:
    """The `design` field of a val file -- everything an estimator needs to know
    about how these rows came to be here."""
    return {
        "method": sample.method, "seed": sample.seed, "N": sample.N, "n": sample.n,
        "weight_sum": round(sample.weight_sum(), 4),
        "pi": {i: round(sample.pi[i], 8) for i in sample.ids},
        "stratum_of": dict(sample.stratum) if any(sample.stratum.values()) else {},
        **{k: v for k, v in sample.meta.items() if k != "pi"},
    }


def val_payload(sample: sampling.Sample, *, query: str, attr: str, split: str,
                labels: dict[str, str] | None, provenance: dict[str, Any]) -> dict[str, Any]:
    """A val file. `labels` is the only field the existing --val-file path reads;
    everything else is additive, so old fixtures keep working unchanged."""
    payload: dict[str, Any] = {"query": query, "attr": attr, "split": split}
    if labels is not None:
        payload["labels"] = {i: labels[i] for i in sample.ids if i in labels}
    payload["ids"] = list(sample.ids)
    payload["weights"] = {i: round(w, 6) for i, w in sample.weights.items()}
    payload["design"] = design_block(sample)
    payload["provenance"] = provenance
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="Corpus CSV.")
    ap.add_argument("--id-col", required=True, help="Unique row id column.")
    ap.add_argument("--text-col", action="append", default=[],
                    help="Unstructured text column (strata / similarity / oracle input). "
                         "Repeatable; the first is used for strata and similarity.")
    ap.add_argument("--image-col", help="Column holding the image filename (image corpora).")
    ap.add_argument("--image-dir", help="Directory the image filenames are relative to.")
    ap.add_argument("--query", required=True, help="Query id, e.g. q3a.")
    ap.add_argument("--attr", required=True, help="Attribute name the labels carry.")
    ap.add_argument("--query-nl", default="", help="Natural-language query text.")
    ap.add_argument("--label-type", choices=["boolean", "text"],
                    help="Override the AI call's output type for a typed joint Oracle "
                         "question supplied with --query-nl.")
    ap.add_argument("--label-choices-json",
                    help="JSON array constraining text labels for a typed joint Oracle "
                         "question.")
    ap.add_argument("--sql", help="Query .sql -- the oracle question is read from its "
                                  "AI.IF/AI.GENERATE/AI.CLASSIFY call (see predicate.py).")
    ap.add_argument("--call-site", type=int,
                    help="Which AI call site to use when the query has several.")
    ap.add_argument("--pairwise", action="store_true",
                    help="The corpus is a PAIR frame from build_pairs.py (id column "
                         "'pair_id', image columns file1/file2). Labels a two-row join "
                         "predicate; keys are '<id1>-<id2>'.")
    ap.add_argument("--pair-image-cols", default="file1,file2",
                    help="Comma-separated image columns of a pair frame, in the order "
                         "the question refers to them.")

    ap.add_argument("--method", default="uniform",
                    choices=["uniform", "stratified", "importance"])
    ap.add_argument("--n", type=int, help="SELECT size (or give --rate).")
    ap.add_argument("--rate", type=float, help="SELECT size as a fraction of the corpus.")
    ap.add_argument("--cert-n", type=int, help="CERT size (default 0 = no cert split).")
    ap.add_argument("--cert-rate", type=float, help="CERT size as a fraction of the corpus.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--strata-by", default="kmeans",
                    help="length | kmeans | score-decile | column:<name>")
    ap.add_argument("--strata-k", type=int, default=4)
    ap.add_argument("--min-per-stratum", type=int, default=2)
    ap.add_argument("--importance-by", default="query-similarity",
                    help="query-similarity | clip-text-similarity | "
                         "clip-similarity | column:<name>")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--score-tilt", type=float, default=2.0,
                    help="How hard to oversample high-score strata (0 = proportional). "
                         "Higher captures more positives but widens the HT weights, "
                         "making the weighted estimate noisier per labeled row.")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="Uniform mixing weight for the importance proposal; "
                         "bounds the weights at N/(n*epsilon).")
    ap.add_argument("--allow-single-class", action="store_true",
                    help="Write the val set even when its SELECT half is single-class. "
                         "Such a set cannot rank programs, so this is off by default.")

    ap.add_argument("--label-source", default="gt", choices=["gt", "none", "oracle"])
    ap.add_argument("--gt-file", help="SemBench ground-truth JSON (--label-source gt).")
    ap.add_argument("--gt-match-col", help="Corpus column the ground truth lists.")
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL (--label-source oracle).")
    ap.add_argument("--oracle-model", help="Oracle model id (--label-source oracle).")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--label-cache", help="Label cache JSON, reused across runs.")
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args(argv)

    text_cols = list(args.text_col)
    rows = read_corpus(args.corpus, args.id_col, text_cols)
    ids = [str(r[args.id_col]) for r in rows]
    texts = _texts(rows, text_cols)

    pair_image_cols = ([c.strip() for c in args.pair_image_cols.split(",") if c.strip()]
                       if args.pairwise else None)
    if pair_image_cols:
        missing = [c for c in pair_image_cols if c not in rows[0]]
        if missing:
            raise SystemExit(f"--pairwise expects a frame from build_pairs.py; "
                             f"column(s) {', '.join(missing)} not in {args.corpus}")

    image_paths = None
    if args.image_col:
        import semextract
        if args.image_col not in rows[0]:
            raise SystemExit(f"--image-col {args.image_col!r} not in {args.corpus}")
        image_paths = [semextract.resolve_image_path(str(r.get(args.image_col, "")),
                                                     args.image_dir) for r in rows]

    n = resolve_size(args.n, args.rate, len(ids), "--n/--rate", required=True)
    cert_n = resolve_size(args.cert_n, args.cert_rate, len(ids), "--cert-n/--cert-rate",
                          required=False)
    total = n + cert_n
    if total > len(ids):
        raise SystemExit(f"SELECT {n} + CERT {cert_n} = {total} exceeds the corpus "
                         f"size {len(ids)}")

    # The oracle's question doubles as the text the pre-program relevance score is
    # computed against, so resolve it before drawing the sample.
    question, choices, boolean, site_info = "", None, True, {}
    if args.label_source == "oracle" or args.sql:
        question, choices, boolean, site_info = resolve_question(
            args.sql, args.call_site, args.query_nl,
            "pairwise" if args.pairwise else "per_row")
    if args.label_type:
        boolean = args.label_type == "boolean"
        site_info = {**site_info, "label_type_override": args.label_type}
    if args.label_choices_json:
        try:
            parsed_choices = json.loads(args.label_choices_json)
        except ValueError as exc:
            raise SystemExit("--label-choices-json must be a JSON array") from exc
        if (not isinstance(parsed_choices, list)
                or not parsed_choices
                or not all(isinstance(value, str) and value.strip()
                           for value in parsed_choices)):
            raise SystemExit("--label-choices-json must be a non-empty JSON string array")
        choices = parsed_choices
        site_info = {**site_info, "choices_override": True}
    score_text = question or args.query_nl

    # --- draw ------------------------------------------------------------
    # Design errors (an infeasible allocation, a stratification finer than the
    # sample) are user errors, not bugs: surface the actionable message without a
    # traceback.
    scores: list[float] | None = None
    needs_scores = args.method == "importance" or args.strata_by == "score-decile"
    if needs_scores:
        scores = build_scores(rows, texts, args.importance_by, score_text,
                              image_paths=image_paths, clip_model=args.clip_model)

    try:
        if args.method == "uniform":
            sample = sampling.sample_uniform(ids, total, args.seed)
        elif args.method == "stratified":
            # A rate can resolve to fewer labeled rows than the configured number of
            # strata (cars sf_200 image tables: ceil(39 * 0.05) = 2 vs k=5). Coarsen
            # automatically instead of rejecting a perfectly valid rate request.
            # Each realized stratum still receives the configured minimum allocation.
            max_feasible_k = max(1, total // max(1, args.min_per_stratum))
            effective_k = min(args.strata_k, max_feasible_k)
            if effective_k < args.strata_k:
                print(f"[build_valset] coarsening --strata-k {args.strata_k} -> "
                      f"{effective_k}: sample size {total} cannot support more strata "
                      f"at --min-per-stratum {args.min_per_stratum}", file=sys.stderr)
            strata = build_strata(rows, texts, args.strata_by, effective_k, args.seed,
                                  scores=scores)
            # A stratum must be big enough to put at least one row on each side of
            # the SELECT/CERT split; anything thinner gets pooled.
            floor = max(args.min_per_stratum, 2 if cert_n else 1)
            merged, report = sampling.collapse_small_strata(dict(zip(ids, strata)), floor)
            if report["merged"]:
                print(f"[build_valset] pooled {len(report['merged'])} stratum/strata with "
                      f"< {floor} rows ({report['pooled_size']} rows) into "
                      f"{report['pooled_into']!r}", file=sys.stderr)
            # With a score, tilt the allocation toward the strata the positives are
            # screened into; without one there is nothing to tilt by.
            allocation, sigma = "proportional", None
            if scores is not None and args.score_tilt > 0:
                merged_strata = [merged[i] for i in ids]
                allocation = "neyman"        # n_h proportional to N_h * sigma_h
                sigma = tilt_from_scores(scores, merged_strata, args.score_tilt)
            if (scores is not None and args.score_tilt > 0
                    and args.strata_by == "score-decile"):
                sample = sample_stratified_with_score_anchors(
                    ids, [merged[i] for i in ids], scores, total, args.seed,
                    gamma=args.score_tilt, min_per_stratum=args.min_per_stratum)
            else:
                sample = sampling.sample_stratified(
                    ids, merged, total, args.seed, allocation=allocation, sigma=sigma,
                    min_per_stratum=args.min_per_stratum)
        else:
            sample = sampling.sample_pareto_pps(ids, scores, total, args.seed,
                                                epsilon=args.epsilon)
        sample.check()

        select, cert = sampling.split_sample(sample, cert_n, args.seed + 1)
        select.check()
        if cert.n:
            cert.check()
    except ValueError as exc:
        raise SystemExit(f"[build_valset] infeasible design: {exc}") from exc

    # --- label -----------------------------------------------------------
    warnings: list[str] = []
    if sample.method == "pareto_pps":
        warnings.append(PPS_WARNING)
        print(f"[build_valset] WARNING: {PPS_WARNING}", file=sys.stderr)
    if args.label_source == "gt":
        if not (args.gt_file and args.gt_match_col):
            raise SystemExit("--label-source gt requires --gt-file and --gt-match-col")
        labels = labels_from_ground_truth(rows, sample.ids, args.id_col,
                                          args.gt_file, args.gt_match_col)
        warnings.append(GT_WARNING)
        print(f"[build_valset] WARNING: {GT_WARNING}", file=sys.stderr)
    elif args.label_source == "none":
        labels = None
    else:
        if not (args.endpoint and args.oracle_model):
            raise SystemExit("--label-source oracle requires --endpoint and --oracle-model")
        if not (args.image_col or text_cols or pair_image_cols):
            raise SystemExit("--label-source oracle needs --image-col or --text-col: "
                             "the oracle has to be shown something")
        labels, oracle_report = labels_from_oracle(
            rows, sample.ids, args.id_col,
            question=question, choices=choices, boolean=boolean,
            model=args.oracle_model, endpoint=args.endpoint, api_key=args.api_key,
            concurrency=args.concurrency, image_col=args.image_col,
            image_dir=args.image_dir, text_cols=text_cols or None,
            cache_path=args.label_cache, pair_image_cols=pair_image_cols)
        if oracle_report["abstained"]:
            warnings.append(
                f"{len(oracle_report['abstained'])} of {sample.n} sampled rows abstained "
                f"(no calibrated score, an error, or an unusable answer) and carry NO "
                f"label. They stay in `ids` and `weights`, so an estimator can account "
                f"for them, but they are not scored.")

    # A single-class SELECT half cannot rank programs, so it is a failed build rather
    # than a weak one. Checked on SELECT only: CERT is never scored during refinement.
    balance = class_balance(select.ids, labels)
    problem = zero_class_message(balance)
    if problem and labels is not None:
        message = f"the SELECT half is unusable: {problem}"
        if not args.allow_single_class:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "failure.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({
                    "reason_code": "single_class_select",
                    "message": message,
                    "class_balance": balance,
                    "population": sample.N,
                    "select_n": select.n,
                    "rate": args.rate,
                }, handle, indent=2)
            raise SystemExit(f"[build_valset] {message}\n"
                             f"  (pass --allow-single-class to write it anyway)")
        warnings.append(message)
        print(f"[build_valset] WARNING: {message}", file=sys.stderr)

    provenance = {
        "label_source": args.label_source,
        "corpus": os.path.abspath(args.corpus),
        "id_col": args.id_col,
        "text_col": text_cols[0] if text_cols else None,
        "text_cols": text_cols,
        "image_col": args.image_col,
        "image_dir": os.path.abspath(args.image_dir) if args.image_dir else None,
        "class_balance": balance,
        "argv": sys.argv[1:],
        "warnings": warnings,
    }
    if args.label_source == "gt":
        provenance["gt_file"] = os.path.abspath(args.gt_file)
        provenance["gt_match_col"] = args.gt_match_col
    if args.label_source == "oracle":
        provenance["oracle"] = oracle_report
        provenance["call_site"] = site_info

    # --- write -----------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    written: dict[str, str] = {}
    for name, part in (("select", select), ("cert", cert)):
        if not part.n:
            continue
        path = os.path.join(args.out, f"{name}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(val_payload(part, query=args.query, attr=args.attr, split=name,
                                  labels=labels, provenance=provenance), handle, indent=2)
        written[name] = path

    overlap = set(select.ids) & set(cert.ids)
    assert not overlap, f"SELECT and CERT overlap on {len(overlap)} ids -- split is broken"
    manifest = {
        "query": args.query, "attr": args.attr, "seed": args.seed,
        "design": design_block(sample),
        "select_ids": list(select.ids), "cert_ids": list(cert.ids),
        "disjoint": True, "files": written, "provenance": provenance,
        "cert_read_log": [],   # certify.py appends here; any read of CERT leaves a trace
    }
    manifest_path = os.path.join(args.out, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    rate_note = f", rate={args.rate}" if args.rate is not None else ""
    print(f"[build_valset] {args.method}: N={sample.N} -> select {select.n} + cert {cert.n} "
          f"(disjoint), seed={args.seed}{rate_note}")
    if labels is not None:
        print("[build_valset] SELECT class balance: " +
              ", ".join(f"{k}={v}" for k, v in sorted(balance.items())))
    if site_info:
        print(f"[build_valset] question ({site_info.get('kind', 'nl')} @ "
              f"{site_info.get('source')}): {question[:100]}"
              f"{'...' if len(question) > 100 else ''}")
    if sample.method == "stratified":
        # Print the realized per-stratum RATE, not just the counts: a tilt that
        # collapsed to a near-uniform allocation is the failure mode this design has,
        # and it is invisible in counts alone when the strata are equal-sized.
        table = sample.meta.get("strata", {})
        print("[build_valset] strata: " +
              ", ".join(f"{h}(N={v['N_h']},n={v['n_h']},rate={v['n_h'] / v['N_h']:.0%})"
                        for h, v in sorted(table.items())))
    if sample.method == "pareto_pps":
        print(f"[build_valset] pps: epsilon={args.epsilon} w_max={sample.meta['w_max']:.1f} "
              f"certainty units={sample.meta['n_certainty']}")
    print(f"[build_valset] wrote {', '.join(written.values())} and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
