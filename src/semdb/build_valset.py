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
    --strata-by column:<name>   an existing categorical column
    --importance-by query-similarity   TF-IDF cosine against the query text
    --importance-by column:<name>      an existing numeric column

Example (uniform, labels from SemBench ground truth, for plumbing only)::

    python3 src/semdb/build_valset.py \
      --corpus /path/lizzy_caplan_text_data.csv --id-col row_id --text-col text \
      --query q3a --attr is_comedy \
      --method uniform --n 60 --cert-n 40 --seed 7 \
      --label-source gt --gt-file /path/Q3a.json --gt-match-col title \
      --out src/semdb/runs/_val/mmqa-q3a
"""
from __future__ import annotations

import argparse
import csv
import json
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

def read_corpus(path: str, id_col: str, text_col: str | None) -> list[dict[str, str]]:
    """Read a CSV into row dicts, validating that the named columns exist."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"corpus {path} is empty")
    header = rows[0].keys()
    for col in filter(None, (id_col, text_col)):
        if col not in header:
            raise SystemExit(f"column {col!r} not in {path}; available: {', '.join(header)}")
    seen: set[str] = set()
    for row in rows:
        rid = str(row[id_col])
        if rid in seen:
            raise SystemExit(f"duplicate id {rid!r} in {path} -- ids must be unique")
        seen.add(rid)
    return rows


def _texts(rows: Sequence[dict[str, str]], text_col: str | None) -> list[str]:
    return [str(r.get(text_col, "") or "") for r in rows] if text_col else [""] * len(rows)


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


def build_strata(rows: Sequence[dict[str, str]], texts: Sequence[str],
                 spec: str, k: int, seed: int) -> list[str]:
    if spec == "length":
        return strata_by_length(texts, k)
    if spec == "kmeans":
        return strata_by_kmeans(texts, k, seed)
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


def build_scores(rows: Sequence[dict[str, str]], texts: Sequence[str],
                 spec: str, query_text: str) -> list[float]:
    if spec == "query-similarity":
        return scores_by_query_similarity(texts, query_text)
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


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

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
    ap.add_argument("--text-col", help="Unstructured text column (strata / similarity).")
    ap.add_argument("--query", required=True, help="Query id, e.g. q3a.")
    ap.add_argument("--attr", required=True, help="Attribute name the labels carry.")
    ap.add_argument("--query-nl", default="", help="Natural-language query text.")

    ap.add_argument("--method", default="uniform",
                    choices=["uniform", "stratified", "importance"])
    ap.add_argument("--n", type=int, required=True, help="SELECT size.")
    ap.add_argument("--cert-n", type=int, default=0, help="CERT size (0 = no cert split).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--strata-by", default="kmeans", help="length | kmeans | column:<name>")
    ap.add_argument("--strata-k", type=int, default=4)
    ap.add_argument("--min-per-stratum", type=int, default=2)
    ap.add_argument("--importance-by", default="query-similarity",
                    help="query-similarity | column:<name>")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="Uniform mixing weight for the importance proposal; "
                         "bounds the weights at N/(n*epsilon).")

    ap.add_argument("--label-source", default="gt", choices=["gt", "none", "oracle"])
    ap.add_argument("--gt-file", help="SemBench ground-truth JSON (--label-source gt).")
    ap.add_argument("--gt-match-col", help="Corpus column the ground truth lists.")
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args(argv)

    rows = read_corpus(args.corpus, args.id_col, args.text_col)
    ids = [str(r[args.id_col]) for r in rows]
    texts = _texts(rows, args.text_col)
    total = args.n + args.cert_n
    if total > len(ids):
        raise SystemExit(f"--n + --cert-n = {total} exceeds the corpus size {len(ids)}")

    # --- draw ------------------------------------------------------------
    # Design errors (an infeasible allocation, a stratification finer than the
    # sample) are user errors, not bugs: surface the actionable message without a
    # traceback.
    try:
        if args.method == "uniform":
            sample = sampling.sample_uniform(ids, total, args.seed)
        elif args.method == "stratified":
            strata = build_strata(rows, texts, args.strata_by, args.strata_k, args.seed)
            # A stratum must be big enough to put at least one row on each side of
            # the SELECT/CERT split; anything thinner gets pooled.
            floor = max(args.min_per_stratum, 2 if args.cert_n else 1)
            merged, report = sampling.collapse_small_strata(dict(zip(ids, strata)), floor)
            if report["merged"]:
                print(f"[build_valset] pooled {len(report['merged'])} stratum/strata with "
                      f"< {floor} rows ({report['pooled_size']} rows) into "
                      f"{report['pooled_into']!r}", file=sys.stderr)
            sample = sampling.sample_stratified(
                ids, merged, total, args.seed, min_per_stratum=args.min_per_stratum)
        else:
            scores = build_scores(rows, texts, args.importance_by, args.query_nl)
            sample = sampling.sample_pareto_pps(ids, scores, total, args.seed,
                                                epsilon=args.epsilon)
        sample.check()

        select, cert = sampling.split_sample(sample, args.cert_n, args.seed + 1)
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
        raise SystemExit(
            "--label-source oracle is not implemented yet (it lands with "
            "oracle_label.py). Use --label-source none to emit the ids now and "
            "label them separately, or --label-source gt for a plumbing fixture.")

    provenance = {
        "label_source": args.label_source,
        "corpus": os.path.abspath(args.corpus),
        "id_col": args.id_col, "text_col": args.text_col,
        "argv": sys.argv[1:],
        "warnings": warnings,
    }
    if args.label_source == "gt":
        provenance["gt_file"] = os.path.abspath(args.gt_file)
        provenance["gt_match_col"] = args.gt_match_col

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

    pos = sum(1 for i in select.ids if labels and labels.get(i) == TRUE)
    print(f"[build_valset] {args.method}: N={sample.N} -> select {select.n} + cert {cert.n} "
          f"(disjoint), seed={args.seed}")
    if labels:
        print(f"[build_valset] SELECT positives: {pos}/{select.n} "
              f"({pos / select.n:.1%})" if select.n else "")
    if sample.method == "stratified":
        table = sample.meta.get("strata", {})
        print("[build_valset] strata: " +
              ", ".join(f"{h}(N={v['N_h']},n={v['n_h']})" for h, v in table.items()))
    if sample.method == "pareto_pps":
        print(f"[build_valset] pps: epsilon={args.epsilon} w_max={sample.meta['w_max']:.1f} "
              f"certainty units={sample.meta['n_certainty']}")
    print(f"[build_valset] wrote {', '.join(written.values())} and {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
