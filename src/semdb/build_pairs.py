"""Materialize the PAIR frame of a self-join / AI-join query as an ordinary corpus.

A join predicate like ecomm q9's

    AI.IF('...both images display objects of the same category and share the same
           dominant surface color...', p1.ref, p2.ref)

is a function of TWO rows, so a per-row validation set is undefined for it: there is
no "the label of row i". The sampling unit has to be the pair.

Rather than duplicate the whole sampling/labeling stack for pairs, this module writes
the pair frame out as a CSV whose id column is ``"<id1>-<id2>"`` -- which is exactly
the composite key SemBench's own join ground truth uses (Q9.csv lists `1645-2606`).
Everything downstream then works unchanged: `build_valset.py` samples it,
`oracle_label.py` labels it (two images per call), and `evaluate.score_inference`
compares against `trace.rows["<id1>-<id2>"]` without knowing pairs exist.

**Restrict the row set first (`--only-ids`).** This is the difference between a usable
pair frame and an unusable one, not a tuning knob. Measured on ecomm q9: pairing all
250 images gives 31,125 pairs holding 6 positives (0.019%), and a 300-pair sample drew
*zero* of them even under a strong score tilt -- CLIP pair-similarity ranks those
positives at 3, 946, 3386, 4503, 16657 and 17891, only about a 2x lift over random.
But q9's own CTE filters the rows deterministically first
(``baseColour IN (...) AND colour1 = '' AND colour2 = '' AND price < 800``), and only
**18** rows survive -- a frame of **153** pairs that still contains all 6 positives, a
3.9% positive rate. At that size the whole frame is labelable in 153 oracle calls and
sampling is unnecessary. Derive the surviving ids by running the query's non-AI
predicates (DuckDB or pandas over the materialized CSVs) and pass them to `--only-ids`.

Two things make the remaining work tractable:

  * **Unordered pairs.** A symmetric predicate ("are these the same colour?") gives
    the same answer either way round, so only N(N-1)/2 pairs are labeled and both
    orderings are emitted at the end. Halves the oracle bill. `--ordered` turns this
    off for a predicate that is genuinely direction-sensitive.
  * **A CLIP pair-similarity column, computed by matmul.** The frame is far too large
    to label uniformly -- ecomm sf_250 has 31,125 unordered pairs holding 6 positives
    (0.019%), where a uniform draw of 63 pairs expects 0.012 of them. The score column
    lets `build_valset.py --strata-by score-decile` concentrate labels the same way it
    does for per-row queries. One corpus encode plus one N x N matmul, no VLM calls.

CLI::

    python3 build_pairs.py --corpus IMAGES.csv --id-col id --image-col filename \\
      --image-dir /path/images --out pairs_q9.csv [--top 20000]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semextract  # noqa: E402
from vadar.paths import resolve_image_path  # noqa: E402

PAIR_SEP = "-"


def pair_id(a: str, b: str, sep: str = PAIR_SEP) -> str:
    return f"{a}{sep}{b}"


def split_pair_id(key: str, sep: str = PAIR_SEP) -> tuple[str, str]:
    """Inverse of `pair_id`. Splits on the LAST separator so ids containing the
    separator (SemBench product ids do not, but nothing guarantees it) round-trip."""
    left, _, right = key.rpartition(sep)
    if not left:
        raise ValueError(f"{key!r} is not a pair id")
    return left, right


def similarity_matrix(image_paths: list[str], clip_model: str) -> np.ndarray:
    """[N, N] CLIP image-image similarity, on the same [0,1] scale as
    `semvision.img_pair_score`. One encode of the corpus and one matmul -- computing
    it pairwise would be N^2 encoder calls for the identical numbers."""
    from vadar import backend as semvision

    encoder = semvision.get_encoder(clip_model)
    matrix = np.asarray(semvision.embed_corpus(image_paths, encoder, on_error="zero"),
                        dtype=np.float32)
    return np.clip((matrix @ matrix.T + 1.0) / 2.0, 0.0, 1.0)


def text_image_similarity(texts: list[str], image_paths: list[str],
                          clip_model: str) -> np.ndarray:
    """[L, R] CLIP text-image similarity, on the same [0,1] scale as
    `similarity_matrix`. One text encode of the left table plus one image encode of the
    right corpus and a single matmul -- the cross-table analogue of `similarity_matrix`.

    This is the score that makes a cross-table frame samplable. Measured on mmqa
    (`--left ap_warrior/tampa_international_airport --right images`), the five ground
    truth pairs of q7 rank 18, 31, 95, 107 and 187 of 40,000 -- all inside the top
    0.47%, a ~200x lift over uniform. Logo recognition is what CLIP is strongest at, so
    the lift here is far larger than the ~2x the image-image score gets on ecomm q9.
    """
    from vadar import backend as semvision

    encoder = semvision.get_encoder(clip_model)
    imgs = np.asarray(semvision.embed_corpus(image_paths, encoder, on_error="zero"),
                      dtype=np.float32)
    txts = np.vstack([semvision.embed_text(t, encoder) for t in texts])
    return np.clip((txts @ imgs.T + 1.0) / 2.0, 0.0, 1.0)


def cross_pair_rows(left_ids: list[str], left_texts: list[str], right_ids: list[str],
                    right_files: list[str], sims: np.ndarray, *,
                    top: int | None) -> list[dict[str, str]]:
    """The CROSS-TABLE pair frame: every (left row, right image) pair, best first.

    Unlike the self-join frame this is the full L x R cartesian product, not the upper
    triangle: the two sides are different entities, so there is no i<j to dedupe on and
    no ordering to collapse. `text1` carries the left side's text and `file2` the right
    side's image path, which is exactly the shape `oracle_label.make_pair_caller` wants
    (`--pair-image-cols file2 --text-col text1`) to ask a one-text-one-image question.

    `top` restricts the sampling frame, with the same caveat as `pair_rows`.
    """
    scores = sims.ravel()
    order = np.argsort(-scores, kind="stable")
    if top is not None:
        order = order[:top]
    width = len(right_ids)
    rows = []
    for k in order:
        i, j = divmod(int(k), width)
        rows.append({"pair_id": pair_id(left_ids[i], right_ids[j]),
                     "id1": left_ids[i], "id2": right_ids[j],
                     "text1": left_texts[i], "file2": right_files[j],
                     "pair_score": f"{float(scores[k]):.6f}"})
    return rows


def pair_rows(ids: list[str], files: list[str], sims: np.ndarray, *,
              ordered: bool, top: int | None,
              include_diagonal: bool = False,
              exclude_equal: list[str] | None = None) -> list[dict[str, str]]:
    """The pair frame, optionally pruned to the `top` most similar pairs.

    Pruning is a real restriction of the sampling frame, not an optimisation: pairs
    below the cut have inclusion probability zero and the resulting estimates describe
    the pruned frame only. The caller records it so the val file says so.
    """
    n = len(ids)
    iu = np.triu_indices(n, k=0 if include_diagonal else 1)
    scores = sims[iu]
    order = np.argsort(-scores, kind="stable")
    if top is not None:
        order = order[:top]
    rows = []
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        if exclude_equal is not None and exclude_equal[i] == exclude_equal[j]:
            continue
        rows.append({"pair_id": pair_id(ids[i], ids[j]), "id1": ids[i], "id2": ids[j],
                     "file1": files[i], "file2": files[j],
                     "pair_score": f"{float(scores[k]):.6f}"})
        if ordered and i != j:
            rows.append({"pair_id": pair_id(ids[j], ids[i]), "id1": ids[j], "id2": ids[i],
                         "file1": files[j], "file2": files[i],
                         "pair_score": f"{float(scores[k]):.6f}"})
    return rows


def text_similarity_matrix(texts: list[str], clip_model: str) -> np.ndarray:
    """[N,N] similarity for a text-text self join using the configured CLIP encoder."""
    from vadar import backend as semvision

    encoder = semvision.get_encoder(clip_model)
    matrix = np.vstack([semvision.embed_text(text, encoder) for text in texts])
    return np.clip((matrix @ matrix.T + 1.0) / 2.0, 0.0, 1.0)


def text_pair_rows(ids: list[str], texts: list[str], sims: np.ndarray, *,
                   ordered: bool, top: int | None,
                   include_diagonal: bool = False,
                   exclude_equal: list[str] | None = None) -> list[dict[str, str]]:
    """Text analogue of :func:`pair_rows`, retaining ordered input roles."""
    n = len(ids)
    iu = np.triu_indices(n, k=0 if include_diagonal else 1)
    scores = sims[iu]
    order = np.argsort(-scores, kind="stable")
    if top is not None:
        order = order[:top]
    rows = []
    for k in order:
        i, j = int(iu[0][k]), int(iu[1][k])
        if exclude_equal is not None and exclude_equal[i] == exclude_equal[j]:
            continue
        rows.append({"pair_id": pair_id(ids[i], ids[j]),
                     "id1": ids[i], "id2": ids[j],
                     "text1": texts[i], "text2": texts[j],
                     "pair_score": f"{float(scores[k]):.6f}"})
        if ordered and i != j:
            rows.append({"pair_id": pair_id(ids[j], ids[i]),
                         "id1": ids[j], "id2": ids[i],
                         "text1": texts[j], "text2": texts[i],
                         "pair_score": f"{float(scores[k]):.6f}"})
    return rows


def _read_csv(path: str, cols: list[str]) -> list[dict[str, str]]:
    """Read a CSV and assert the named columns exist. Shared by both frame modes."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"cannot read corpus {path}: {exc}") from exc
    if not rows:
        raise SystemExit(f"corpus {path} is empty")
    for col in cols:
        if col and col not in rows[0]:
            raise SystemExit(f"column {col!r} not in {path}; "
                             f"available: {', '.join(rows[0].keys())}")
    return rows


def _keep_only(rows: list[dict[str, str]], id_col: str, only_ids: str | None,
               label: str) -> list[dict[str, str]]:
    if not only_ids:
        return rows
    try:
        with open(only_ids, encoding="utf-8") as handle:
            keep = {ln.strip() for ln in handle if ln.strip()}
    except OSError as exc:
        raise SystemExit(f"cannot read --only-ids {only_ids}: {exc}") from exc
    before = len(rows)
    rows = [r for r in rows if str(r[id_col]) in keep]
    if not rows:
        raise SystemExit(f"--only-ids matched none of the {before} {label} rows")
    print(f"[build_pairs] --only-ids: {len(rows)} of {before} {label} rows kept",
          file=sys.stderr)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True,
                    help="Row corpus CSV. Self-join mode: the image manifest. "
                         "Cross-table mode: the LEFT (structured) table.")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--image-col",
                    help="Image filename column for an image-image self join.")
    ap.add_argument("--image-dir")
    # --- cross-table mode (a join between two DIFFERENT tables) ---
    ap.add_argument("--right",
                    help="RIGHT table CSV (the image manifest). Giving this switches to "
                         "CROSS-TABLE mode: the frame is the full L x R cartesian "
                         "product scored by CLIP text-image similarity, for a query "
                         "whose AI predicate joins a structured table to an image table "
                         "(mmqa q2a/q7). Without it, the self-join mode above applies.")
    ap.add_argument("--right-id-col",
                    help="Right-side id column (cross-table mode). Must be the value "
                         "the ground truth lists -- for mmqa that is the image filename.")
    ap.add_argument("--right-image-col",
                    help="Right-side image filename column (cross-table mode).")
    ap.add_argument("--right-image-dir",
                    help="Directory the right-side image filenames are relative to.")
    ap.add_argument("--text-col",
                    help="Self-join mode: text column on both sides of a text-text pair. "
                         "Cross-table mode: LEFT-side text column shown to the oracle "
                         "and encoded by CLIP for the pair score.")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--only-ids",
                    help="File of row ids (one per line) to pair. Use the rows that "
                         "survive the query's DETERMINISTIC predicates -- see the module "
                         "docstring: on ecomm q9 this is the difference between a "
                         "31,125-pair frame at 0.019%% positive and a 153-pair frame at "
                         "3.9%% holding the same positives.")
    ap.add_argument("--ordered", action="store_true",
                    help="Emit both orderings as separate frame rows. Only for a "
                         "direction-sensitive predicate; doubles the oracle bill.")
    ap.add_argument("--include-diagonal", action="store_true",
                    help="Include (row,row). Required when the SQL self join does not "
                         "exclude equal aliases (EComm q7).")
    ap.add_argument("--exclude-equal-col",
                    help="Apply the ordinary SQL predicate left.col <> right.col "
                         "before semantic sampling.")
    ap.add_argument("--top", type=int,
                    help="Keep only the N most similar pairs. This RESTRICTS the "
                         "sampling frame: pruned pairs get inclusion probability 0 and "
                         "the estimates then describe the pruned frame only.")
    ap.add_argument("--out", required=True, help="Pair corpus CSV to write.")
    args = ap.parse_args(argv)

    cross = bool(args.right)
    if cross:
        missing = [f for f, v in (("--right-id-col", args.right_id_col),
                                  ("--right-image-col", args.right_image_col),
                                  ("--text-col", args.text_col)) if not v]
        if missing:
            raise SystemExit(f"cross-table mode (--right) also needs "
                             f"{', '.join(missing)}")
    elif bool(args.image_col) == bool(args.text_col):
        raise SystemExit("self-join mode needs exactly one of --image-col or "
                         "--text-col (or use --right for a cross-table frame)")

    try:
        source_stat = os.stat(args.corpus)
    except OSError as exc:
        raise SystemExit(f"cannot read corpus {args.corpus}: {exc}") from exc
    cache_spec = {
        "version": 2,
        "corpus": {
            "path": os.path.abspath(args.corpus),
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        },
        "id_col": args.id_col,
        "image_col": args.image_col,
        "text_col": args.text_col,
        "right": os.path.abspath(args.right) if args.right else None,
        "right_id_col": args.right_id_col,
        "right_image_col": args.right_image_col,
        "image_dir": os.path.abspath(args.image_dir) if args.image_dir else None,
        "right_image_dir": (os.path.abspath(args.right_image_dir)
                            if args.right_image_dir else None),
        "clip_model": args.clip_model,
        "ordered": args.ordered,
        "include_diagonal": args.include_diagonal,
        "exclude_equal_col": args.exclude_equal_col,
        "top": args.top,
    }
    if args.right:
        try:
            right_stat = os.stat(args.right)
        except OSError as exc:
            raise SystemExit(f"cannot read corpus {args.right}: {exc}") from exc
        cache_spec["right_source"] = {
            "path": os.path.abspath(args.right),
            "size": right_stat.st_size,
            "mtime_ns": right_stat.st_mtime_ns,
        }
    meta_path = args.out + ".meta.json"
    try:
        with open(meta_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if os.path.exists(args.out) and cached.get("spec") == cache_spec:
            print(f"[build_pairs] inputs unchanged; reusing {args.out}")
            return 0
    except (OSError, ValueError, TypeError):
        pass

    left = _read_csv(args.corpus,
                     [args.id_col] + ([args.text_col] if args.text_col
                                      else [args.image_col])
                     + ([args.exclude_equal_col] if args.exclude_equal_col else []))
    left = _keep_only(left, args.id_col, args.only_ids, "left" if cross else "corpus")
    ids = [str(r[args.id_col]) for r in left]
    if len(set(ids)) != len(ids):
        raise SystemExit("row ids must be unique to form pair ids")
    exclude_equal = ([str(r[args.exclude_equal_col]) for r in left]
                     if args.exclude_equal_col else None)

    if cross:
        right = _read_csv(args.right, [args.right_id_col, args.right_image_col])
        rids = [str(r[args.right_id_col]) for r in right]
        if len(set(rids)) != len(rids):
            raise SystemExit(f"--right-id-col {args.right_id_col!r} is not unique in "
                             f"{args.right}; pair ids would collide")
        rpaths = [resolve_image_path(str(r[args.right_image_col]),
                                                args.right_image_dir or args.image_dir)
                  for r in right]
        texts = [str(r[args.text_col]) for r in left]
        total = len(ids) * len(rids)
        print(f"[build_pairs] cross-table: {len(ids)} x {len(rids)} = {total} pairs"
              f"{f' (keeping the top {args.top})' if args.top else ''}", file=sys.stderr)
        sims = text_image_similarity(texts, rpaths, args.clip_model)
        frame = cross_pair_rows(ids, texts, rids, rpaths, sims, top=args.top)
        fields = ["pair_id", "id1", "id2", "text1", "file2", "pair_score"]
    elif args.image_col:
        files = [str(r[args.image_col]) for r in left]
        paths = [resolve_image_path(f, args.image_dir) for f in files]
        n = len(ids)
        total = (n * n if args.ordered and args.include_diagonal
                 else n * (n - 1) if args.ordered
                 else n * (n - 1) // 2 + (n if args.include_diagonal else 0))
        print(f"[build_pairs] {n} rows -> {total} "
              f"{'ordered' if args.ordered else 'unordered'} image pair(s)"
              f"{f' (keeping the top {args.top})' if args.top else ''}", file=sys.stderr)
        sims = similarity_matrix(paths, args.clip_model)
        frame = pair_rows(ids, paths, sims, ordered=args.ordered, top=args.top,
                          include_diagonal=args.include_diagonal,
                          exclude_equal=exclude_equal)
        fields = ["pair_id", "id1", "id2", "file1", "file2", "pair_score"]
    else:
        texts = [str(r[args.text_col]) for r in left]
        n = len(ids)
        total = (n * n if args.ordered and args.include_diagonal
                 else n * (n - 1) if args.ordered
                 else n * (n - 1) // 2 + (n if args.include_diagonal else 0))
        print(f"[build_pairs] {n} rows -> {total} "
              f"{'ordered' if args.ordered else 'unordered'} text pair(s)"
              f"{f' (keeping the top {args.top})' if args.top else ''}", file=sys.stderr)
        sims = text_similarity_matrix(texts, args.clip_model)
        frame = text_pair_rows(
            ids, texts, sims, ordered=args.ordered, top=args.top,
            include_diagonal=args.include_diagonal,
            exclude_equal=exclude_equal)
        fields = ["pair_id", "id1", "id2", "text1", "text2", "pair_score"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({"spec": cache_spec, "rows": len(frame)}, handle, indent=2)
        handle.write("\n")

    kept = len(frame)
    print(f"[build_pairs] wrote {kept} frame row(s) -> {args.out}")
    if args.top and args.top < total:
        print(f"[build_pairs] WARNING: the frame is PRUNED to the {args.top} most "
              f"similar of {total} pairs. Estimates from a sample of it describe the "
              f"pruned frame, not all pairs.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
