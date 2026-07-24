# SemBench Table-Input Normalization Scheme

- **Date:** 2026-07-22
- **Status:** Design (proposed)
- **Goal:** Convert any SemBench workload's raw, heterogeneous table files into ONE
  canonical, self-describing, path-resolved **bundle**, so the orchestrator / extractor
  / evaluator read a single uniform format — and a pre-run **validation gate** fails fast
  with a clear message instead of crashing mid-pipeline.

## Motivation — the failure class this eliminates
Every crash this session shared one root cause: downstream code reads RAW files and
*guesses* (table→file, path, prefix, format, scale). Concretely:

| Failure mode | Example |
|---|---|
| table name ≠ file name | cars `car_mm` → `image_car_data_9836.csv` |
| parquet vs CSV | ecomm `styles_details.parquet`; `IMAGES` = an image *directory* |
| scale-factor suffix | cars `Q1_157376.csv`, medical `Q1_20.csv` |
| image path base varies | mmqa `sf/images/<file>`; cars/medical repo-relative; animals absolute |
| **stale absolute paths** | animals `ImagePath = /home/jiale/SemBench/...` (another machine) |
| GT format / variant | mmqa JSON vs others CSV; scale/sample variants |
| labels_from prefix | designer wrote `mmqa.ap_warrior.Track` (SQL form) |
| flat vs sf data dir | medical `data/` (flat) vs cars `data/sf_9836/` |
| audio tables | `audio_mm` — content sometimes in a col named `image` |

Each is a special case scattered across `orchestrator.mjs` (tableDir, materialize,
resolveGroundTruth, imageDir/imageBase, resolveLabelsFrom) and `semextract`/`semvision`
(`resolve_image_path`). The fix: centralize all of it in ONE normalization step.

## Architecture

`benchmarks.mjs` stays the **declarative config** (the per-scenario rules). A new
`normalize.py` (extends today's `materialize.py`) is the **engine** that APPLIES those
rules once per `(benchmark, scale_factor)` and writes a canonical bundle:

```
runs/_normalized/<bench>_sf<sf>/
├── manifest.json            # bench, sf, sembench_root, checks, table list, gt dir
├── registry.json            # table -> {file, modality, key, unstructured_col, n_rows}
├── tables/<table>.csv       # canonical CSV per table (parquet converted)
│                            #   header always has: __id, and for images __image_path (ABS, verified)
├── values/<table>__<col>.json   # distinct values (value spaces for labels_from / enums)
└── ground_truth/<Qid>.<json|csv> # the resolved GT for THIS sf (right variant)
```

Downstream reads **only** the bundle:
- orchestrator planning: `registry.json` (table→file/modality/key, all absolute) — no
  `tableDir`/`materialize`/`imageBase` logic.
- extraction: `tables/<table>.csv` already has `__image_path` resolved → `semvision`/
  `semextract` drop `resolve_image_path` guessing.
- `labels_from`: reads `values/<table>__<col>.json` (canonical table names, prefix
  stripped) — the parser bug class disappears.
- evaluation: `ground_truth/` is the single resolved GT.

## Canonicalization rules (one per failure mode)

`normalize.py --benchmark B --data-dir D --sf S --sembench-root R --out OUT`:

1. **table→file:** from `benchmarks.tables[t].file` with `{sf}` filled; **parquet→CSV**
   (pandas). Unknown table → error listed in the report.
2. **`__id`:** copy the config `key` column to a stable `__id` (keep original too).
3. **`__image_path` (image tables only):** resolve the raw path column by `imageBase`:
   - `sf` → `join(D, imageRoot, basename(col))`
   - `root` → `join(R, col)` (repo-relative)
   - `absolute` → `col` **with STALE-ROOT REMAP**: if `col` is absolute but missing,
     re-root its `/files/...` (or `source_data/...`) tail under `R` and retry.
   Write the resolved absolute path; count/flag missing.
4. **ground truth:** pick the variant for `S` (`Q<id>_<S>.csv` else `Q<id>.csv`; or
   `q<id>.json`); copy into `ground_truth/`. Record format per query.
5. **value spaces:** precompute distinct values for every `key`, `enum`, and
   join-referenced column → `values/<table>__<col>.json` (drives `labels_from`).
6. **audio:** keep `modality:"audio"` in the registry; the orchestrator already skips it.

## Validation gate (fail fast, before any agent/extraction/vLLM)

`normalize.py --check` asserts and prints a summary; **exit non-zero** on hard failures:
- every table file exists and is non-empty;
- for image tables, ≥ `min_resolve` (default 90%) of `__image_path` exist on disk;
- GT present for each discovered query id;
- every `labels_from` referenced column has a non-empty value space.
The orchestrator runs `normalize --check` as step 0; a failure prints exactly what's
missing (path, table, GT) and stops — no more mid-run `FileNotFoundError`/`no values`.

## Integration changes (what gets deleted)
- `orchestrator.mjs`: replace `tableDir`/inline materialize call/`resolveGroundTruth`/
  `imageBase` derivation/`resolveLabelsFrom` file-reading with reads from `registry.json`
  / `values/` / `ground_truth/`. `resolveLabelsFrom` becomes a lookup, not a CSV parse.
- `semextract.py`/`semvision.py`: drivers read `__image_path` (pre-resolved) → remove
  `resolve_image_path` heuristics from the hot path (keep as a fallback only).
- `benchmarks.mjs`: unchanged in spirit; `imageBase`/`parquet`/`tables` now consumed by
  `normalize.py` rather than scattered in the orchestrator.

## Rollout
- **Phase 1 (non-breaking):** build `normalize.py` + `--check`; run it for all 6
  benchmarks × their scale factors; verify every bundle passes the gate (surfaces the
  animals stale-path and medical missing-x-ray issues explicitly, up front).
- **Phase 2:** point orchestrator/extractor/evaluator at the bundle; delete the ad-hoc
  path/format/prefix logic. Regression: mmqa q2a/q7 + ecomm q2 produce identical results
  via the bundle.

## Non-goals
- Downloading missing raw media (animals/medical images not on this host) — the gate
  *reports* them; fetching is separate.
- Changing metrics or extraction tiers.

## Open questions
- Bundle location: under `runs/_normalized/` (repo, keeps SemBench pristine) — chosen.
- Cache invalidation: re-normalize on `--force` or when the source mtime is newer.
