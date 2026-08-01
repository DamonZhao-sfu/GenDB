// benchmarks.mjs — per-SemBench-scenario descriptors that make the orchestrator
// scenario-agnostic. The orchestrator was originally hardcoded to mmqa (table
// qualifier `mmqa.`, table-name == <name>.csv, GT as JSON, images in <dataDir>/images);
// every one of those assumptions varies per scenario. This config captures the
// differences so the SQL/table/modality/GT logic is data-driven.
//
// Per table: { file, modality, col, key }
//   file     — CSV/parquet basename (use "{sf}" for scale-suffixed names); null = derive
//   modality — "image" | "text" | "audio" | "structured"  (audio is UNSUPPORTED → skipped)
//   col      — the unstructured column the AI operator reads (image path / text / ref)
//   key      — the join/id key column (null = single-table filter/aggregate, no join key)
//
// modality is authoritative here — do NOT infer it from the table name (audio tables
// sometimes expose their content through a column literally named `image`).

export const BENCHMARKS = {
  mmqa: {
    prefix: "mmqa",
    queryDir: "query/bigquery",
    dataLayout: "sf",                 // tables under data/sf_<sf>/
    gtDir: "raw_results/ground_truth",
    gtFormat: "json",                 // {nl_question, ground_truth:[...]}
    imageRoot: "images",              // subdir under dataDir
    imageBase: "sf",                  // images under <dataDir>/images; col = bare filename
    identityFiles: true,              // SQL table name == <name>.csv
    imageTables: ["images", "thalamusdb_images"],
  },

  movie: {                            // text-only; cleanest scenario
    prefix: "movie",
    queryDir: "query/bigquery",
    dataLayout: "sf",
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: null,
    tables: {
      reviews: { file: "Reviews.csv", modality: "text", col: "reviewText", key: "reviewId" },
      Movies:  { file: "Movies.csv",  modality: "structured", key: "id" },
    },
  },

  cars: {
    prefix: "cars_dataset",
    queryDir: "query/bigquery",
    dataLayout: "sf",
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: "all_car_images",      // image_path column holds a repo-relative path
    imageBase: "root",                // image_path = "files/cars/data/all_car_images/…" (from sembench root)
    tables: {
      cars:       { file: "car_data_{sf}.csv",             modality: "structured", key: "car_id" },
      complaints: { file: "text_complaints_data_{sf}.csv", modality: "text",  col: "summary",    key: "car_id" },
      car_mm:     { file: "image_car_data_{sf}.csv",       modality: "image", col: "image_path", key: "car_id" },
      audio_mm:   { file: "audio_car_data_{sf}.csv",       modality: "audio", col: "audio_path", key: "car_id" },
    },
  },

  medical: {
    prefix: "medical_dataset",
    queryDir: "query/bigquery",
    dataLayout: "flat",               // tables directly under data/
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: null,                  // image_path column holds a repo-relative path
    imageBase: "root",                // image_path resolved from sembench root
    tables: {
      patients:       { file: "patient_data.csv",        modality: "structured", key: "patient_id" },
      symptoms_texts: { file: "text_symptoms_data.csv",  modality: "text",  col: "symptoms",   key: "patient_id" },
      x_ray_mm:       { file: "image_x_ray_data.csv",    modality: "image", col: "image_path", key: "patient_id" },
      skin_cancer_mm: { file: "image_skin_data.csv",     modality: "image", col: "image_path", key: "patient_id" },
      audio_mm:       { file: "audio_lung_data.csv",     modality: "audio", col: "path",       key: "patient_id" },
    },
  },

  animals: {                          // single-table filters/aggregates; no structured join side
    prefix: "animals_dataset",
    queryDir: "query/bigquery",
    dataLayout: "sf",
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: null,                  // ImagePath/AudioPath are absolute source_data paths
    imageBase: "absolute",            // path column is already absolute
    tables: {
      image_data_mm: { file: "image_data.csv", modality: "image", col: "ImagePath", key: null },
      audio_data_mm: { file: "audio_data.csv", modality: "audio", col: "AudioPath", key: null },
    },
  },

  supg: {                             // SUPG/BARGAIN approximate-selection datasets
    // NOT a SemBench scenario — a SemBench-SHAPED tree built by
    // data/supg/build_supg_scenario.py from the SUPG artifact
    // (github.com/stanford-futuredata/supg). SUPG ships only (id, label,
    // proxy_score), so each dataset lands in one of two corpus modes:
    //
    //   image/text  — the real content was rehydrated (only `imagenet`, via
    //                 fetch_imagenet.py); AI.IF runs a real VLM over it.
    //   proxy       — the content was never published. The corpus is `id,proxy_score`
    //                 and the label is reachable ONLY through the metered
    //                 supg_oracle (ORACLE LIMIT distinct ids). This is SUPG's own
    //                 protocol, not a degraded fallback.
    //
    // `proxy` tables carry no unstructured content, so they never enter the
    // extraction path; `col` names the free numeric evidence column instead.
    prefix: "supg",
    queryDir: "query/bigquery",
    dataLayout: "sf",
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: "images",
    imageBase: "sf",                  // bare filename under <dataDir>/images
    oracleDir: "_oracle",             // labels backing the metered oracle; sibling of
                                      // data/, never readable by generated code
    tables: {
      imagenet:     { file: "imagenet.csv",     modality: "image", col: "image_filename", key: "id" },
      night_street: { file: "night_street.csv", modality: "proxy", col: "proxy_score",    key: "id" },
      ontonotes:    { file: "ontonotes.csv",    modality: "proxy", col: "proxy_score",    key: "id" },
      tacred:       { file: "tacred.csv",       modality: "proxy", col: "proxy_score",    key: "id" },
    },
  },

  ecomm: {                            // HARDEST: parquet, queries/dialects/, .ref images, multi-AI-op
    prefix: "fashion_product_images",
    queryDir: "queries/dialects/bigquery",
    dataLayout: "sf",
    gtDir: "raw_results/ground_truth",
    gtFormat: "csv",
    imageRoot: "images",
    imageBase: "sf",
    parquet: true,                    // tables materialized to CSV by materialize.py
    tables: {
      // files below are the MATERIALIZED CSVs (see materialize.py), resolved under
      // the materialized dir, not the raw parquet. IMAGES is an id+filename manifest;
      // the image lives at <dataDir>/images/<filename>.
      STYLES_DETAILS: { file: "styles_details.csv", modality: "text",  col: "productDisplayName", key: "id" },
      IMAGES:         { file: "IMAGES.csv",         modality: "image", col: "filename",           key: "id" },
      IMAGE_MAPPING:  { file: "IMAGES.csv",         modality: "structured", key: "id" },
      styles:         { file: "styles.csv",         modality: "structured", key: "id" },
    },
  },
};

// Scenarios that only involve AI.IF / AI.GENERATE over tables (the shape SemDB
// compiles). Retrieval/entity-matching datasets (lro, company, flickr30k,
// roxford5k, veri) are out of scope for now.
export const SUPPORTED = Object.keys(BENCHMARKS);

/** Resolve a SQL table name to its descriptor for a benchmark (case-insensitive). */
export function tableDesc(bench, sqlName) {
  const b = BENCHMARKS[bench];
  if (!b) return null;
  if (b.identityFiles) {
    const isImg = (b.imageTables || []).some((t) => t.toLowerCase() === sqlName.toLowerCase())
      || /image/i.test(sqlName);
    return { file: `${sqlName}.csv`, modality: isImg ? "image" : "text", col: null, key: null };
  }
  const keys = Object.keys(b.tables || {});
  const hit = keys.find((k) => k.toLowerCase() === sqlName.toLowerCase());
  return hit ? b.tables[hit] : null;
}

/** Fill a "{sf}"-templated filename with the active scale factor. */
export function tableFile(desc, sf) {
  if (!desc || !desc.file) return null;
  return desc.file.replace("{sf}", String(sf));
}
