/** FROZEN plan fixtures, transcribed from the runs that exhibited each defect.
 *
 *  These were originally read straight out of `runs/`, which was wrong: those artifacts
 *  are regenerated on every run, so a later run silently fixed the shape a test existed
 *  to detect and the test started failing for the best possible reason. A regression
 *  fixture has to be immutable. Only the fields `lintImagePlan` reads are kept.
 */

/** mmqa q2a, first PGO run: the image is collapsed to ONE label
 *  (`racetrack_logo_name(image, list[string]) -> string`) and compared with `==` in the
 *  solver. The ground truth is one wordmark that must match TWO Track values, so this
 *  shape caps recall at 3/5 before any image is read. */
export const Q2A_LABEL_COLLAPSE = {
  modality: "image",
  compilability: { class: "bounded_approximation" },
  semantic_sites: [{
    site_id: "site_0",
    sampling_unit: "pair",
    output_type: "boolean",
    inputs: [
      { table: "ap_warrior", column: "Track", type: "string" },
      { table: "images", column: "image_filepath", type: "ImagePatch" },
    ],
  }],
  helper_dag: [{
    helper_id: "helper_0",
    name: "racetrack_logo_name",
    args: [
      { name: "image", type: "ImagePatch" },
      { name: "track_names", type: "list[string]" },
    ],
    return_type: "string",
    depends_on: [],
    primitive_steps: [{ primitive: "best_ocr_match_detail", source: "", inputs: [], output_type: "tuple" }],
    confidence_signal: null,
  }],
};

/** mmqa q2a, later run: the pair predicate is correct (evidence + per-pair bool) but it
 *  decides membership with `classify_multi` over the Track value space, so every admitted
 *  image matched every track — 65 rows out, 0 correct. */
export const Q2A_MULTILABEL_FILTER = {
  modality: "image",
  compilability: { class: "bounded_approximation" },
  semantic_sites: [{
    site_id: "site_0", sampling_unit: "pair", output_type: "boolean",
    inputs: [
      { table: "ap_warrior", column: "Track", type: "string" },
      { table: "images", column: "image_filepath", type: "ImagePatch" },
    ],
  }],
  helper_dag: [
    {
      helper_id: "helper_0", name: "racetrack_logo_evidence",
      args: [{ name: "image", type: "ImagePatch" }, { name: "track_options", type: "list[string]" }],
      return_type: "RacetrackLogoEvidence", depends_on: [],
      primitive_steps: [
        { primitive: "topk_text", source: "", inputs: [], output_type: "list" },
        { primitive: "classify_multi", source: "", inputs: [], output_type: "tuple" },
        { primitive: "best_ocr_match_detail", source: "", inputs: [], output_type: "tuple" },
      ],
      confidence_signal: null,
    },
    {
      helper_id: "helper_1", name: "racetrack_logo_for_track",
      args: [{ name: "evidence", type: "RacetrackLogoEvidence" }, { name: "track", type: "string" }],
      return_type: "boolean", depends_on: ["helper_0"],
      primitive_steps: [], confidence_signal: null,
    },
  ],
};

/** mmqa q7: a filter site bound to `classify_detail` (argmax, cannot return "none") with
 *  a `confidence_signal` that names a source but no threshold. All 200 images were
 *  assigned an airline; tp=0, fp=16. */
export const Q7_ARGMAX_NO_THRESHOLD = {
  modality: "image",
  compilability: { class: "bounded_approximation" },
  semantic_sites: [{
    site_id: "site_0", sampling_unit: "pair", output_type: "bool",
    inputs: [
      { table: "tampa_international_airport", column: "Airlines", type: "string" },
      { table: "images", column: "ref", type: "ImagePatch" },
    ],
  }],
  helper_dag: [
    {
      helper_id: "helper_0", name: "infer_airline_logo_name",
      args: [{ name: "image", type: "ImagePatch" }, { name: "airline_options", type: "list[string]" }],
      return_type: "string", depends_on: [],
      primitive_steps: [
        { primitive: "best_ocr_match_detail", source: "", inputs: [], output_type: "tuple" },
        { primitive: "classify_detail", source: "", inputs: [], output_type: "tuple" },
      ],
      confidence_signal: { source: "classify_match.score", range: [0, 1], higher_is_more_confident: true },
    },
    {
      helper_id: "helper_1", name: "is_european_airline_logo",
      args: [{ name: "image", type: "ImagePatch" }],
      return_type: "boolean", depends_on: [],
      primitive_steps: [{ primitive: "verify_detail", source: "", inputs: [], output_type: "tuple" }],
      confidence_signal: { source: "europe_evidence.score", range: [0, 1], higher_is_more_confident: true },
    },
  ],
};

export const clonePlan = (plan) => JSON.parse(JSON.stringify(plan));
