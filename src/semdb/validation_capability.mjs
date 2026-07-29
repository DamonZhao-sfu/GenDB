/**
 * Classify whether SemDB can construct leakage-safe whole-query validation.
 *
 * This is separate from semantic-plan compilability: an offline solver may exist
 * while joint Oracle feedback for all call sites does not.
 */
export function classifyValidationCapability(validationPlan, options = {}) {
  const candidate = validationPlan?.candidate || {};
  const unit = candidate.unit || "unknown";
  const sites = validationPlan?.sites || [];
  const common = {
    version: 1,
    stage: "validation_frame",
    benchmark: options.benchmark || validationPlan?.benchmark || "",
    query: options.query || validationPlan?.query || "",
    candidate_unit: unit,
    site_count: sites.length,
  };
  if (options.audioTables?.length) {
    return {
      ...common,
      class: "not_compilable",
      reason_code: "unsupported_audio_runtime",
      reason: `query references unsupported audio table(s): ${options.audioTables.join(", ")}`,
      obligations: ["Provide a repository-local audio primitive and runtime adapter."],
    };
  }
  if (unit === "row" || unit === "pair") {
    const composition = candidate.composition?.kind || null;
    return {
      ...common,
      class: "executable",
      reason_code: composition
        ? `${unit}_${composition}_validation_frame`
        : `${unit}_validation_frame`,
      reason: composition
        ? `SemDB has a leakage-safe ${unit} frame and typed ${composition} Oracle label.`
        : `SemDB has a leakage-safe ${unit} validation-frame path.`,
      obligations: [],
    };
  }
  if (unit === "tuple" || unit === "multi_site") {
    return {
      ...common,
      class: "not_compilable",
      reason_code: "joint_multi_site_oracle_not_implemented",
      reason:
        `automatic whole-query validation requires one joint ${unit} label that `
        + `composes ${sites.length} semantic call sites; the current Oracle protocol `
        + `labels one call site at a time and cannot produce that feedback without `
        + `changing the optimization objective`,
      obligations: [
        "Materialize the typed joint candidate domain after ordinary SQL predicates.",
        "Evaluate all semantic sites with their role-bearing inputs.",
        "Compose site outputs according to the query's boolean/grouped/aggregate plan.",
        "Emit one SELECT/CERT-isolated label and trace value per joint candidate key.",
      ],
    };
  }
  return {
    ...common,
    class: "not_compilable",
    reason_code: "unknown_validation_unit",
    reason: `validation candidate unit ${JSON.stringify(unit)} has no runtime contract`,
    obligations: ["Add a typed validation-frame and trace-key contract for this unit."],
  };
}
