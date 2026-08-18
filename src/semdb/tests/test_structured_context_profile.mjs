import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import {
  buildPlannerTableProfile,
  lintPlanAgainstTableProfile,
  parseCsvSample,
} from "../agent-runtime/context-profile.mjs";

const parsed = parseCsvSample('id,title,text\n1,"A, Film","line one\nline two"\n2,B,plain\n', 4);
assert.equal(parsed.rows.length, 2);
assert.equal(parsed.rows[0].title, "A, Film");
assert.equal(parsed.rows[0].text, "line one\nline two");
assert.equal(parsed.complete, true);

const root = await mkdtemp(resolve(tmpdir(), "semdb-context-profile-"));
const textPath = resolve(root, "text.csv");
const structuredPath = resolve(root, "structured.csv");
const textRows = Array.from({ length: 300 }, (_, index) => (
  `${index + 1},Unrelated title ${index + 1},description ${index + 1}`
));
textRows.push('301,"The Bad News Bears (film)","description with\na quoted newline"');
await writeFile(textPath, `id,title,text\n${textRows.join("\n")}\n`);
await writeFile(structuredPath, "id,Title,Role\n7,The Bad News Bears,Bob Whitewood\n");

const profile = await buildPlannerTableProfile([
  { table: "text_data", path: textPath },
  { table: "movies", path: structuredPath },
]);
assert.equal(profile.profile_version, "2.0");
assert.equal(profile.scan_mode, "full_file_streaming");
assert.equal(profile.tables[0].rows_scanned, 301);
assert.equal(profile.tables[0].full_scan_complete, true);
assert.equal(profile.tables[0].columns.title.non_empty_rows, 301);
assert.equal(profile.tables[0].columns.title.distinct_values, 301);
assert.equal(profile.same_name_join_candidates.length, 1);
assert.equal(profile.same_name_join_candidates[0].full_scan_complete, true);
assert.equal(profile.same_name_join_candidates[0].value_sets_complete, true);
assert.equal(profile.same_name_join_candidates[0].exact_overlap_count, 0);
assert.equal(profile.same_name_join_candidates[0].normalized_overlap_count, 0);
assert.deepEqual(
  profile.same_name_join_candidates[0].containment_near_match_examples[0],
  ["The Bad News Bears (film)", "The Bad News Bears"],
);

const findings = lintPlanAgainstTableProfile({
  relational_plan: [{
    operation: "inner join",
    condition: "normalize(text_data.title) == normalize(movies.Title)",
  }],
}, profile);
assert.equal(findings.length, 1);
assert.match(findings[0], /strict equality join is therefore empty/);
assert.equal(lintPlanAgainstTableProfile({
  relational_plan: [{
    operation: "join",
    condition: "canonical title containment between text_data.title and movies.Title",
  }],
}, profile).length, 0);

const boundedProfile = await buildPlannerTableProfile([
  { table: "text_data", path: textPath },
  { table: "movies", path: structuredPath },
], { maxDistinctValues: 2 });
assert.equal(boundedProfile.tables[0].rows_scanned, 301,
  "the distinct-memory cap must not truncate the file scan");
assert.equal(boundedProfile.tables[0].columns.title.distinct_tracking_complete, false);
assert.equal(boundedProfile.same_name_join_candidates[0].value_sets_complete, false);
assert.equal(boundedProfile.same_name_join_candidates[0].exact_overlap_count, null);
assert.equal(lintPlanAgainstTableProfile({
  relational_plan: [{
    operation: "inner join",
    condition: "normalize(text_data.title) == normalize(movies.Title)",
  }],
}, boundedProfile).length, 0, "a truncated value set must never prove an empty join");

console.log("test_structured_context_profile OK");
