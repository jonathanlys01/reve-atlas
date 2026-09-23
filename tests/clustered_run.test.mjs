import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { parseClusteredRun, verifyClusteredMemberships } from "../src/clusteredRun.ts";

const valid = {
  schema_version: 1, selector: "balanced_two_means_dpp", source_rows: 21987613,
  selected_rows: 2198761, display_rows: 1800000, display_selected_rows: 190000,
  group_size: 1000, keep_per_group: 100, quality_weight: 0, seed: 29,
  memberships_sha256: "a".repeat(64), display_policy: "intersection_with_existing_display_population",
};
test("preserves global and display counts separately", () => {
  assert.deepEqual(parseClusteredRun(valid), valid);
  assert.equal(parseClusteredRun({...valid, display_selected_rows: 0}).display_selected_rows, 0);
});
for (const change of [
  {selected_rows: 2198762}, {display_selected_rows: 2198761}, {group_size: 0},
  {source_rows: -1}, {keep_per_group: 1001}, {quality_weight: Infinity},
  {memberships_sha256: "not-a-hash"}, {schema_version: 2}, {seed: 0.5},
  {display_policy: "all_selected_visible"}, {selector: "ranking"},
]) {
  test(`rejects inconsistent metadata ${JSON.stringify(change)}`, () => {
    assert.throws(() => parseClusteredRun({...valid, ...change}));
  });
}
test("verifies exact overlay bytes before using them", async () => {
  const bytes = new TextEncoder().encode("actual memberships");
  const hash = createHash("sha256").update(bytes).digest("hex");
  await verifyClusteredMemberships(bytes, hash);
  await assert.rejects(verifyClusteredMemberships(bytes, "0".repeat(64)), /checksum/);
});
