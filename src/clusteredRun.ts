/** Public, display-scoped metadata for a selection computed over the full corpus. */
export type ClusteredRun = {
  schema_version: 1;
  selector: "balanced_two_means_dpp";
  source_rows: number;
  selected_rows: number;
  display_rows: number;
  display_selected_rows: number;
  group_size: number;
  keep_per_group: number;
  quality_weight: number;
  seed: number;
  memberships_sha256: string;
  display_policy: "intersection_with_existing_display_population";
};

export function parseClusteredRun(value: unknown): ClusteredRun {
  if (!value || typeof value !== "object") throw new Error("Invalid clustered run metadata");
  const row = value as Record<string, unknown>;
  for (const key of ["source_rows", "selected_rows", "display_rows", "display_selected_rows", "group_size", "keep_per_group", "seed"]) {
    if (typeof row[key] !== "number" || !Number.isSafeInteger(row[key]) || row[key] < 0) {
      throw new Error(`Invalid clustered run ${key}`);
    }
  }
  const run = row as unknown as ClusteredRun;
  if (run.schema_version !== 1 || run.selector !== "balanced_two_means_dpp" ||
      run.display_policy !== "intersection_with_existing_display_population" ||
      run.group_size < 1 || run.keep_per_group < 1 || run.keep_per_group > run.group_size ||
      run.selected_rows < 1 || run.display_rows > run.source_rows ||
      run.display_selected_rows > Math.min(run.selected_rows, run.display_rows) ||
      BigInt(run.selected_rows) !== BigInt(run.source_rows) * BigInt(run.keep_per_group) / BigInt(run.group_size) ||
      typeof run.quality_weight !== "number" || !Number.isFinite(run.quality_weight) ||
      run.quality_weight < 0 || run.quality_weight > 50 ||
      typeof run.memberships_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(run.memberships_sha256)) {
    throw new Error("Inconsistent clustered run metadata");
  }
  return run;
}

export async function verifyClusteredMemberships(bytes: Uint8Array, expectedHash: string): Promise<void> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes)));
  const actual = Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
  if (actual !== expectedHash) throw new Error("Clustered membership checksum mismatch");
}
