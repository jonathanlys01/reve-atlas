#!/usr/bin/env python
"""Gate 2 regression: compare membership sets of a default-grouping build with the published tables.

Runs are matched on (config_id, big_recording_index, kernel_method, w_interaction), not run_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def memberships(directory: Path, run_ids: list[str] | None = None) -> dict[str, frozenset[str]]:
    table = pq.read_table(directory / "selections.parquet", columns=["run_id", "row_id"])
    if run_ids is not None:
        table = table.filter(pc.is_in(table["run_id"], value_set=__import__("pyarrow").array(run_ids)))
    grouped: dict[str, set[str]] = {}
    for run, row in zip(table["run_id"].to_pylist(), table["row_id"].to_pylist(), strict=True):
        grouped.setdefault(run, set()).add(row)
    return {run: frozenset(rows) for run, rows in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    columns = ["run_id", "config_id", "method", "big_recording_index", "kernel_method", "w_interaction"]
    candidate_runs = pq.read_table(args.candidate / "selection_runs.parquet", columns=columns).to_pylist()
    published_runs = pq.read_table(args.published / "selection_runs.parquet", columns=columns).to_pylist()
    key = lambda r: (r["config_id"], r["big_recording_index"], r["kernel_method"], r["w_interaction"])  # noqa: E731
    published_by_key = {key(r): r["run_id"] for r in published_runs}
    matched = [(r, published_by_key.get(key(r))) for r in candidate_runs]
    missing = [key(r) for r, p in matched if p is None]
    candidate_sets = memberships(args.candidate)
    published_sets = memberships(args.published, [p for _, p in matched if p is not None])
    rows = []
    for run, published_id in matched:
        if published_id is None:
            continue
        same = candidate_sets[run["run_id"]] == published_sets[published_id]
        rows.append({**{k: run[k] for k in columns[1:]}, "identical": same, "same_run_id": run["run_id"] == published_id})
    by_method: dict[str, dict[str, int]] = {}
    for row in rows:
        label = row["method"] if row["method"] != "dpp" else f"dpp:{row['kernel_method']}:{'w0' if row['w_interaction'] == 0 else 'w>0'}"
        entry = by_method.setdefault(label, {"runs": 0, "identical": 0, "same_run_id": 0})
        entry["runs"] += 1
        entry["identical"] += int(row["identical"])
        entry["same_run_id"] += int(row["same_run_id"])
    dpp_recordings = sorted({r["big_recording_index"] for r in rows if r["method"] == "dpp"})
    per_recording = {
        recording: all(r["identical"] for r in rows if r["method"] == "dpp" and r["big_recording_index"] == recording)
        for recording in dpp_recordings
    }
    result = {
        "compared_runs": len(rows),
        "missing_in_published": len(missing),
        "by_method": by_method,
        "dpp_recordings_fully_identical": per_recording,
        "mismatches": [r for r in rows if not r["identical"]][:50],
    }
    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "mismatches"}, indent=2, default=str))
    print("first mismatches:", json.dumps(result["mismatches"][:8], default=str))


if __name__ == "__main__":
    main()
