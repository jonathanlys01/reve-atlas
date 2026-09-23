#!/usr/bin/env python
"""Gate 4 layout for one artifact set: display.parquet with group columns, the web runs copy,
grouping/manifest bookkeeping, and the per-set assertions on the grouping."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.real_atlas import atomic_write_json, atomic_write_table

GROUP_COLUMNS = ("group_id", "group_path", "group_depth", "group_split", "group_size", "attached")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--display", type=Path, required=True)
    parser.add_argument("--n-min", type=int, required=True)
    parser.add_argument("--n-max", type=int, required=True)
    args = parser.parse_args()
    groups = pq.read_table(args.set_dir / "groups.parquet")
    display = pq.read_table(args.display)
    expected = display.num_rows

    sizes = np.bincount(groups["group_id"].to_numpy())
    assert int(sizes.sum()) == expected == groups.num_rows, "group sizes must sum to the display population"
    assert len(set(groups["row_id"].to_pylist())) == expected, "every row must be in exactly one group"
    assert sizes.min() >= args.n_min and sizes.max() <= args.n_max, (sizes.min(), sizes.max())

    position = {row: i for i, row in enumerate(groups["row_id"].to_pylist())}
    take = pa.array([position[row] for row in display["row_id"].to_pylist()], pa.int64())
    joined = display
    for column in GROUP_COLUMNS:
        joined = joined.append_column(column, groups[column].take(take))
    atomic_write_table(joined, args.set_dir / "display.parquet")
    shutil.copyfile(args.set_dir / "selection_runs.parquet", args.set_dir / "selection_runs_web.parquet")

    manifest_path = args.set_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {
        **manifest.get("artifacts", {}),
        "atlas": "display.parquet",
        "runs_web": "selection_runs_web.parquet",
        "groups": "groups.parquet",
        "index_metrics": "index_metrics.parquet",
    }
    metrics_path = args.set_dir / "grouping_metrics.json"
    grouping = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    grouping.pop("dbscan_call_log", None)
    manifest["grouping"] = {
        "groups": int(len(sizes)),
        "n_min": args.n_min,
        "n_max": args.n_max,
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
        **{k: v for k, v in grouping.items() if k not in {"group_size", "group_depth"}},
    }
    atomic_write_json(manifest, manifest_path)
    print(f"{args.set_dir.name}: {len(sizes)} groups, sizes {sizes.min()}..{sizes.max()}, display rows {joined.num_rows}")


if __name__ == "__main__":
    main()
