#!/usr/bin/env python
"""Compact browser payload for one embedding-space group set.

The browser only needs, per display point, its group columns and which DPP sweep points
(direction, eta, kernel, w) selected it. Instead of the per-run membership table (~20M rows) and
the full runs table, write:

- ``overlay.parquet``: one row per display point with the group columns and one uint32 bitmask
  per (direction, eta, kernel); bit i is set when the point is in the DPP run at the i-th value of
  that kernel's w grid (``sweep.json``).
- ``sweep.json``: the w grid per kernel, the mask columns, and consistency counts.
- ``curve_summary.parquet``: copied unchanged (the operating curves are already aggregated).

The overlay is checked to reproduce every DPP membership exactly before anything is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

GROUP_COLUMNS = ("group_id", "group_colour", "group_split", "group_depth", "group_size")


def mask_column(direction: str, eta: float, kernel: str) -> str:
    return f"dpp_{direction}_eta_{round(eta * 100):03d}_{kernel}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-dir", type=Path, required=True, help="finished set directory (display/selections/runs)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    display = pq.read_table(args.set_dir / "display.parquet", columns=["row_id", *GROUP_COLUMNS])
    runs = pq.read_table(
        args.set_dir / "selection_runs_web.parquet",
        columns=["run_id", "method", "direction", "selection_eta", "kernel_method", "w_interaction"],
    ).to_pylist()
    dpp = [run for run in runs if run["method"] == "dpp"]
    grid = {
        kernel: sorted({run["w_interaction"] for run in dpp if run["kernel_method"] == kernel})
        for kernel in sorted({run["kernel_method"] for run in dpp})
    }
    if max(len(values) for values in grid.values()) > 32:
        raise ValueError("a kernel w grid has more than 32 values; uint32 masks cannot hold it")
    column_of = {run["run_id"]: mask_column(run["direction"], run["selection_eta"], run["kernel_method"]) for run in dpp}
    bit_of = {run["run_id"]: grid[run["kernel_method"]].index(run["w_interaction"]) for run in dpp}
    columns = sorted(set(column_of.values()))
    for column in columns:  # every mask column must cover its kernel's full grid
        kernel = column.rsplit("_", 1)[1]
        present = {bit_of[r] for r, c in column_of.items() if c == column}
        if present != set(range(len(grid[kernel]))):
            raise ValueError(f"{column} does not cover the {kernel} w grid")

    row_ids = display["row_id"].to_pylist()
    position = {row: i for i, row in enumerate(row_ids)}
    masks = {column: np.zeros(len(row_ids), dtype=np.uint32) for column in columns}
    selections = pq.read_table(args.set_dir / "selections.parquet", columns=["run_id", "row_id"])
    expected = 0
    for run, row in zip(selections["run_id"].to_pylist(), selections["row_id"].to_pylist(), strict=True):
        column = column_of.get(run)
        if column is None:
            continue
        masks[column][position[row]] |= np.uint32(1 << bit_of[run])
        expected += 1
    # Each (run, row) membership is one bit; runs of one column never share a bit, so the bit
    # count must equal the DPP membership count exactly.
    bits = sum(int(np.unpackbits(mask.view(np.uint8)).sum()) for mask in masks.values())
    if bits != expected:
        raise ValueError(f"overlay holds {bits} membership bits, expected {expected}")

    overlay = display
    for column in columns:
        overlay = overlay.append_column(column, pa.array(masks[column], pa.uint32()))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(overlay, output / "overlay.parquet", compression="zstd", row_group_size=250_000)
    shutil.copyfile(args.set_dir / "curve_summary.parquet", output / "curve_summary.parquet")
    etas = sorted({run["selection_eta"] for run in dpp})
    sweep = {
        "schema_version": 1,
        "set": args.set_dir.name,
        "rows": len(row_ids),
        "groups": int(pa.compute.max(display["group_id"]).as_py()) + 1,
        "w_grid": grid,
        "etas": etas,
        "directions": sorted({run["direction"] for run in dpp}),
        "mask_columns": columns,
        "dpp_runs": len(dpp),
        "dpp_memberships": expected,
        "overlay_sha256": hashlib.sha256((output / "overlay.parquet").read_bytes()).hexdigest(),
    }
    (output / "sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    size = (output / "overlay.parquet").stat().st_size / 1e6
    print(f"{args.set_dir.name}: {len(columns)} mask columns, {expected:,} memberships, overlay {size:.1f} MB")


if __name__ == "__main__":
    main()
