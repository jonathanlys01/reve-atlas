#!/usr/bin/env python
"""Build a deterministic, selection-complete display subset of the atlas table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=Path("data/real/atlas.parquet"))
    parser.add_argument("--selections", type=Path, default=Path("data/real/selections.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/real/display.parquet"))
    parser.add_argument("--sample-cap", type=int, default=1_760_000)
    parser.add_argument("--stride", type=int, default=13)
    return parser.parse_args()


def build(atlas_path: Path, selections_path: Path, output_path: Path, sample_cap: int, stride: int) -> int:
    selection_table = pq.read_table(selections_path, columns=["row_id"])
    required_ids = set(selection_table["row_id"].combine_chunks().to_pylist())
    initial_required_count = len(required_ids)
    parquet_file = pq.ParquetFile(atlas_path)
    selected_tables: list[pa.Table] = []
    sampled_tables: list[pa.Table] = []
    row_start = 0
    sampled_count = 0
    columns = parquet_file.schema_arrow.names
    for batch in parquet_file.iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        row_ids = table["row_id"].to_pylist()
        required_mask = np.fromiter((row_id in required_ids for row_id in row_ids), dtype=bool, count=len(row_ids))
        if required_mask.any():
            selected_tables.append(table.filter(pa.array(required_mask)))
            required_ids.difference_update(row_id for row_id, keep in zip(row_ids, required_mask, strict=True) if keep)
        remaining = sample_cap - sampled_count
        if remaining > 0:
            offsets = np.arange(row_start, row_start + len(row_ids), dtype=np.int64)
            sample_mask = (offsets % stride == 0) & ~required_mask
            sample_indices = np.flatnonzero(sample_mask)[:remaining]
            if len(sample_indices):
                sampled_tables.append(table.take(pa.array(sample_indices, type=pa.int64())))
                sampled_count += len(sample_indices)
        row_start += len(row_ids)

    if required_ids:
        preview = ", ".join(sorted(required_ids)[:3])
        raise ValueError(f"{len(required_ids)} selection members were missing from the atlas, e.g. {preview}")
    display = pa.concat_tables([*selected_tables, *sampled_tables], promote_options="default")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(display, output_path, compression="zstd", row_group_size=100_000)
    print(
        f"Wrote {display.num_rows} display rows ({initial_required_count} selection members, "
        f"{sampled_count} deterministic samples) to {output_path}"
    )
    return display.num_rows


def main() -> None:
    args = parse_args()
    if args.sample_cap < 0 or args.stride < 1:
        raise ValueError("sample-cap must be non-negative and stride must be positive")
    build(args.atlas, args.selections, args.output, args.sample_cap, args.stride)


if __name__ == "__main__":
    main()
