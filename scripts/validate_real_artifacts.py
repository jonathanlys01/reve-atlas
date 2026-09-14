#!/usr/bin/env python
"""Validate the public real-atlas artifact set against its manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.real_atlas import AtlasDataError, config_fingerprint, fingerprint_path, load_config


def validate(config: dict[str, object]) -> None:
    output_directory = Path(str(config.get("output", {}).get("directory", "data/real"))).expanduser()
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    atlas = pq.read_table(output_directory / "atlas.parquet")
    selections = pq.read_table(output_directory / "selections.parquet")
    runs = pq.read_table(output_directory / "selection_runs.parquet")
    for table_name, table in (("atlas", atlas), ("selections", selections), ("selection_runs", runs)):
        if "embedding" in table.column_names:
            raise AtlasDataError(f"Public {table_name} artifact contains an embedding column")
        for column in table.column_names:
            field = table.schema.field(column)
            if pa.types.is_floating(field.type):
                chunk = table[column].combine_chunks()
                values = chunk.to_numpy(zero_copy_only=False)
                valid = chunk.is_valid().to_numpy(zero_copy_only=False)
                if not np.all(np.isfinite(values[valid])):
                    raise AtlasDataError(f"Public {table_name}.{column} contains non-finite values")
    row_ids = atlas["row_id"].combine_chunks().to_pylist()
    if len(set(row_ids)) != len(row_ids):
        raise AtlasDataError("Atlas row_id values are not unique")
    atlas_ids = set(row_ids)
    selection_ids = selections["row_id"].combine_chunks().to_pylist()
    if not set(selection_ids) <= atlas_ids:
        raise AtlasDataError("Selection membership references a row absent from atlas.parquet")
    run_rows = runs.to_pylist()
    selection_rows = selections.to_pylist()
    counts: dict[str, int] = {}
    for row in selection_rows:
        counts[row["run_id"]] = counts.get(row["run_id"], 0) + 1
    for row in run_rows:
        if counts.get(row["run_id"], 0) != row["actual_size"]:
            raise AtlasDataError(f"Selection size mismatch for run {row['run_id']}")
        vendi = float(row["vendi_score"])
        size = int(row["actual_size"])
        if not np.isfinite(vendi) or not 1.0 - 1e-6 <= vendi <= size + 1e-6:
            raise AtlasDataError(f"Vendi out of bounds for run {row['run_id']}: {vendi} not in [1, {size}]")
    source_fingerprint = fingerprint_path(str(config["input"]["embeddings"]))
    if manifest.get("source", {}).get("fingerprint") != source_fingerprint:
        raise AtlasDataError("Embedding source fingerprint does not match manifest")
    if manifest.get("selection", {}).get("source_fingerprint") != source_fingerprint:
        raise AtlasDataError("Selection source fingerprint does not match manifest")
    if manifest.get("selection", {}).get("configuration_fingerprint") != config_fingerprint(config):
        raise AtlasDataError("Selection configuration fingerprint does not match manifest")
    print(f"Validated {len(row_ids):,} atlas rows, {len(selection_rows):,} memberships, and {len(run_rows):,} runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load_config(args.config))


if __name__ == "__main__":
    main()
