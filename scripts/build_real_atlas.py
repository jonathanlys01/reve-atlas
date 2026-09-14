#!/usr/bin/env python
"""Build the public 2-D REVE atlas from the original embeddings database."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa

from scripts.real_atlas import (
    DEFAULT_METADATA_COLUMNS,
    atomic_write_json,
    atomic_write_table,
    config_fingerprint,
    fingerprint_path,
    join_metadata,
    load_config,
    project_embeddings,
    read_recording_metadata,
    read_source,
    run_id,
)


def build(config: dict[str, object]) -> dict[str, object]:
    input_config = dict(config.get("input", {}))
    output_config = dict(config.get("output", {}))
    source_path = Path(str(input_config["embeddings"])).expanduser()
    metadata_path = Path(str(input_config["recording_metadata"])).expanduser()
    metadata_columns = tuple(input_config.get("metadata_columns", DEFAULT_METADATA_COLUMNS))
    table = read_source(source_path)
    metadata = read_recording_metadata(metadata_path, metadata_columns)
    table = join_metadata(table, metadata, metadata_columns)
    embeddings = np.asarray(
        table["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    ).reshape(table.num_rows, 512)
    projection, projection_provenance = project_embeddings(embeddings, dict(config.get("projection", {})))

    arrays = [
        pa.array(table["window_id"], type=pa.string()),
        pa.array(table["window_id"], type=pa.string()),
        pa.array(projection[:, 0], type=pa.float32()),
        pa.array(projection[:, 1], type=pa.float32()),
        pa.array(table["recon_loss"], type=pa.float32()),
        pa.array(table["big_recording_index"], type=pa.int64()),
        pa.array(table["session_index"], type=pa.int64()),
        pa.array(table["offset"], type=pa.int64()),
    ]
    fields = [
        "row_id",
        "window_id",
        "projection_x",
        "projection_y",
        "recon_loss",
        "big_recording_index",
        "session_index",
        "offset",
    ]
    for column in metadata_columns:
        arrays.append(table[column])
        fields.append(column)
    atlas = pa.Table.from_arrays(arrays, names=fields)

    output_directory = Path(str(output_config.get("directory", "data/real"))).expanduser()
    atlas_path = output_directory / "atlas.parquet"
    manifest_path = output_directory / "manifest.json"
    atomic_write_table(atlas, atlas_path)
    source_fingerprint = fingerprint_path(source_path)
    metadata_fingerprint = fingerprint_path(metadata_path)
    manifest = {
        "schema_version": 1,
        "artifacts": {"atlas": atlas_path.name},
        "source": {
            "embeddings": source_path.name,
            "recording_metadata": metadata_path.name,
            "fingerprint": source_fingerprint,
            "metadata_fingerprint": metadata_fingerprint,
            "rows": table.num_rows,
            "embedding_dim": 512,
        },
        "configuration_fingerprint": config_fingerprint(config),
        "projection": projection_provenance,
        "metadata_allowlist": list(metadata_columns),
        "atlas_schema": atlas.schema.to_string(show_field_metadata=False),
        "selection": None,
        "run_id": run_id(source_fingerprint, {"stage": "projection", "projection": config.get("projection", {})}),
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(load_config(args.config))
    print(
        f"Wrote {manifest['source']['rows']:,} atlas rows to {Path(str(load_config(args.config).get('output', {}).get('directory', 'data/real'))) / 'atlas.parquet'}"
    )


if __name__ == "__main__":
    main()
