#!/usr/bin/env python
"""Reproject the public display population with an explicit, resumable job."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.real_atlas import load_config, project_embeddings, read_source


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pq.write_table(table, temporary_path, compression="zstd", row_group_size=100_000)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build(config: dict[str, object], source_path: Path, display_path: Path, output_path: Path, status_path: Path) -> None:
    status = {
        "state": "running",
        "current_run": "display-projection",
        "completed_runs": [],
        "last_progress_at": _now(),
        "last_result_at": None,
        "message": "loading display embeddings",
        "source": str(source_path),
        "display": str(display_path),
        "output": str(output_path),
        "pid": os.getpid(),
    }
    _write_json(status_path, status)
    try:
        source = read_source(source_path)
        display = pq.read_table(display_path)
        source_ids = source["window_id"].combine_chunks()
        display_ids = display["row_id"].combine_chunks()
        positions = pc.index_in(display_ids, value_set=source_ids)
        if positions.null_count:
            raise ValueError(f"{positions.null_count} display rows are absent from the private embedding cache")
        embeddings = np.asarray(
            source["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(source.num_rows, 512)
        status.update(
            last_progress_at=_now(),
            message=f"projecting {source.num_rows:,} display embeddings",
            rows=source.num_rows,
        )
        _write_json(status_path, status)
        projection, provenance = project_embeddings(embeddings, dict(config.get("projection", {})))
        status.update(last_progress_at=_now(), message="aligning projected coordinates to display.parquet")
        _write_json(status_path, status)
        position_values = positions.to_numpy(zero_copy_only=False).astype(np.int64)
        output = display.set_column(
            display.schema.get_field_index("projection_x"),
            "projection_x",
            pa.array(projection[position_values, 0], type=pa.float32()),
        )
        output = output.set_column(
            output.schema.get_field_index("projection_y"),
            "projection_y",
            pa.array(projection[position_values, 1], type=pa.float32()),
        )
        _write_table(output, output_path)
        _write_json(
            status_path.with_name("display_projection_manifest.json"),
            {"rows": output.num_rows, "projection": provenance, "source": str(source_path)},
        )
        status.update(
            state="completed",
            current_run=None,
            completed_runs=["display-projection"],
            last_progress_at=_now(),
            last_result_at=_now(),
            message=f"wrote {output.num_rows:,} projected display rows",
            projection=provenance,
        )
        _write_json(status_path, status)
    except Exception as error:
        status.update(state="failed", last_progress_at=_now(), message=str(error))
        _write_json(status_path, status)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--display", type=Path, default=Path("data/real/display.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/real/display_reprojected.parquet"))
    parser.add_argument("--status", type=Path, default=Path("data/real/display_projection_status.json"))
    args = parser.parse_args()
    build(load_config(args.config), args.source, args.display, args.output, args.status)


if __name__ == "__main__":
    main()
