#!/usr/bin/env python
"""Extract original embeddings for exactly the public display population."""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE_COLUMNS = (
    "window_id",
    "big_recording_index",
    "session_index",
    "offset",
    "recon_loss",
    "embedding",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original embedding shard directory")
    parser.add_argument("--display", type=Path, default=Path("data/real/display.parquet"))
    parser.add_argument("--output", type=Path, required=True, help="Private local Parquet output")
    return parser.parse_args()


def build(source: Path, display: Path, output: Path) -> int:
    if not source.is_dir():
        raise ValueError(f"Expected a shard directory: {source}")
    display_rows = pq.read_table(display, columns=["window_id", "big_recording_index"])
    ids_by_recording: dict[int, set[str]] = defaultdict(set)
    for row in display_rows.to_pylist():
        ids_by_recording[int(row["big_recording_index"])].add(str(row["window_id"]))
    remaining = {value for values in ids_by_recording.values() for value in values}
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        for index in sorted(ids_by_recording):
            shard = source / f"{index}.parquet"
            if not shard.is_file():
                raise FileNotFoundError(f"Missing embedding shard for display recording {index}: {shard}")
            wanted = pa.array(sorted(ids_by_recording[index]), type=pa.string())
            parquet_file = pq.ParquetFile(shard)
            for batch in parquet_file.iter_batches(batch_size=100_000, columns=list(SOURCE_COLUMNS)):
                table = pa.Table.from_batches([batch])
                mask = pc.is_in(table["window_id"], value_set=wanted)
                if not pc.any(mask).as_py():
                    continue
                selected = table.filter(mask)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, selected.schema, compression="zstd")
                writer.write_table(selected)
                ids = selected["window_id"].to_pylist()
                remaining.difference_update(ids)
                written += len(ids)
            print(f"recording={index} display_rows={len(ids_by_recording[index])} extracted", flush=True)
        if remaining:
            preview = ", ".join(sorted(remaining)[:3])
            raise ValueError(f"{len(remaining)} display rows were absent from the embedding shards, e.g. {preview}")
        if writer is None:
            raise ValueError("No display rows were extracted")
        writer.close()
        writer = None
        os.replace(temporary_path, output)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
    print(f"Wrote {written:,} display embedding rows to {output}")
    return written


def main() -> None:
    args = parse_args()
    build(args.source, args.display, args.output)


if __name__ == "__main__":
    main()
