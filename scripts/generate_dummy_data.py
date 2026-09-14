#!/usr/bin/env python
"""Generate a deterministic, synthetic REVE Embedding Atlas dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATASETS = (
    ("TUAB", "EEG", 200.0, 21),
    ("TUEV", "EEG", 200.0, 23),
    ("PhysioNet Motor Imagery", "EEG", 200.0, 64),
    ("HMC Sleep", "EEG", 200.0, 4),
    ("MEG Face Recognition", "MEG", 400.0, 102),
)
CLUSTER_CENTERS = np.asarray(
    [(-4.0, -1.5), (-1.5, 2.8), (1.2, -2.5), (3.8, 1.8), (0.2, 0.4)],
    dtype=np.float32,
)


def atlas_schema() -> pa.Schema:
    """Return the public schema consumed by the demo frontend."""
    return pa.schema(
        [
            ("row_id", pa.int64()),
            ("window_id", pa.string()),
            ("projection_x", pa.float32()),
            ("projection_y", pa.float32()),
            ("recon_loss", pa.float32()),
            ("big_recording_index", pa.int64()),
            ("session_index", pa.int64()),
            ("offset", pa.int64()),
            ("dataset", pa.string()),
            ("modality", pa.string()),
            ("sampling_rate", pa.float32()),
            ("n_channels", pa.int64()),
        ],
    )


def generate_table(row_count: int = 2_000, seed: int = 42) -> pa.Table:
    """Build deterministic clustered points with representative window metadata."""
    if row_count <= 0:
        raise ValueError(f"row_count must be positive, got {row_count}")

    rng = np.random.default_rng(seed)
    row_ids = np.arange(row_count, dtype=np.int64)
    recording_indices = row_ids // 120
    session_indices = (row_ids // 40) % 3
    offsets = (row_ids % 40) * 2_000
    dataset_indices = recording_indices % len(DATASETS)

    spread = rng.normal(0.0, 0.7, size=(row_count, 2)).astype(np.float32)
    angle = (session_indices.astype(np.float32) - 1.0) * np.float32(0.22)
    spread[:, 0] += angle
    coordinates = CLUSTER_CENTERS[dataset_indices] + spread

    loss_base = np.asarray([0.18, 0.27, 0.13, 0.34, 0.22], dtype=np.float32)
    reconstruction_losses = np.maximum(
        np.float32(0.001),
        loss_base[dataset_indices] + rng.normal(0.0, 0.045, size=row_count).astype(np.float32),
    )

    dataset_names = [DATASETS[index][0] for index in dataset_indices]
    modalities = [DATASETS[index][1] for index in dataset_indices]
    sampling_rates = np.asarray([DATASETS[index][2] for index in dataset_indices], dtype=np.float32)
    channel_counts = np.asarray([DATASETS[index][3] for index in dataset_indices], dtype=np.int64)
    window_ids = [
        f"{recording}_-_{session}_-_{offset}"
        for recording, session, offset in zip(recording_indices, session_indices, offsets, strict=True)
    ]

    return pa.Table.from_arrays(
        [
            pa.array(row_ids),
            pa.array(window_ids),
            pa.array(coordinates[:, 0]),
            pa.array(coordinates[:, 1]),
            pa.array(reconstruction_losses),
            pa.array(recording_indices),
            pa.array(session_indices),
            pa.array(offsets),
            pa.array(dataset_names),
            pa.array(modalities),
            pa.array(sampling_rates),
            pa.array(channel_counts),
        ],
        schema=atlas_schema(),
    )


def write_dummy_data(output_path: str | Path, row_count: int = 2_000, seed: int = 42) -> Path:
    """Generate and write the demo table as compressed Parquet."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(generate_table(row_count, seed), path, compression="zstd", row_group_size=512)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo.parquet"))
    parser.add_argument("--rows", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = write_dummy_data(args.output, args.rows, args.seed)
    print(f"Wrote {args.rows:,} synthetic windows to {path}")


if __name__ == "__main__":
    main()
