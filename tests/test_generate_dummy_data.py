from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.generate_dummy_data import atlas_schema, generate_table, write_dummy_data

SCHEMA_TEST_ROWS = 137
ROUND_TRIP_ROWS = 33


def test_generate_table_has_exact_atlas_schema_and_valid_rows() -> None:
    table = generate_table(row_count=SCHEMA_TEST_ROWS, seed=7)

    assert table.schema == atlas_schema()
    assert table.num_rows == SCHEMA_TEST_ROWS
    assert "embedding" not in table.column_names
    assert len(set(table["row_id"].to_pylist())) == SCHEMA_TEST_ROWS
    assert len(set(table["window_id"].to_pylist())) == SCHEMA_TEST_ROWS
    assert set(table["modality"].to_pylist()) <= {"EEG", "MEG"}

    for column in ("projection_x", "projection_y", "recon_loss"):
        assert np.all(np.isfinite(table[column].to_numpy()))
    assert np.all(table["recon_loss"].to_numpy() > 0.0)


def test_generate_table_is_deterministic() -> None:
    first = generate_table(row_count=64, seed=123)
    second = generate_table(row_count=64, seed=123)

    assert first.equals(second)


def test_write_dummy_data_round_trips(tmp_path) -> None:
    output = write_dummy_data(tmp_path / "nested" / "demo.parquet", row_count=ROUND_TRIP_ROWS, seed=5)

    assert output.is_file()
    table = pq.read_table(output)
    assert table.schema == atlas_schema()
    assert table.num_rows == ROUND_TRIP_ROWS
    assert pq.ParquetFile(output).metadata.num_row_groups == 1


def test_generate_table_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="row_count must be positive"):
        generate_table(row_count=0)


def test_schema_uses_compact_numeric_types() -> None:
    schema = atlas_schema()

    assert schema.field("projection_x").type == pa.float32()
    assert schema.field("projection_y").type == pa.float32()
    assert schema.field("recon_loss").type == pa.float32()
