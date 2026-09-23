import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.import_clustered_selection import build_overlay


def fixture(tmp_path, ids=("a", "d"), full=("a", "b", "c", "d", "e"), display=("a", "b", "c")):
    manifest = tmp_path / "keep.txt"
    manifest.write_text("".join(name + "\n" for name in ids))
    report = {
        "selector": "balanced_two_means_dpp",
        "source_rows": 5,
        "manifest_rows": 2,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "config": {"group_size": 2, "keep_per_group": 1, "seed": 29, "quality_weight": 0},
        "sources": [{"path": "/private/secret-source"}],
    }
    manifest.with_suffix(".json").write_text(json.dumps(report))
    atlas_path, display_path = tmp_path / "atlas.parquet", tmp_path / "display.parquet"
    pq.write_table(pa.table({"row_id": list(full)}), atlas_path)
    pq.write_table(pa.table({"row_id": list(display)}), display_path)
    return manifest, atlas_path, display_path, tmp_path / "public"


def test_display_intersection_and_global_counts(tmp_path):
    args = fixture(tmp_path)
    metadata = build_overlay(*args, atlas_revision="pinned-revision")
    assert metadata["source_rows"] == 5
    assert metadata["selected_rows"] == 2
    assert metadata["display_rows"] == 3
    assert metadata["display_selected_rows"] == 1
    assert metadata["atlas_revision"] == "pinned-revision"
    assert pq.read_table(args[-1] / "clustered_selections.parquet").to_pydict() == {"row_id": ["a"]}
    assert (args[-1] / "clustered_manifest.txt").read_bytes() == args[0].read_bytes()
    assert "/private/" not in (args[-1] / "clustered_run.json").read_text()
    assert "embedding" not in pq.read_schema(args[-1] / "clustered_selections.parquet").names
    with pytest.raises(FileExistsError):
        build_overlay(*args)


@pytest.mark.parametrize("ids", [("a", "a"), ("d", "a"), ("a", "d", "e")])
def test_bad_manifest_fails_without_publication(tmp_path, ids):
    args = fixture(tmp_path, ids=ids)
    with pytest.raises(ValueError):
        build_overlay(*args)
    assert not args[-1].exists()


@pytest.mark.parametrize("full", [("a", "b", "c", "e", "f"), ("a", "d", "d", "e", "f"), ("a", "b", "c", "d")])
def test_reject_missing_duplicate_or_wrong_source_population(tmp_path, full):
    args = fixture(tmp_path, full=full)
    with pytest.raises(ValueError):
        build_overlay(*args)
    assert not args[-1].exists()


def test_duplicate_display_member(tmp_path):
    args = fixture(tmp_path, display=("a", "a", "b"))
    with pytest.raises(ValueError, match="duplicated in the display"):
        build_overlay(*args)


def test_empty_display_intersection(tmp_path):
    args = fixture(tmp_path, display=("b", "c"))
    assert build_overlay(*args)["display_selected_rows"] == 0
    assert pq.read_table(args[-1] / "clustered_selections.parquet").num_rows == 0


def test_checksum_mismatch(tmp_path):
    args = fixture(tmp_path)
    args[0].write_text("a\ne\n")
    with pytest.raises(ValueError, match="SHA-256"):
        build_overlay(*args)


def test_private_embedding_input(tmp_path):
    args = fixture(tmp_path)
    pq.write_table(pa.table({"row_id": ["a", "b", "c", "d", "e"], "embedding": [[1.0]] * 5}), args[1])
    with pytest.raises(ValueError, match="without embeddings"):
        build_overlay(*args)


@pytest.mark.parametrize("display", [("a", "z"), ("b", "b")])
def test_rejects_bad_unselected_display_ids(tmp_path, display):
    args = fixture(tmp_path, display=display)
    with pytest.raises(ValueError):
        build_overlay(*args)
    assert not args[-1].exists()
