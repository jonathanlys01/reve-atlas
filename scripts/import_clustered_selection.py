#!/usr/bin/env python
"""Validate a full-corpus DPP manifest and export its existing atlas-display intersection."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BATCH_SIZE = 8192


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_report(manifest: Path) -> dict:
    report = json.loads(manifest.with_suffix(".json").read_text())
    config = report["config"]
    n, k = config["group_size"], config["keep_per_group"]
    if report.get("selector") != "balanced_two_means_dpp" or not 1 <= k <= n:
        raise ValueError("Expected a clustered DPP report with 1 <= k <= N")
    if report["manifest_rows"] <= 0 or report["manifest_rows"] != report["source_rows"] * k // n:
        raise ValueError("Manifest report does not have the exact nonempty retention quota")
    if sha256_file(manifest) != report["manifest_sha256"]:
        raise ValueError("Manifest SHA-256 does not match its report")
    return report


def _load_ids(db: sqlite3.Connection, manifest: Path, expected: int) -> None:
    db.execute(
        "CREATE TABLE selected (id TEXT PRIMARY KEY, atlas_hits INTEGER DEFAULT 0, display_hits INTEGER DEFAULT 0)"
    )
    count = 0
    previous = ""
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            name = line.removesuffix("\n")
            if not name or any(character.isspace() for character in name) or name <= previous:
                raise ValueError("Manifest IDs must be nonempty, sorted and unique, without whitespace")
            db.execute("INSERT INTO selected (id) VALUES (?)", (name,))
            previous = name
            count += 1
            if count % BATCH_SIZE == 0:
                db.commit()
    db.commit()
    if count != expected:
        raise ValueError(f"Manifest contains {count} IDs, expected {expected}")


def _mark_matches(db: sqlite3.Connection, path: Path, field: str) -> int:
    count = 0
    with pq.ParquetFile(path) as parquet:
        if "embedding" in parquet.schema_arrow.names:
            raise ValueError(f"Expected a public atlas table without embeddings: {path}")
        if "row_id" not in parquet.schema_arrow.names or not pa.types.is_string(
            parquet.schema_arrow.field("row_id").type
        ):
            raise ValueError(f"Expected string row_id in {path}")
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=["row_id"], use_threads=False):
            ids = batch.column("row_id")
            if ids.null_count:
                raise ValueError(f"Null row_id in {path}")
            names = ids.to_pylist()
            if field == "display_hits":
                try:
                    db.executemany("INSERT INTO display_ids (id) VALUES (?)", ((name,) for name in names))
                except sqlite3.IntegrityError as error:
                    raise ValueError("An ID is duplicated in the display table") from error
            else:
                db.executemany("UPDATE display_ids SET atlas_hits=atlas_hits+1 WHERE id=?", ((name,) for name in names))
            # field is an internal constant, never a CLI value.
            db.executemany(f"UPDATE selected SET {field}={field}+1 WHERE id=?", ((name,) for name in names))
            db.commit()
            count += len(batch)
    return count


def _write_memberships(db: sqlite3.Connection, output: Path) -> int:
    schema = pa.schema([("row_id", pa.string())])
    count = 0
    cursor = db.execute("SELECT id FROM selected WHERE display_hits=1 ORDER BY id")
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        while rows := cursor.fetchmany(BATCH_SIZE):
            writer.write_table(pa.table({"row_id": [row[0] for row in rows]}, schema=schema))
            count += len(rows)
    return count


def build_overlay(
    manifest: Path,
    atlas: Path,
    display: Path,
    output_dir: Path,
    *,
    work_dir: Path | None = None,
    atlas_revision: str | None = None,
) -> dict:
    """Export only IDs and allowlisted metadata; keep atlas/display inputs read-only."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    report = _read_report(manifest)
    source_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in (manifest, atlas, display)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clustered-overlay-db-", dir=work_dir) as scratch:
        db = sqlite3.connect(Path(scratch) / "members.sqlite")
        try:
            db.execute("PRAGMA cache_size=-32768")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA mmap_size=0")
            _load_ids(db, manifest, report["manifest_rows"])
            db.execute("CREATE TABLE display_ids (id TEXT PRIMARY KEY, atlas_hits INTEGER DEFAULT 0)")
            display_rows = _mark_matches(db, display, "display_hits")
            atlas_rows = _mark_matches(db, atlas, "atlas_hits")
            if atlas_rows != report["source_rows"]:
                raise ValueError(f"Atlas has {atlas_rows} rows but the selection source has {report['source_rows']}")
            if db.execute("SELECT COUNT(*) FROM selected WHERE atlas_hits != 1").fetchone()[0]:
                raise ValueError("A selected ID is missing or duplicated in the full atlas")
            if db.execute("SELECT COUNT(*) FROM display_ids WHERE atlas_hits != 1").fetchone()[0]:
                raise ValueError("A display ID is missing or duplicated in the full atlas")
            with tempfile.TemporaryDirectory(prefix=".clustered-overlay-", dir=output_dir.parent) as staging:
                public = Path(staging) / "public"
                public.mkdir()
                memberships = public / "clustered_selections.parquet"
                displayed_selected = _write_memberships(db, memberships)
                shutil.copyfile(manifest, public / "clustered_manifest.txt")
                config = report["config"]
                metadata = {
                    "schema_version": 1,
                    "run_id": f"clustered-dpp-{report['manifest_sha256'][:16]}",
                    "selector": report["selector"],
                    "source_rows": report["source_rows"],
                    "selected_rows": report["manifest_rows"],
                    "display_rows": display_rows,
                    "display_selected_rows": displayed_selected,
                    "group_size": config["group_size"],
                    "keep_per_group": config["keep_per_group"],
                    "quality_weight": config["quality_weight"],
                    "seed": config["seed"],
                    "selection_eta": config["keep_per_group"] / config["group_size"],
                    "manifest_sha256": report["manifest_sha256"],
                    "selection_report_sha256": sha256_file(manifest.with_suffix(".json")),
                    "memberships_sha256": sha256_file(memberships),
                    "display_sha256": sha256_file(display),
                    "atlas_revision": atlas_revision,
                    "code_sha256": report.get("code_sha256", {}),
                    "display_policy": "intersection_with_existing_display_population",
                }
                if sha256_file(public / "clustered_manifest.txt") != metadata["manifest_sha256"]:
                    raise ValueError("Manifest changed while copying")
                for path, before in source_stats.items():
                    if (path.stat().st_size, path.stat().st_mtime_ns) != before:
                        raise ValueError(f"Input changed during import: {path}")
                (public / "clustered_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                if output_dir.exists():
                    raise FileExistsError(f"Refusing to overwrite {output_dir}")
                public.rename(output_dir)
                return metadata
        finally:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Full .txt selection with sibling .json report.")
    parser.add_argument("--atlas", type=Path, required=True, help="Existing full public atlas.parquet.")
    parser.add_argument("--display", type=Path, required=True, help="Existing browser display.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for three publishable artifacts.")
    parser.add_argument("--work-dir", type=Path, help="Existing scratch directory for disk-backed ID validation.")
    parser.add_argument("--atlas-revision", help="Pinned source dataset revision recorded for provenance.")
    args = parser.parse_args()
    metadata = build_overlay(
        args.manifest,
        args.atlas,
        args.display,
        args.output_dir,
        work_dir=args.work_dir,
        atlas_revision=args.atlas_revision,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
