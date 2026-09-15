#!/usr/bin/env python
"""Run the display selection sweep as resumable per-candidate tasks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from scripts.real_atlas import atomic_write_json, atomic_write_table, config_fingerprint, fingerprint_path, load_config

HEARTBEAT_SECONDS = 60


def now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(path: Path, updates: dict[str, Any]) -> None:
    status = {}
    if path.exists():
        status = read_json(path)
    status.update(updates)
    atomic_write_json(status, path)


def task_is_complete(task_directory: Path, task_config: dict[str, Any], source_fp: str) -> bool:
    manifest_path = task_directory / "manifest.json"
    runs_path = task_directory / "selection_runs.parquet"
    selections_path = task_directory / "selections.parquet"
    if not (manifest_path.is_file() and runs_path.is_file() and selections_path.is_file()):
        return False
    try:
        selection = read_json(manifest_path).get("selection", {})
        return (
            selection.get("configuration_fingerprint") == config_fingerprint(task_config)
            and selection.get("source_fingerprint") == source_fp
            and pq.ParquetFile(runs_path).metadata.num_rows > 0
            and pq.ParquetFile(selections_path).metadata.num_rows > 0
        )
    except (OSError, ValueError, KeyError):
        return False


def build_tasks(config: dict[str, Any], task_root: Path) -> list[tuple[str, dict[str, Any], Path]]:
    rankings = [dict(item) for item in config.get("rankings", [])]
    dpp_by_candidate = {str(item["candidates"]): dict(item) for item in config.get("dpp", [])}
    baselines = [dict(item) for item in config.get("baselines", [])]
    if not rankings:
        raise ValueError("The overnight config has no rankings")
    tasks = []
    for ranking in rankings:
        ranking_id = str(ranking["id"])
        dpp = dpp_by_candidate.get(ranking_id)
        if dpp is None:
            raise ValueError(f"No DPP definition found for ranking {ranking_id}")
        task_config = copy.deepcopy(config)
        task_config["rankings"] = [ranking]
        task_config["dpp"] = [dpp]
        # Put each random-stratified baseline in exactly one task per eta.
        task_config["baselines"] = []
        if str(ranking.get("direction")) == "top":
            eta = float(ranking["selection_eta"])
            task_config["baselines"] = [
                baseline for baseline in baselines if float(baseline["selection_eta"]) == eta
            ]
        task_id = ranking_id
        task_directory = task_root / task_id
        task_config["output"] = {**dict(config.get("output", {})), "directory": str(task_directory)}
        tasks.append((task_id, task_config, task_directory))
    return tasks


def aggregate(tasks: list[tuple[str, dict[str, Any], Path]], output_root: Path, base_config: dict[str, Any], source_fp: str) -> dict[str, Any]:
    selection_tables = [pq.read_table(task_directory / "selections.parquet") for _, _, task_directory in tasks]
    run_tables = [pq.read_table(task_directory / "selection_runs.parquet") for _, _, task_directory in tasks]
    selections = pa.concat_tables(selection_tables, promote_options="default")
    runs = pa.concat_tables(run_tables, promote_options="default")
    atomic_write_table(selections, output_root / "selections.parquet")
    atomic_write_table(runs, output_root / "selection_runs.parquet")
    # Merge into any existing manifest.json (e.g. the full atlas build's own
    # manifest, when this sweep's output directory is the shared data/real/
    # directory) rather than overwriting it — matches the read-merge-write
    # pattern build_selections.build() already uses for the same file.
    manifest_path = output_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("schema_version", 1)
    manifest["artifacts"] = {
        **manifest.get("artifacts", {}),
        "memberships": "selections.parquet",
        "runs": "selection_runs.parquet",
        "curves": "curve_summary.parquet",
    }
    manifest["selection"] = {
        "population": str(base_config.get("input", {}).get("population", "display")),
        "source_fingerprint": source_fp,
        "configuration_fingerprint": config_fingerprint(base_config),
        "rows": selections.num_rows,
        "runs": runs.num_rows,
        "ranking_runs": sum(table.filter(pa.compute.equal(table["method"], "ranking")).num_rows for table in run_tables),
        "dpp_runs": sum(table.filter(pa.compute.equal(table["method"], "dpp")).num_rows for table in run_tables),
        "random_stratified_runs": sum(table.filter(pa.compute.equal(table["method"], "random_stratified")).num_rows for table in run_tables),
    }
    manifest["selection_sweep_tasks"] = [task_id for task_id, _, _ in tasks]
    atomic_write_json(manifest, manifest_path)
    return manifest


def write_curve_summary(runs_path: Path, output_path: Path) -> None:
    rows = pq.read_table(runs_path).to_pylist()
    metric_names = ("vendi_score", "vendi_score_normalized", "mean_pairwise_cosine", "mean_recon_loss")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["config_id"],
            row["method"],
            row["direction"],
            row["scope"],
            row["selection_eta"],
            row["kernel_method"],
            row["w_interaction"],
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, group in sorted(groups.items(), key=lambda item: tuple("" if value is None else value for value in item[0])):
        summary = {
            "config_id": key[0],
            "method": key[1],
            "direction": key[2],
            "scope": key[3],
            "selection_eta": key[4],
            "kernel_method": key[5],
            "w_interaction": key[6],
            "recording_count": len(group),
        }
        for metric in metric_names:
            values = sorted(float(row[metric]) for row in group if row[metric] is not None)
            middle = values[len(values) // 2] if len(values) % 2 else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2
            summary[f"{metric}_mean"] = sum(values) / len(values)
            summary[f"{metric}_median"] = middle
            summary[f"{metric}_q25"] = values[max(0, int(0.25 * (len(values) - 1)))]
            summary[f"{metric}_q75"] = values[min(len(values) - 1, int(0.75 * (len(values) - 1)))]
        sizes = [int(row["actual_size"]) for row in group]
        summary["actual_size_mean"] = sum(sizes) / len(sizes)
        summary_rows.append(summary)
    schema = pa.schema([
        ("config_id", pa.string()),
        ("method", pa.string()),
        ("direction", pa.string()),
        ("scope", pa.string()),
        ("selection_eta", pa.float64()),
        ("kernel_method", pa.string()),
        ("w_interaction", pa.float64()),
        ("recording_count", pa.int32()),
        *[(f"{metric}_{stat}", pa.float64()) for metric in metric_names for stat in ("mean", "median", "q25", "q75")],
        ("actual_size_mean", pa.float64()),
    ])
    arrays = [pa.array([row.get(field.name) for row in summary_rows], type=field.type) for field in schema]
    atomic_write_table(pa.Table.from_arrays(arrays, schema=schema), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(str(config.get("output", {}).get("directory", "data/real/overnight"))).expanduser().resolve()
    task_root = output_root / "tasks"
    status_path = (args.status or output_root / "overnight_status.json").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    task_root.mkdir(parents=True, exist_ok=True)
    source_fp = fingerprint_path(str(config["input"]["embeddings"]))
    tasks = build_tasks(config, task_root)
    completed_ids = [task_id for task_id, task_config, task_directory in tasks if task_is_complete(task_directory, task_config, source_fp)]
    write_status(status_path, {"state": "running", "current_run": None, "completed_runs": completed_ids, "total_runs": len(tasks), "last_progress_at": now(), "last_result_at": None, "message": "overnight sweep started", "config": str(args.config.resolve()), "output_root": str(output_root), "pid": os.getpid()})
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            write_status(status_path, {"last_progress_at": now(), "message": "active task still running"})

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for task_id, task_config, task_directory in tasks:
            if task_id in completed_ids:
                continue
            task_directory.mkdir(parents=True, exist_ok=True)
            task_config_path = task_directory / "config.yaml"
            task_config_path.write_text(yaml.safe_dump(task_config, sort_keys=False), encoding="utf-8")
            write_status(status_path, {"current_run": task_id, "last_progress_at": now(), "message": f"starting {task_id}"})
            env = os.environ.copy()
            # build_selections parallelizes across recordings (one process per
            # recording); each of those processes must stay single-threaded or
            # BLAS threading (inherited by fork before any per-worker env
            # reset can take effect) oversubscribes the machine on top of that.
            env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            result = subprocess.run([sys.executable, "-m", "scripts.build_selections", "--config", str(task_config_path)], check=False, env=env)
            if result.returncode:
                raise RuntimeError(f"task {task_id} failed with exit code {result.returncode}")
            if not task_is_complete(task_directory, task_config, source_fp):
                raise RuntimeError(f"task {task_id} exited successfully but its artifacts failed validation")
            completed_ids.append(task_id)
            write_status(status_path, {"completed_runs": completed_ids, "current_run": None, "last_progress_at": now(), "last_result_at": now(), "message": f"completed {task_id}"})
        manifest = aggregate(tasks, output_root, config, source_fp)
        write_curve_summary(output_root / "selection_runs.parquet", output_root / "curve_summary.parquet")
        write_status(status_path, {"state": "completed", "current_run": None, "completed_runs": completed_ids, "last_progress_at": now(), "last_result_at": now(), "message": f"all tasks completed: {manifest["selection"]["runs"]:,} runs", "pid": None})
    except Exception as error:
        write_status(status_path, {"state": "failed", "last_progress_at": now(), "message": str(error), "pid": None})
        raise
    finally:
        stop_heartbeat.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
