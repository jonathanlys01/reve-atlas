#!/usr/bin/env python
"""Build ratio-based ranking and full-display DPP selection artifacts."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.real_atlas import (
    AtlasDataError,
    atomic_write_json,
    config_fingerprint,
    cosine_gram,
    ensure_workspace,
    fingerprint_path,
    greedy_map_lazy,
    jaccard,
    load_config,
    mean_pairwise_cosine,
    normalize_utility,
    read_source,
    run_id,
    summarize_losses,
    vendi_score,
)
from scripts.torch_dpp import available as torch_dpp_available
from scripts.torch_dpp import greedy_map as torch_greedy_map

# Populated in the parent process before the DPP process pool is created, so
# forked workers inherit these (large, read-only) arrays via copy-on-write
# instead of having them pickled through the task queue.
_WORKER_DATA: dict[str, Any] = {}


def _dpp_worker_init() -> None:
    # One recording per worker process at a time; avoid each worker also
    # fanning out into multi-threaded BLAS on top of process-level parallelism.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _process_dpp_recording(task: dict[str, Any]) -> dict[str, Any]:
    """Run every kernel/w sweep point for one recording. Independent across recordings."""
    embeddings = _WORKER_DATA["embeddings"]
    losses = _WORKER_DATA["losses"]
    window_ids = _WORKER_DATA["window_ids"]
    recording_ids = _WORKER_DATA["recording_ids"]
    source_indices_by_run = _WORKER_DATA["source_indices_by_run"]

    recording = task["recording"]
    candidate_config_id = task["candidate_config_id"]
    selection_eta = task["selection_eta"]
    full_population = task["full_population"]
    kernel_definitions = task["kernel_definitions"]
    direction = task["direction"]
    scope = task["scope"]
    candidate_run_id = task["candidate_run_id"]
    epsilon = task["epsilon"]
    max_workspace_gib = task["max_workspace_gib"]
    seed = task["seed"]
    source_fingerprint = task["source_fingerprint"]
    configuration_fingerprint = task["configuration_fingerprint"]

    candidate_indices = (
        np.flatnonzero(recording_ids == recording).tolist()
        if full_population
        else source_indices_by_run[candidate_run_id]
    )
    if not full_population and scope == "global":
        candidate_indices = [index for index in candidate_indices if int(recording_ids[index]) == recording]
    candidate_k = len(candidate_indices)
    selection_k = max(1, min(candidate_k, int(np.ceil(selection_eta * candidate_k))))
    if candidate_k < selection_k:
        raise AtlasDataError(
            f"DPP {candidate_config_id} recording {recording} has {candidate_k} candidates, "
            f"fewer than selection_k={selection_k} for selection_eta={selection_eta}"
        )
    ensure_workspace(candidate_k, selection_k, max_workspace_gib)
    candidate_indices_array = np.asarray(candidate_indices, dtype=np.int64)
    candidate_losses = losses[candidate_indices_array]
    utility, utility_min, utility_max = normalize_utility(candidate_losses, direction)
    similarity = cosine_gram(embeddings[candidate_indices_array], epsilon=epsilon)
    baseline_indices = source_indices_by_run[candidate_run_id]
    baseline_set = {window_ids[int(index)] for index in baseline_indices}
    gpu_similarity = None
    use_gpu = torch_dpp_available()
    if use_gpu:
        import torch

        gpu_similarity = torch.as_tensor(similarity, dtype=torch.float64, device="cuda")

    runs_out: list[dict[str, Any]] = []
    row_members_out: list[list[dict[str, Any]]] = []
    selected_indices_out: list[list[int]] = []
    for kernel_method, kernel_values in kernel_definitions.items():
        for interaction in [float(value) for value in kernel_values["w_interaction"]]:
            if gpu_similarity is not None:
                selected_positions, gains, logdet = torch_greedy_map(
                    gpu_similarity, utility, str(kernel_method), interaction, selection_k, epsilon
                )
                solver_backend = "torch-gpu-greedy-cholesky"
                device = "cuda"
            else:
                selected_positions, gains, logdet = greedy_map_lazy(
                    similarity, utility, str(kernel_method), interaction, selection_k, epsilon
                )
                solver_backend = "numpy-greedy-cholesky"
                device = "cpu"
            selected_indices = [int(candidate_indices_array[position]) for position in selected_positions]
            selected_losses = losses[selected_indices]
            selected_utility = utility[selected_positions]
            selected_similarity = similarity[np.ix_(selected_positions, selected_positions)]
            selected_vendi = vendi_score(selected_similarity)
            run_configuration = {
                "stage": "dpp",
                "definition": task["definition"],
                "candidates": candidate_config_id,
                "recording": recording,
                "kernel_method": str(kernel_method),
                "w_interaction": interaction,
            }
            current_run_id = run_id(source_fingerprint, run_configuration)
            row_members = [
                {
                    "run_id": current_run_id,
                    "row_id": window_ids[index],
                    "rank": rank,
                    "marginal_logdet_gain": float(gains[rank]),
                    "utility": float(utility[position]),
                    "recon_loss": float(losses[index]),
                }
                for rank, (position, index) in enumerate(zip(selected_positions, selected_indices, strict=True))
            ]
            summary = summarize_losses(selected_losses, selected_utility)
            selected_set = {window_ids[index] for index in selected_indices}
            run = {
                "run_id": current_run_id,
                "config_id": f"{candidate_config_id}:{kernel_method}",
                "method": "dpp",
                "direction": direction,
                "scope": scope,
                "candidate_run_id": candidate_run_id,
                "big_recording_index": recording,
                "candidate_k": candidate_k,
                "selection_eta": selection_eta,
                "actual_size": len(selected_indices),
                "kernel_method": str(kernel_method),
                "w_interaction": interaction,
                "vendi_score": selected_vendi,
                "vendi_score_normalized": selected_vendi / len(selected_indices),
                "mean_pairwise_cosine": mean_pairwise_cosine(selected_similarity),
                "log_det": logdet,
                **summary,
                "baseline_jaccard": jaccard(selected_set, baseline_set),
                "adjacent_jaccard": None,
                "seed": seed,
                "epsilon": epsilon,
                "solver_backend": solver_backend,
                "device": device,
                "source_fingerprint": source_fingerprint,
                "configuration_fingerprint": configuration_fingerprint,
                "greedy_order": selected_positions,
                "marginal_logdet_gains": gains,
                "utility_min": utility_min,
                "utility_max": utility_max,
            }
            runs_out.append(run)
            row_members_out.append(row_members)
            selected_indices_out.append(selected_indices)
    if gpu_similarity is not None:
        del gpu_similarity
        import torch

        torch.cuda.empty_cache()
    return {
        "recording": recording,
        "runs": runs_out,
        "row_members": row_members_out,
        "selected_indices": selected_indices_out,
    }


def _ranking_order(losses: np.ndarray, window_ids: list[str], direction: str, indices: np.ndarray) -> np.ndarray:
    sign = -1.0 if direction == "top" else 1.0
    # Sort numerically first. Sorting all 22M object/string IDs with lexsort is
    # prohibitively expensive; IDs are only consulted for exact loss ties.
    values = sign * losses[indices]
    order = np.argsort(values, kind="stable")
    ordered = indices[order].copy()
    ordered_values = values[order]
    boundaries = np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    ends = np.concatenate((boundaries, np.array([len(ordered)])))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if end - start > 1:
            tied = ordered[start:end].tolist()
            tied.sort(key=window_ids.__getitem__)
            ordered[start:end] = tied
    return ordered


def _selection_schema() -> pa.Schema:
    return pa.schema(
        [
            ("run_id", pa.string()),
            ("row_id", pa.string()),
            ("rank", pa.int32()),
            ("marginal_logdet_gain", pa.float64()),
            ("utility", pa.float64()),
            ("recon_loss", pa.float32()),
        ],
    )


def _run_schema() -> pa.Schema:
    return pa.schema(
        [
            ("run_id", pa.string()),
            ("config_id", pa.string()),
            ("method", pa.string()),
            ("direction", pa.string()),
            ("scope", pa.string()),
            ("candidate_run_id", pa.string()),
            ("big_recording_index", pa.int64()),
            ("candidate_k", pa.int32()),
            ("selection_eta", pa.float64()),
            ("actual_size", pa.int32()),
            ("kernel_method", pa.string()),
            ("w_interaction", pa.float64()),
            ("vendi_score", pa.float64()),
            ("vendi_score_normalized", pa.float64()),
            ("mean_pairwise_cosine", pa.float64()),
            ("log_det", pa.float64()),
            ("mean_recon_loss", pa.float64()),
            ("median_recon_loss", pa.float64()),
            ("min_recon_loss", pa.float64()),
            ("max_recon_loss", pa.float64()),
            ("mean_utility", pa.float64()),
            ("recon_loss_q1", pa.float64()),
            ("recon_loss_q5", pa.float64()),
            ("recon_loss_q25", pa.float64()),
            ("recon_loss_q50", pa.float64()),
            ("recon_loss_q75", pa.float64()),
            ("recon_loss_q95", pa.float64()),
            ("recon_loss_q99", pa.float64()),
            ("baseline_jaccard", pa.float64()),
            ("adjacent_jaccard", pa.float64()),
            ("seed", pa.int64()),
            ("epsilon", pa.float64()),
            ("solver_backend", pa.string()),
            ("device", pa.string()),
            ("source_fingerprint", pa.string()),
            ("configuration_fingerprint", pa.string()),
            ("greedy_order", pa.list_(pa.int32())),
            ("marginal_logdet_gains", pa.list_(pa.float64())),
        ],
    )


def _cast_rows(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        arrays.append(pa.array([row.get(field.name) for row in rows], type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _write_tables_atomically(tables: list[tuple[pa.Table, Path]]) -> None:
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for table, destination in tables:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            os.close(fd)
            temporary_path = Path(temporary)
            pq.write_table(table, temporary_path, compression="zstd", row_group_size=100_000)
            temporary_paths.append((temporary_path, destination))
        for temporary_path, destination in temporary_paths:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def build(config: dict[str, Any]) -> dict[str, Any]:
    input_config = dict(config.get("input", {}))
    output_config = dict(config.get("output", {}))
    numerical_config = dict(config.get("numerics", {}))
    source_path = Path(str(input_config["embeddings"])).expanduser()
    source_fingerprint = fingerprint_path(source_path)
    configuration_fingerprint = config_fingerprint(config)
    source = read_source(source_path)
    window_ids = source["window_id"].combine_chunks().to_pylist()
    recording_ids = source["big_recording_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    losses = source["recon_loss"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    embeddings = np.asarray(
        source["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(-1, 512)
    epsilon = float(numerical_config.get("epsilon", 1.0e-6))
    max_workspace_gib = float(numerical_config.get("max_workspace_gib", 8.0))
    seed = int(config.get("projection", {}).get("seed", 0))

    selections: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    membership_by_run: dict[str, set[str]] = {}
    source_indices_by_run: dict[str, list[int]] = {}
    run_by_config_recording: dict[tuple[str, int | None], str] = {}
    dpp_groups: dict[tuple[str, int, str], list[tuple[float, str]]] = {}

    def add_run(run: dict[str, Any], member_rows: list[dict[str, Any]], source_indices: list[int]) -> None:
        runs.append(run)
        selections.extend(member_rows)
        membership_by_run[run["run_id"]] = {row["row_id"] for row in member_rows}
        source_indices_by_run[run["run_id"]] = source_indices

    for definition in config.get("rankings", []):
        definition = dict(definition)
        config_id = str(definition["id"])
        direction = str(definition["direction"])
        scope = str(definition["scope"])
        selection_eta_value = definition.get("selection_eta")
        selection_eta = None if selection_eta_value is None else float(selection_eta_value)
        if selection_eta is not None and not 0.0 < selection_eta <= 1.0:
            raise AtlasDataError(f"Invalid selection_eta for ranking {config_id}: {selection_eta}")
        configured_candidate_k = int(definition.get("candidate_k", 0))
        if selection_eta is None and configured_candidate_k <= 0:
            raise AtlasDataError(f"Ranking {config_id} needs selection_eta or positive candidate_k")
        if direction not in {"top", "bottom"} or scope not in {"global", "per_recording"}:
            raise AtlasDataError(f"Invalid ranking definition {config_id}: direction={direction}, scope={scope}")
        groups = [None] if scope == "global" else sorted(set(map(int, recording_ids.tolist())))
        for recording in groups:
            indices = (
                np.arange(len(window_ids), dtype=np.int64)
                if recording is None
                else np.flatnonzero(recording_ids == recording)
            )
            selection_k = (
                max(1, min(len(indices), int(np.ceil(selection_eta * len(indices)))))
                if selection_eta is not None
                else min(len(indices), configured_candidate_k)
            )
            order = _ranking_order(losses, window_ids, direction, indices)[:selection_k]
            selected_losses = losses[order]
            utility, utility_min, utility_max = normalize_utility(selected_losses, direction)
            run_configuration = {"stage": "ranking", "definition": definition, "recording": recording}
            current_run_id = run_id(source_fingerprint, run_configuration)
            row_members = [
                {
                    "run_id": current_run_id,
                    "row_id": window_ids[int(index)],
                    "rank": rank,
                    "marginal_logdet_gain": None,
                    "utility": float(utility[rank]),
                    "recon_loss": float(losses[int(index)]),
                }
                for rank, index in enumerate(order.tolist())
            ]
            summary = summarize_losses(selected_losses, utility)
            selected_similarity = cosine_gram(embeddings[order], epsilon=epsilon)
            selected_vendi = vendi_score(selected_similarity)
            run = {
                "run_id": current_run_id,
                "config_id": config_id,
                "method": "ranking",
                "direction": direction,
                "scope": scope,
                "candidate_run_id": None,
                "big_recording_index": recording,
                "candidate_k": len(indices),
                "selection_eta": selection_eta if selection_eta is not None else float(len(order) / len(indices)),
                "actual_size": len(order),
                "kernel_method": None,
                "w_interaction": None,
                "vendi_score": selected_vendi,
                "vendi_score_normalized": selected_vendi / len(order),
                "mean_pairwise_cosine": mean_pairwise_cosine(selected_similarity),
                "log_det": None,
                **summary,
                "baseline_jaccard": None,
                "adjacent_jaccard": None,
                "seed": seed,
                "epsilon": epsilon,
                "solver_backend": "none",
                "device": "cpu",
                "source_fingerprint": source_fingerprint,
                "configuration_fingerprint": configuration_fingerprint,
                "greedy_order": None,
                "marginal_logdet_gains": None,
                "utility_min": utility_min,
                "utility_max": utility_max,
            }
            add_run(run, row_members, [int(index) for index in order.tolist()])
            run_by_config_recording[(config_id, recording)] = current_run_id

    for definition in config.get("baselines", []):
        definition = dict(definition)
        config_id = str(definition["id"])
        scope = str(definition.get("scope", "per_recording"))
        selection_eta = float(definition["selection_eta"])
        baseline_seed = int(definition.get("seed", seed))
        if scope != "per_recording":
            raise AtlasDataError(f"Random baseline {config_id} must use scope=per_recording")
        if not 0.0 < selection_eta <= 1.0:
            raise AtlasDataError(f"Invalid selection_eta for random baseline {config_id}: {selection_eta}")
        for recording in sorted(set(map(int, recording_ids.tolist()))):
            indices = np.flatnonzero(recording_ids == recording)
            selection_k = max(1, min(len(indices), int(np.ceil(selection_eta * len(indices)))))
            generator = np.random.default_rng(baseline_seed + recording)
            order = generator.choice(indices, size=selection_k, replace=False).astype(np.int64)
            selected_losses = losses[order]
            utility = np.ones(selection_k, dtype=np.float64)
            run_configuration = {"stage": "random_stratified", "definition": definition, "recording": recording}
            current_run_id = run_id(source_fingerprint, run_configuration)
            row_members = [
                {
                    "run_id": current_run_id,
                    "row_id": window_ids[int(index)],
                    "rank": rank,
                    "marginal_logdet_gain": None,
                    "utility": 1.0,
                    "recon_loss": float(losses[int(index)]),
                }
                for rank, index in enumerate(order.tolist())
            ]
            summary = summarize_losses(selected_losses, utility)
            selected_similarity = cosine_gram(embeddings[order], epsilon=epsilon)
            selected_vendi = vendi_score(selected_similarity)
            run = {
                "run_id": current_run_id,
                "config_id": config_id,
                "method": "random_stratified",
                "direction": "random",
                "scope": scope,
                "candidate_run_id": None,
                "big_recording_index": recording,
                "candidate_k": len(indices),
                "selection_eta": selection_eta,
                "actual_size": selection_k,
                "kernel_method": None,
                "w_interaction": None,
                "vendi_score": selected_vendi,
                "vendi_score_normalized": selected_vendi / selection_k,
                "mean_pairwise_cosine": mean_pairwise_cosine(selected_similarity),
                "log_det": None,
                **summary,
                "baseline_jaccard": None,
                "adjacent_jaccard": None,
                "seed": baseline_seed,
                "epsilon": epsilon,
                "solver_backend": "none",
                "device": "cpu",
                "source_fingerprint": source_fingerprint,
                "configuration_fingerprint": configuration_fingerprint,
                "greedy_order": None,
                "marginal_logdet_gains": None,
                "utility_min": 1.0,
                "utility_max": 1.0,
            }
            add_run(run, row_members, [int(index) for index in order.tolist()])

    if config.get("dpp"):
        # Large, read-only arrays for forked DPP workers (see _process_dpp_recording).
        _WORKER_DATA["embeddings"] = embeddings
        _WORKER_DATA["losses"] = losses
        _WORKER_DATA["window_ids"] = window_ids
        _WORKER_DATA["recording_ids"] = recording_ids
        _WORKER_DATA["source_indices_by_run"] = source_indices_by_run

    for definition in config.get("dpp", []):
        definition = dict(definition)
        candidate_config_id = str(definition["candidates"])
        selection_eta = float(definition["selection_eta"])
        if not 0.0 < selection_eta <= 1.0:
            raise AtlasDataError(f"Invalid selection_eta for DPP {candidate_config_id}: {selection_eta}")
        requested_recordings = [
            int(value) for value in definition.get("recording_indices", sorted(set(map(int, recording_ids.tolist()))))
        ]
        full_population = bool(definition.get("full_population", True))
        kernel_definitions = dict(definition["kernels"])
        candidate_definition = next(
            (dict(item) for item in config.get("rankings", []) if str(item["id"]) == candidate_config_id), None
        )
        if candidate_definition is None:
            raise AtlasDataError(f"DPP references unknown ranking definition: {candidate_config_id}")
        direction = str(candidate_definition["direction"])
        scope = str(candidate_definition["scope"])

        tasks = []
        for recording in requested_recordings:
            candidate_run_id = run_by_config_recording.get((candidate_config_id, recording))
            if candidate_run_id is None:
                candidate_run_id = run_by_config_recording.get((candidate_config_id, None))
            if candidate_run_id is None:
                raise AtlasDataError(f"No ranking membership for DPP recording {recording} from {candidate_config_id}")
            tasks.append(
                {
                    "recording": recording,
                    "candidate_config_id": candidate_config_id,
                    "selection_eta": selection_eta,
                    "full_population": full_population,
                    "kernel_definitions": kernel_definitions,
                    "direction": direction,
                    "scope": scope,
                    "candidate_run_id": candidate_run_id,
                    "epsilon": epsilon,
                    "max_workspace_gib": max_workspace_gib,
                    "seed": seed,
                    "source_fingerprint": source_fingerprint,
                    "configuration_fingerprint": configuration_fingerprint,
                    "definition": definition,
                }
            )

        # Recordings are fully independent (adjacent_jaccard is computed in a
        # separate pass below), so this is embarrassingly parallel; each task
        # is single-threaded (_dpp_worker_init), so this is the only level of
        # parallelism, avoiding oversubscription.
        max_workers = max(1, min(len(tasks), (os.cpu_count() or 1) - 2, 32))
        if max_workers > 1 and len(tasks) > 1:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=context, initializer=_dpp_worker_init
            ) as executor:
                results = list(executor.map(_process_dpp_recording, tasks))
        else:
            results = [_process_dpp_recording(task) for task in tasks]

        for result in results:
            recording = result["recording"]
            for run, row_members, selected_indices in zip(
                result["runs"], result["row_members"], result["selected_indices"], strict=True
            ):
                add_run(run, row_members, selected_indices)
                dpp_groups.setdefault((candidate_config_id, recording, str(run["kernel_method"])), []).append(
                    (run["w_interaction"], run["run_id"])
                )

    # Adjacent overlap is defined only among runs with the same candidate,
    # recording, and kernel, and is symmetric at the ends of each sweep.
    for group in dpp_groups.values():
        group.sort()
        for position, (_, current_run_id) in enumerate(group):
            neighbors = []
            if position:
                neighbors.append(membership_by_run[current_run_id] & membership_by_run[group[position - 1][1]])
            if position + 1 < len(group):
                neighbors.append(membership_by_run[current_run_id] & membership_by_run[group[position + 1][1]])
            union_sizes = []
            for neighbor_position in (position - 1, position + 1):
                if 0 <= neighbor_position < len(group):
                    other = membership_by_run[group[neighbor_position][1]]
                    union_sizes.append(jaccard(membership_by_run[current_run_id], other))
            next(run for run in runs if run["run_id"] == current_run_id)["adjacent_jaccard"] = (
                float(np.mean(union_sizes)) if union_sizes else None
            )

    output_directory = Path(str(output_config.get("directory", "data/real"))).expanduser()
    selection_table = _cast_rows(selections, _selection_schema())
    run_table = _cast_rows(runs, _run_schema())
    _write_tables_atomically(
        [
            (selection_table, output_directory / "selections.parquet"),
            (run_table, output_directory / "selection_runs.parquet"),
        ],
    )
    manifest_path = output_directory / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"] = {
        "artifacts": {"memberships": "selections.parquet", "runs": "selection_runs.parquet"},
        "rows": len(selections),
        "runs": len(runs),
        "configuration_fingerprint": configuration_fingerprint,
        "source_fingerprint": source_fingerprint,
        "population": str(input_config.get("population", "source")),
        "ranking_runs": sum(run["method"] == "ranking" for run in runs),
        "dpp_runs": sum(run["method"] == "dpp" for run in runs),
    }
    atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    build(config)
    output_directory = Path(str(config.get("output", {}).get("directory", "data/real"))).expanduser()
    print(
        f"Wrote selections to {output_directory / 'selections.parquet'} and {output_directory / 'selection_runs.parquet'}"
    )


if __name__ == "__main__":
    main()
