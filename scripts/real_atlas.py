"""Shared offline implementation for the real-data REVE Embedding Atlas.

The public artifacts intentionally contain no embedding vectors.  This module
keeps the source-table validation, projection, ranking, DPP, and metric logic
in one place so the two offline commands cannot silently disagree.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

REQUIRED_COLUMNS = (
    "window_id",
    "big_recording_index",
    "session_index",
    "offset",
    "recon_loss",
    "embedding",
)
DEFAULT_METADATA_COLUMNS = ("dataset", "modality", "sampling_rate", "n_channels")
WINDOW_ID_RE = re.compile(r"^(?P<recording>-?\d+)_-_(?P<session>-?\d+)_-_(?P<offset>-?\d+)$")


class AtlasDataError(ValueError):
    """Raised for an invalid source or public artifact input."""


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise AtlasDataError(f"Configuration must be a mapping: {path}")
    return config


def source_dataset(path: str | Path) -> pads.Dataset:
    source = Path(path).expanduser()
    if not source.exists():
        raise AtlasDataError(f"Embedding input does not exist: {source}")
    if source.is_dir():
        files = sorted(source.glob("*.parquet"))
        if not files:
            raise AtlasDataError(f"Embedding directory contains no Parquet shards: {source}")
    elif source.suffix.lower() != ".parquet":
        raise AtlasDataError(f"Embedding input must be a Parquet file or shard directory: {source}")
    return pads.dataset(str(source), format="parquet")


def read_source(path: str | Path, columns: Sequence[str] = REQUIRED_COLUMNS) -> pa.Table:
    dataset = source_dataset(path)
    missing = sorted(set(columns) - set(dataset.schema.names))
    if missing:
        raise AtlasDataError(f"Embedding input is missing required columns: {', '.join(missing)}")
    table = dataset.to_table(columns=list(columns), use_threads=True)
    validate_source_table(table)
    return table


def _first_duplicate(values: Iterable[Any]) -> Any | None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def validate_source_table(table: pa.Table) -> None:
    """Validate the exact embeddings-db contract and identifier consistency."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(table.column_names))
    if missing:
        raise AtlasDataError(f"Source table is missing required columns: {', '.join(missing)}")
    if table.num_rows == 0:
        raise AtlasDataError("Source table is empty")
    embedding_type = table.schema.field("embedding").type
    if not pa.types.is_fixed_size_list(embedding_type) or embedding_type.list_size != 512:
        raise AtlasDataError(f"embedding must be fixed-size float32[512], got {embedding_type}")
    if embedding_type.value_type != pa.float32():
        raise AtlasDataError(f"embedding must contain float32 values, got {embedding_type.value_type}")

    window_id_array = table["window_id"].combine_chunks()
    distinct_count = int(pc.count_distinct(window_id_array).as_py())
    if distinct_count != table.num_rows:
        counts = pc.value_counts(window_id_array)
        duplicate = next((row["values"] for row in counts.to_pylist() if row["counts"] > 1), "unknown")
        raise AtlasDataError(f"window_id values are not unique; first duplicate: {duplicate}")
    window_ids = window_id_array.to_pylist()

    for column in ("big_recording_index", "session_index", "offset", "recon_loss"):
        values = table[column].combine_chunks().to_numpy(zero_copy_only=False)
        if not np.all(np.isfinite(values)):
            bad = int(np.flatnonzero(~np.isfinite(values))[0])
            raise AtlasDataError(f"Column {column} contains a non-finite value at row {bad}")

    embeddings = embedding_numpy(table)
    if not np.all(np.isfinite(embeddings)):
        bad = np.argwhere(~np.isfinite(embeddings))[0]
        raise AtlasDataError(f"embedding contains a non-finite value at row {int(bad[0])}, dimension {int(bad[1])}")

    # Arrow performs the identifier extraction in native code.  This keeps
    # the exact consistency check affordable on tens of millions of rows.
    parsed = pc.extract_regex(table["window_id"].combine_chunks(), WINDOW_ID_RE.pattern)
    if parsed.null_count:
        return
    actual_columns = {
        column: table[column].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        for column in ("big_recording_index", "session_index", "offset")
    }
    for field_name, column in (("recording", "big_recording_index"), ("session", "session_index"), ("offset", "offset")):
        expected = pc.cast(parsed.field(field_name), pa.int64()).to_numpy(zero_copy_only=False)
        mismatch = np.flatnonzero(expected != actual_columns[column])
        if len(mismatch):
            row = int(mismatch[0])
            raise AtlasDataError(
                f"window_id {window_ids[row]!r} disagrees with explicit identifier columns "
                f"{tuple(int(actual_columns[name][row]) for name in ("big_recording_index", "session_index", "offset"))}"
            )


def embedding_numpy(table: pa.Table) -> np.ndarray:
    values = table["embedding"].combine_chunks().values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(table.num_rows, 512)


def read_recording_metadata(
    path: str | Path, columns: Sequence[str] = DEFAULT_METADATA_COLUMNS
) -> dict[int, dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise AtlasDataError(f"Recording metadata input does not exist: {source}")
    if source.suffix.lower() == ".csv":
        table = pacsv.read_csv(source)
    else:
        table = pq.read_table(source)
    required = ("big_recording_index", *columns)
    missing = sorted(set(required) - set(table.column_names))
    if missing:
        raise AtlasDataError(f"Recording metadata is missing required columns: {', '.join(missing)}")
    selected = table.select(list(required)).to_pylist()
    metadata: dict[int, dict[str, Any]] = {}
    for row in selected:
        recording = int(row["big_recording_index"])
        if recording in metadata:
            raise AtlasDataError(f"Recording metadata contains duplicate big_recording_index={recording}")
        public = {column: row[column] for column in columns}
        for column, value in public.items():
            if value is None:
                raise AtlasDataError(f"Recording metadata has null {column} for big_recording_index={recording}")
        metadata[recording] = public
    return metadata


def join_metadata(table: pa.Table, metadata: dict[int, dict[str, Any]], columns: Sequence[str]) -> pa.Table:
    recordings = table["big_recording_index"].combine_chunks().to_pylist()
    missing = sorted({int(recording) for recording in recordings} - set(metadata))
    if missing:
        preview = ", ".join(map(str, missing[:12]))
        raise AtlasDataError(f"Recording metadata has no rows for recording indices: {preview}")
    arrays = [table[column] for column in table.column_names]
    fields = list(table.schema)
    for column in columns:
        values = [metadata[int(recording)][column] for recording in recordings]
        arrays.append(pa.array(values))
        fields.append(pa.field(column, arrays[-1].type))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _hash_update(hasher: hashlib._Hash, value: str) -> None:
    hasher.update(value.encode("utf-8"))
    hasher.update(b"\0")


def fingerprint_path(path: str | Path) -> str:
    """Return a deterministic, cheap source fingerprint.

    For large Parquet trees the fingerprint uses relative names, byte sizes,
    modification times, and schema text rather than rereading tens of GB.
    """
    source = Path(path).expanduser().resolve()
    hasher = hashlib.sha256()
    if source.is_file():
        files = [source]
        root = source.parent
    else:
        files = sorted(source.glob("*.parquet"))
        root = source
    for file in files:
        stat = file.stat()
        _hash_update(hasher, str(file.relative_to(root)))
        _hash_update(hasher, str(stat.st_size))
        _hash_update(hasher, str(stat.st_mtime_ns))
        try:
            _hash_update(hasher, str(pq.read_schema(file)))
        except Exception:
            pass
    return hasher.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def run_id(source_fingerprint: str, configuration: dict[str, Any]) -> str:
    payload = canonical_json({"source_fingerprint": source_fingerprint, "configuration": configuration})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def normalize_utility(losses: np.ndarray, direction: str) -> tuple[np.ndarray, float, float]:
    values = np.asarray(losses, dtype=np.float64)
    oriented = values if direction == "top" else -values
    lower = float(np.min(oriented))
    upper = float(np.max(oriented))
    if upper == lower:
        utility = np.ones_like(values, dtype=np.float64)
    else:
        utility = (oriented - lower) / (upper - lower)
    return utility, lower, upper


def cosine_gram(embeddings: np.ndarray, epsilon: float = 1.0e-6) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    bad = np.flatnonzero(~np.isfinite(norms) | (norms <= 0))
    if len(bad):
        raise AtlasDataError(f"zero-norm or non-finite embedding at candidate row {int(bad[0])}")
    normalized = values / norms[:, None]
    similarity = normalized @ normalized.T
    similarity = (similarity + similarity.T) * 0.5
    similarity[np.diag_indices_from(similarity)] += epsilon
    return similarity


def vendi_score(similarity: np.ndarray, tolerance: float = 1.0e-7) -> float:
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise AtlasDataError("Vendi requires a non-empty square similarity matrix")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
    materially_negative = eigenvalues[eigenvalues < -tolerance]
    if len(materially_negative):
        raise AtlasDataError(f"cosine similarity has materially negative eigenvalues: {materially_negative[:3]}")
    probabilities = np.clip(eigenvalues / matrix.shape[0], 0.0, None)
    total = float(np.sum(probabilities))
    if not math.isfinite(total) or total <= 0:
        raise AtlasDataError("Vendi eigenvalue normalization is non-finite")
    probabilities /= total
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])))
    score = float(np.exp(entropy))
    if not math.isfinite(score):
        raise AtlasDataError("Vendi score is non-finite")
    return min(float(matrix.shape[0]), max(1.0, score))


def mean_pairwise_cosine(similarity: np.ndarray) -> float:
    """Return the mean off-diagonal cosine similarity for a selected set."""
    matrix = np.asarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise AtlasDataError("Mean pairwise cosine requires a non-empty square similarity matrix")
    size = matrix.shape[0]
    if size == 1:
        return 1.0
    off_diagonal_sum = float(np.sum(matrix) - np.trace(matrix))
    score = off_diagonal_sum / float(size * (size - 1))
    if not math.isfinite(score):
        raise AtlasDataError("Mean pairwise cosine is non-finite")
    return score


def build_kernel(
    similarity: np.ndarray, utility: np.ndarray, method: str, interaction: float, epsilon: float
) -> np.ndarray:
    if method == "multiplicative":
        centered = interaction * (utility - float(np.max(utility)))
        diagonal = np.exp(centered)
        kernel = diagonal[:, None] * similarity * diagonal[None, :]
    elif method == "additive":
        kernel = np.diag(utility + epsilon) + interaction * similarity
    else:
        raise AtlasDataError(f"Unknown DPP kernel method: {method}")
    return (kernel + kernel.T) * 0.5


def _stable_marginal(kernel: np.ndarray, selected: list[int], candidate: int, epsilon: float) -> float:
    if not selected:
        return max(float(kernel[candidate, candidate]), epsilon)
    sub = kernel[np.ix_(selected, selected)]
    sub = (sub + sub.T) * 0.5
    sub[np.diag_indices_from(sub)] += epsilon
    cross = kernel[np.ix_(selected, [candidate])].reshape(-1)
    try:
        correction = float(cross @ np.linalg.solve(sub, cross))
    except np.linalg.LinAlgError:
        correction = float(cross @ np.linalg.pinv(sub) @ cross)
    return max(float(kernel[candidate, candidate]) - correction, epsilon)


def _tie_break_winner(scores: np.ndarray, tie_order: np.ndarray | None, tie_tolerance: float | None) -> int:
    """argmax, or among near-maximal scores (>= max*(1-tol)) the one with the smallest tie_order."""
    winner = int(np.argmax(scores))
    if tie_order is None or tie_tolerance is None:
        return winner
    best = float(scores[winner])
    tied = np.flatnonzero(scores >= best - abs(best) * tie_tolerance)
    return int(tied[np.argmin(tie_order[tied])]) if len(tied) > 1 else winner


def _greedy_select(
    diagonal: np.ndarray,
    row_fn: Callable[[int], np.ndarray],
    selection_k: int,
    epsilon: float,
    tie_order: np.ndarray | None = None,
    tie_tolerance: float | None = None,
) -> tuple[list[int], list[float]]:
    """Core single-start incremental-Cholesky greedy selection.

    ``row_fn(i)`` must return a fresh, safely-mutable float64 array holding
    kernel row ``i`` (length ``len(diagonal)``) — this function updates it
    in place, so a row_fn backed by a view into shared storage must copy
    before returning. It is called once per selection step, so it never needs
    the full n x n kernel materialized up front (see ``greedy_map_lazy``,
    which computes rows on demand from the similarity matrix instead).

    With ``tie_order`` (e.g. window-id ranks) and ``tie_tolerance``, candidates whose
    residual is within ``tie_tolerance`` (relative) of the maximum are resolved by the
    smallest ``tie_order``, so the greedy path does not depend on rounding noise.
    """
    candidate_count = int(diagonal.shape[0])
    if not 0 < selection_k <= candidate_count:
        raise AtlasDataError(f"selection_k={selection_k} must be in [1, {candidate_count}]")
    residual = np.asarray(diagonal, dtype=np.float64).copy()
    factors = np.zeros((selection_k, candidate_count), dtype=np.float64)
    selected_mask = np.zeros(candidate_count, dtype=bool)
    selected: list[int] = []
    gains: list[float] = []
    for step in range(selection_k):
        scores = np.where(selected_mask, -np.inf, residual)
        winner = _tie_break_winner(scores, tie_order, tie_tolerance)
        pivot_value = max(float(residual[winner]), epsilon)
        pivot = math.sqrt(pivot_value)
        row = np.asarray(row_fn(winner), dtype=np.float64)
        if step:
            row -= factors[:step, :].T @ factors[:step, winner]
        row /= pivot
        factors[step, :] = row
        residual -= row * row
        selected_mask[winner] = True
        selected.append(winner)
        gains.append(float(np.log(max(pivot * pivot, epsilon))))
    return selected, gains


def greedy_map(kernel: np.ndarray, selection_k: int, epsilon: float) -> tuple[list[int], list[float], float]:
    """Single-start greedy MAP-DPP selection using incremental Cholesky updates.

    O(n*k^2) via rank-1 residual updates (Chen, Zhang & Avron 2018,
    https://github.com/laming-chen/fast-map-dpp), matching the intent of the
    single-start reference in
    https://github.com/jonathanlys01/d3p2/blob/main/src/d5p4/subsample/_greedy_map.py
    (that file's own ``fast_greedy_map`` masks the pivot to -inf before reading
    it back as the normalizer on the next step, which poisons every
    subsequent update with NaN; it is only ever exercised from a ``timeit``
    call, never checked for correctness, so this reimplements the same
    algorithm without that bug rather than porting it verbatim).
    """
    candidate_count = int(kernel.shape[0])
    if kernel.ndim != 2 or kernel.shape[1] != candidate_count:
        raise AtlasDataError("DPP kernel must be square")
    kernel = np.asarray(kernel, dtype=np.float64)
    selected, gains = _greedy_select(np.diag(kernel), lambda i: kernel[i, :].copy(), selection_k, epsilon)
    final = kernel[np.ix_(selected, selected)]
    sign, logdet = np.linalg.slogdet((final + final.T) * 0.5 + epsilon * np.eye(selection_k))
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise AtlasDataError("DPP kernel produced a non-positive final determinant")
    return selected, gains, float(logdet)


def greedy_map_lazy(
    similarity: np.ndarray,
    utility: np.ndarray,
    method: str,
    interaction: float,
    selection_k: int,
    epsilon: float,
    tie_order: np.ndarray | None = None,
    tie_tolerance: float | None = None,
) -> tuple[list[int], list[float], float]:
    """Same result as ``greedy_map(build_kernel(similarity, utility, method, interaction, epsilon), ...)``,
    without ever materializing the full n x n kernel matrix.

    ``build_kernel`` is O(n^2) and was being rebuilt from scratch once per
    sweep point (31 times per recording) even though the greedy selection only
    ever reads it one row at a time. Both kernel methods are simple, cheaply
    row-computable transforms of ``similarity``, so this computes rows on
    demand instead; the final log-determinant still only needs the small
    selected-by-selected submatrix, built with the existing ``build_kernel``
    at O(k^2) cost.
    """
    if method == "multiplicative":
        weights = np.exp(float(interaction) * (utility - float(np.max(utility))))
        diagonal = (weights**2) * np.diag(similarity)

        def row_fn(i: int) -> np.ndarray:
            return weights[i] * similarity[i, :] * weights
    elif method == "additive":
        diagonal = utility + epsilon + float(interaction) * np.diag(similarity)

        def row_fn(i: int) -> np.ndarray:
            row = float(interaction) * similarity[i, :]
            row[i] += utility[i] + epsilon
            return row
    else:
        raise AtlasDataError(f"Unknown DPP kernel method: {method}")

    selected, gains = _greedy_select(diagonal, row_fn, selection_k, epsilon, tie_order, tie_tolerance)
    final = build_kernel(similarity[np.ix_(selected, selected)], utility[selected], method, interaction, epsilon)
    sign, logdet = np.linalg.slogdet((final + final.T) * 0.5 + epsilon * np.eye(selection_k))
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise AtlasDataError("DPP kernel produced a non-positive final determinant")
    return selected, gains, float(logdet)


def workspace_bytes(candidate_k: int, selection_k: int, embedding_dim: int = 512) -> int:
    # Source embeddings, normalized embeddings, Gram matrix, and temporary
    # Cholesky workspaces.  This is intentionally conservative.
    return int(
        candidate_k * embedding_dim * 4 * 2 + candidate_k * candidate_k * 8 * 3 + selection_k * selection_k * 8 * 2
    )


def ensure_workspace(candidate_k: int, selection_k: int, max_workspace_gib: float) -> None:
    required = workspace_bytes(candidate_k, selection_k)
    budget = float(max_workspace_gib) * (1024**3)
    if required > budget:
        raise AtlasDataError(
            f"DPP workspace preflight rejected candidate_k={candidate_k}, selection_k={selection_k}: "
            f"estimated {required / 1024**3:.2f} GiB exceeds {max_workspace_gib:.2f} GiB"
        )


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {f"q{q}": float(np.quantile(values, q / 100.0)) for q in (1, 5, 25, 50, 75, 95, 99)}


def summarize_losses(losses: np.ndarray, utility: np.ndarray) -> dict[str, float]:
    values = np.asarray(losses, dtype=np.float64)
    summary = {
        "mean_recon_loss": float(np.mean(values)),
        "median_recon_loss": float(np.median(values)),
        "min_recon_loss": float(np.min(values)),
        "max_recon_loss": float(np.max(values)),
        "mean_utility": float(np.mean(utility)),
    }
    summary.update({f"recon_loss_{key}": value for key, value in quantiles(values).items()})
    return summary


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def runtime_provenance() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pyarrow", "umap", "sklearn", "cuml"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "available"))
        except Exception:
            versions[name] = "unavailable"
    return versions


def _interpolate_projection(
    embeddings: np.ndarray,
    fit_values: np.ndarray,
    fit_indices: np.ndarray,
    fit_projection: np.ndarray,
    projection: dict[str, Any],
    device_requested: str,
) -> tuple[np.ndarray, str]:
    """Extend a sampled UMAP embedding with deterministic cosine kNN interpolation."""
    chunk_size = int(projection.get("transform_chunk_size", 50_000))
    neighbor_count = min(int(projection.get("interpolation_neighbors", 8)), len(fit_values))
    temperature = float(projection.get("interpolation_temperature", 12.0))
    fit_norms = np.linalg.norm(fit_values.astype(np.float32), axis=1)
    fit_normalized = fit_values.astype(np.float32) / fit_norms[:, None]
    output = np.empty((len(embeddings), 2), dtype=np.float32)
    output[fit_indices] = fit_projection
    remaining = np.setdiff1d(np.arange(len(embeddings)), fit_indices, assume_unique=True)
    backend = "numpy-cosine-knn-interpolation"
    torch = None
    if device_requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                fit_tensor = torch.from_numpy(fit_normalized).to("cuda")
                projection_tensor = torch.from_numpy(fit_projection).to("cuda")
                backend = "torch-cuda-cosine-knn-interpolation"
            else:
                torch = None
        except ImportError:
            torch = None
    for start in range(0, len(remaining), chunk_size):
        batch_indices = remaining[start : start + chunk_size]
        batch = embeddings[batch_indices].astype(np.float32)
        norms = np.linalg.norm(batch, axis=1)
        normalized = batch / norms[:, None]
        if torch is not None and backend.startswith("torch"):
            batch_tensor = torch.from_numpy(normalized).to("cuda")
            scores = batch_tensor @ fit_tensor.T
            values, positions = torch.topk(scores, k=neighbor_count, dim=1)
            weights = torch.softmax(values * temperature, dim=1)
            output[batch_indices] = (weights.unsqueeze(-1) * projection_tensor[positions]).sum(dim=1).cpu().numpy()
        else:
            scores = normalized @ fit_normalized.T
            positions = np.argpartition(scores, -neighbor_count, axis=1)[:, -neighbor_count:]
            values = np.take_along_axis(scores, positions, axis=1)
            weights = np.exp((values - values.max(axis=1, keepdims=True)) * temperature)
            weights /= weights.sum(axis=1, keepdims=True)
            output[batch_indices] = (weights[..., None] * fit_projection[positions]).sum(axis=1)
    return output, backend


def project_embeddings(embeddings: np.ndarray, projection: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    method = str(projection.get("method", "umap")).lower()
    device_requested = str(projection.get("device", "cuda")).lower()
    seed = int(projection.get("seed", 0))
    fit_sample_size = int(projection.get("fit_sample_size", min(len(embeddings), 200_000)))
    fit_sample_size = min(len(embeddings), max(2, fit_sample_size))
    rng = np.random.default_rng(seed)
    fit_indices = np.sort(rng.choice(len(embeddings), size=fit_sample_size, replace=False))
    fit_values = np.asarray(embeddings[fit_indices], dtype=np.float32)
    embedding_preprocessing = "none"
    if str(projection.get("metric", "euclidean")).lower() == "cosine":
        fit_norms = np.linalg.norm(fit_values, axis=1)
        if not np.all(np.isfinite(fit_norms)) or np.any(fit_norms <= 0):
            raise AtlasDataError("Projection fit sample contains zero-norm or non-finite embeddings")
        fit_values = fit_values / fit_norms[:, None]
        embedding_preprocessing = "l2_normalize_for_cosine"
    backend = ""
    device = "cpu"
    parameters = {
        key: value
        for key, value in projection.items()
        if key not in {"method", "seed", "device", "fit_sample_size", "transform_chunk_size", "out_of_sample", "interpolation_neighbors", "interpolation_temperature", "parallel"}
    }

    parallel = bool(projection.get("parallel", False))
    if method == "umap":
        try:
            if device_requested == "cuda":
                from cuml import UMAP  # type: ignore

                # cuML's `init` only accepts 'spectral'/'random'/a precomputed array, unlike
                # umap-learn's 'pca' string; compute the PCA embedding ourselves and keep the
                # human-readable "pca" tag (not the raw array) in `parameters` for provenance.
                cuml_parameters = dict(parameters)
                cuml_parameters["random_state"] = seed
                if str(parameters.get("init", "")).lower() == "pca":
                    from cuml.decomposition import PCA  # type: ignore

                    pca_embedding = np.asarray(
                        PCA(n_components=2, output_type="numpy").fit_transform(fit_values), dtype=np.float32
                    )
                    # Match umap-learn's own PCA-init convention (umap.umap_.noisy_scale_coords):
                    # rescale so the largest coordinate is 10, plus a touch of noise to break
                    # exact ties. Raw PCA output on L2-normalized embeddings is an order of
                    # magnitude smaller than the scale UMAP's spread/min_dist dynamics assume,
                    # which otherwise leaves most points unable to separate within n_epochs.
                    max_coord = float(np.max(np.abs(pca_embedding)))
                    if max_coord > 0:
                        pca_embedding *= 10.0 / max_coord
                    pca_embedding += np.random.default_rng(seed).normal(
                        scale=0.0001, size=pca_embedding.shape
                    ).astype(np.float32)
                    cuml_parameters["init"] = pca_embedding
                model = UMAP(**cuml_parameters)
                projected = model.fit_transform(fit_values)
                backend = "cuml"
                device = "cuda"
            else:
                raise ImportError
        except ImportError:
            try:
                import umap
            except ImportError as error:
                raise AtlasDataError("UMAP requested but neither cuML nor umap-learn is installed") from error
            if not parallel:
                parameters["random_state"] = seed
            model = umap.UMAP(**parameters)
            projected = model.fit_transform(fit_values)
            backend = "umap-learn"
            device = "cpu"
    elif method == "tsne":
        if fit_sample_size != len(embeddings):
            raise AtlasDataError("t-SNE requires projection.fit_sample_size equal to the full input row count")
        try:
            from cuml import TSNE  # type: ignore

            model = TSNE(**parameters)
            projected = model.fit_transform(fit_values)
            backend = "cuml"
            device = "cuda"
        except ImportError:
            try:
                from sklearn.manifold import TSNE
            except ImportError as error:
                raise AtlasDataError("t-SNE requested but neither cuML nor scikit-learn is installed") from error
            parameters["random_state"] = seed
            model = TSNE(**parameters)
            projected = model.fit_transform(fit_values)
            backend = "sklearn"
            device = "cpu"
    else:
        raise AtlasDataError(f"Unknown projection method: {method}")

    projected = np.asarray(projected, dtype=np.float32)
    if fit_sample_size != len(embeddings):
        if str(projection.get("out_of_sample", "nearest_neighbor")) != "nearest_neighbor":
            if not hasattr(model, "transform"):
                raise AtlasDataError(f"Projection backend {backend} cannot transform the non-fit rows")
            output = np.empty((len(embeddings), 2), dtype=np.float32)
            output[fit_indices] = projected
            chunk_size = int(projection.get("transform_chunk_size", 50_000))
            remaining = np.setdiff1d(np.arange(len(embeddings)), fit_indices, assume_unique=True)
            for start in range(0, len(remaining), chunk_size):
                batch_indices = remaining[start : start + chunk_size]
                output[batch_indices] = np.asarray(model.transform(embeddings[batch_indices]), dtype=np.float32)
            projected = output
        else:
            projected, interpolation_backend = _interpolate_projection(
                embeddings, fit_values, fit_indices, projected, projection, device_requested
            )
            backend = f"{backend}+{interpolation_backend}"
    if projected.shape != (len(embeddings), 2) or not np.all(np.isfinite(projected)):
        raise AtlasDataError("Projection returned non-finite coordinates or an invalid shape")
    provenance = {
        "method": method,
        "backend": backend,
        "device": device,
        "requested_device": device_requested,
        "seed": seed,
        "fit_sample_size": fit_sample_size,
        "transform_chunk_size": int(projection.get("transform_chunk_size", 50_000)),
        "out_of_sample": str(projection.get("out_of_sample", "nearest_neighbor")),
        "interpolation_neighbors": int(projection.get("interpolation_neighbors", 8)),
        "interpolation_temperature": float(projection.get("interpolation_temperature", 12.0)),
        "embedding_preprocessing": embedding_preprocessing,
        "parallel": parallel,
        "parameters": parameters,
        "packages": runtime_provenance(),
    }
    return projected, provenance


def atomic_write_table(table: pa.Table, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pq.write_table(table, temporary_path, compression="zstd", row_group_size=100_000)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, destination)
