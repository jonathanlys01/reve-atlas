#!/usr/bin/env python
"""Gate 1b: cardinality-controlled recursive DBSCAN groups (and the k-means control).

Groups the display population in the L2-normalized 512-d embedding space so every group has
``n_min <= size <= n_max``. See ``_scratchpad/hierarchical_dpp_clusters_handoff.md``.

node(idx, eps): leaf if |idx| <= n_max; otherwise shrink eps by gamma until DBSCAN yields
>= 2 viable (>= n_min) clusters holding >= viable_frac of the node with the largest below
largest_frac; attach the rest to the nearest viable row and recurse with eps*gamma. If eps
reaches eps_floor, fall back to spherical k-means (leaves marked ``kmeans``). Small leaves are
merged into the nearest leaf (centroid cosine) with room; the cap is enforced after merging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

CHUNK_ROWS = 8192
TARGET_CHUNK = 262144


@dataclass
class Params:
    n_max: int
    n_min: int
    eps0: float = 0.0
    min_samples: int = 10
    gamma: float = 0.85
    viable_frac: float = 0.5
    largest_frac: float = 0.97
    eps_floor_divisor: float = 64.0
    max_mbytes_per_batch: int = 16000
    kmeans_fallback_fill: float = 0.75
    kmeans_iterations: int = 50
    seed: int = 20260922
    mode: str = "dbscan"  # dbscan | kmeans_control
    control_fill: float = 0.6
    # Skip eps levels whose exact clustered fraction (core + border rows, see
    # hier_dbscan_prune) is below viable_frac: such a level can never be accepted.
    prune: bool = True


@dataclass
class Leaf:
    indices: np.ndarray  # ascending source positions
    path: str
    depth: int
    split: str


@dataclass
class Stats:
    dbscan_calls: int = 0
    dbscan_cache_hits: int = 0
    dbscan_seconds: float = 0.0
    kmeans_fallback_nodes: int = 0
    attach_seconds: float = 0.0
    pruned_levels: int = 0
    prune_seconds: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)


class GpuMemoryPeak:
    """Samples device memory in use (nvidia-smi) while the grouping runs."""

    def __init__(self, interval: float = 1.0) -> None:
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)

    def _run(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, check=False, timeout=10,
                ).stdout
                self.peak_mib = max(self.peak_mib, int(out.split()[0]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass

    def __enter__(self) -> GpuMemoryPeak:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 0..C-1 by first (smallest) position; noise stays -1."""
    out = np.full(labels.shape, -1, dtype=np.int64)
    valid = labels >= 0
    if not valid.any():
        return out
    values, first = np.unique(labels[valid], return_index=True)
    order = np.argsort(first, kind="stable")
    mapping = np.empty(values.max() + 1, dtype=np.int64)
    mapping[values[order]] = np.arange(len(values))
    out[valid] = mapping[labels[valid]]
    return out


class Grouper:
    def __init__(
        self,
        unit: torch.Tensor,
        params: Params,
        cache_dir: Path | None,
        log: Any = print,
        root_reach: np.ndarray | None = None,
    ) -> None:
        self.unit = unit
        self.root_reach = root_reach
        self.params = params
        self.cache_dir = cache_dir
        self.log = log
        self.stats = Stats()
        count = unit.shape[0]
        self.attached = np.zeros(count, dtype=bool)
        self.leaves: list[Leaf] = []

    # ------------------------------------------------------------------ DBSCAN
    def dbscan(self, indices: np.ndarray, eps: float) -> np.ndarray:
        m = self.params.min_samples
        key = hashlib.sha1(indices.astype(np.int64).tobytes()).hexdigest()[:20]
        cache_path = None if self.cache_dir is None else self.cache_dir / f"dbscan_{key}_{eps:.10f}_{m}.npy"
        if cache_path is not None and cache_path.exists():
            self.stats.dbscan_cache_hits += 1
            return np.load(cache_path)
        import cupy as cp
        from cuml.cluster import DBSCAN

        start = time.time()
        subset = cp.from_dlpack(self.unit[torch.as_tensor(indices, device=self.unit.device)])
        labels = DBSCAN(
            eps=float(eps), min_samples=m, max_mbytes_per_batch=self.params.max_mbytes_per_batch
        ).fit_predict(subset)
        labels = canonical_labels(cp.asnumpy(labels).astype(np.int64))
        del subset
        cp.get_default_memory_pool().free_all_blocks()
        elapsed = time.time() - start
        self.stats.dbscan_calls += 1
        self.stats.dbscan_seconds += elapsed
        self.stats.calls.append({"n": int(len(indices)), "eps": float(eps), "seconds": elapsed})
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, labels)
        return labels

    # ------------------------------------------------------------ geometry ops
    def nearest(self, queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """For each query position, the position (into ``targets``) of its max-cosine target.

        Exact fp32 (TF32 off); ties resolve to the earliest (smallest source index) target.
        """
        device = self.unit.device
        target_tensor = torch.as_tensor(targets, device=device)
        best_value = torch.full((len(queries),), -torch.inf, device=device)
        best_index = torch.zeros(len(queries), dtype=torch.long, device=device)
        query_tensor = torch.as_tensor(queries, device=device)
        for t0 in range(0, len(targets), TARGET_CHUNK):
            block = self.unit[target_tensor[t0 : t0 + TARGET_CHUNK]]
            for q0 in range(0, len(queries), CHUNK_ROWS):
                sims = self.unit[query_tensor[q0 : q0 + CHUNK_ROWS]] @ block.T
                value, index = sims.max(dim=1)
                better = value > best_value[q0 : q0 + CHUNK_ROWS]
                best_value[q0 : q0 + CHUNK_ROWS] = torch.where(better, value, best_value[q0 : q0 + CHUNK_ROWS])
                best_index[q0 : q0 + CHUNK_ROWS] = torch.where(better, index + t0, best_index[q0 : q0 + CHUNK_ROWS])
        return best_index.cpu().numpy()

    def assign(self, indices: np.ndarray, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.as_tensor(indices, device=self.unit.device)
        labels = torch.empty(len(indices), dtype=torch.long, device=self.unit.device)
        values = torch.empty(len(indices), dtype=self.unit.dtype, device=self.unit.device)
        for q0 in range(0, len(indices), CHUNK_ROWS):
            sims = self.unit[rows[q0 : q0 + CHUNK_ROWS]] @ centroids.T
            values[q0 : q0 + CHUNK_ROWS], labels[q0 : q0 + CHUNK_ROWS] = sims.max(dim=1)
        return labels, values

    def spherical_kmeans(self, indices: np.ndarray, k: int, seed: int) -> np.ndarray:
        """Deterministic spherical k-means (k-means++ init on 1-cos). Returns labels."""
        device = self.unit.device
        n = len(indices)
        k = max(1, min(k, n))
        rows = torch.as_tensor(indices, device=device)
        data = self.unit[rows]
        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(n))]
        closest = torch.clamp(1.0 - data @ data[chosen[0]], min=0.0)
        for _ in range(1, k):
            weights = closest.double().cpu().numpy()
            total = weights.sum()
            if total <= 0:  # fewer distinct points than k: take the next unused positions
                unused = np.setdiff1d(np.arange(n), chosen)
                pick = int(unused[0])
            else:
                pick = int(rng.choice(n, p=weights / total))
            chosen.append(pick)
            closest = torch.minimum(closest, torch.clamp(1.0 - data @ data[pick], min=0.0))
        centroids = data[torch.as_tensor(chosen, device=device)].clone()
        labels = None
        for _ in range(self.params.kmeans_iterations):
            new_labels, values = self.assign(indices, centroids)
            if labels is not None and torch.equal(new_labels, labels):
                break
            labels = new_labels
            sums = torch.zeros_like(centroids).index_add_(0, labels, data)
            counts = torch.bincount(labels, minlength=k)
            empty = torch.nonzero(counts == 0).flatten()
            if len(empty):  # reseed empty clusters at the worst-fit points
                worst = torch.argsort(values, stable=True)[: len(empty)]
                sums[empty] = data[worst]
            centroids = torch.nn.functional.normalize(sums, dim=1)
        return labels.cpu().numpy().astype(np.int64)

    def bisect(self, indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Balanced 2-means split; both halves have >= n_min rows (requires |idx| >= 2*n_min)."""
        n = len(indices)
        labels = self.spherical_kmeans(indices, 2, seed)
        data = self.unit[torch.as_tensor(indices, device=self.unit.device)]
        centroids = []
        for label in (0, 1):
            members = torch.as_tensor(np.flatnonzero(labels == label), device=self.unit.device)
            centroid = data[members].sum(0) if len(members) else data[0]
            centroids.append(torch.nn.functional.normalize(centroid, dim=0))
        margin = (data @ centroids[0] - data @ centroids[1]).double().cpu().numpy()
        order = np.lexsort((indices, -margin))  # most-c0 first, ties by source index
        low = min(self.params.n_min, n // 2)
        first = int(np.clip(int(np.sum(labels == 0)), low, n - low))
        return np.sort(indices[order[:first]]), np.sort(indices[order[first:]])

    def kmeans_parts(self, indices: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
        labels = self.spherical_kmeans(indices, k, seed)
        parts = [np.sort(indices[labels == label]) for label in np.unique(labels)]
        out: list[np.ndarray] = []
        stack = sorted(parts, key=lambda part: int(part[0]), reverse=True)
        while stack:
            part = stack.pop()
            if len(part) <= self.params.n_max:
                out.append(part)
                continue
            left, right = self.bisect(part, seed + len(part))
            stack.extend(sorted([left, right], key=lambda p: int(p[0]), reverse=True))
        return sorted(out, key=lambda part: int(part[0]))

    # --------------------------------------------------------------- recursion
    def node(self, indices: np.ndarray, eps: float, depth: int, path: str) -> None:
        p = self.params
        size = len(indices)
        if size <= p.n_max:
            self.leaves.append(Leaf(indices, path, depth, "root" if depth == 0 else "dbscan"))
            return
        eps_floor = p.eps0 / p.eps_floor_divisor
        reach = None
        if p.prune:
            if depth == 0 and self.root_reach is not None:
                reach = self.root_reach
            else:
                from scripts.hier_dbscan_prune import reach_distances

                start = time.time()
                reach = reach_distances(self.unit, indices, p.min_samples)
                self.stats.prune_seconds += time.time() - start
        current = eps
        while current >= eps_floor:
            if reach is not None and float(np.mean(reach <= current)) < p.viable_frac:
                self.stats.pruned_levels += 1
                self.log(f"  node {path} n={size} eps={current:.5f} clustered_frac={float(np.mean(reach <= current)):.3f} PRUNED")
                current *= p.gamma
                continue
            labels = self.dbscan(indices, current)
            clusters, counts = np.unique(labels[labels >= 0], return_counts=True)
            viable = clusters[counts >= p.n_min]
            viable_rows = int(counts[counts >= p.n_min].sum())
            largest = int(counts.max()) if len(counts) else 0
            accepted = len(viable) >= 2 and viable_rows >= p.viable_frac * size and largest < p.largest_frac * size
            self.log(
                f"  node {path} n={size} eps={current:.5f} clusters={len(clusters)} viable={len(viable)} "
                f"viable_frac={viable_rows / size:.3f} largest={largest / size:.3f} {'ACCEPT' if accepted else ''}"
            )
            if accepted:
                child_of = np.full(size, -1, dtype=np.int64)
                is_viable = np.isin(labels, viable)
                remap = {int(cluster): position for position, cluster in enumerate(viable.tolist())}
                child_of[is_viable] = [remap[int(value)] for value in labels[is_viable]]
                loose = np.flatnonzero(~is_viable)
                if len(loose):
                    start = time.time()
                    targets = np.flatnonzero(is_viable)
                    nearest = self.nearest(indices[loose], indices[targets])
                    child_of[loose] = child_of[targets[nearest]]
                    self.attached[indices[loose]] = True
                    self.stats.attach_seconds += time.time() - start
                for child in range(len(viable)):
                    self.node(np.sort(indices[child_of == child]), current * p.gamma, depth + 1, f"{path}.{child}")
                return
            current *= p.gamma
        self.stats.kmeans_fallback_nodes += 1
        k = math.ceil(size / (p.kmeans_fallback_fill * p.n_max))
        for position, part in enumerate(self.kmeans_parts(indices, k, p.seed + depth)):
            self.leaves.append(Leaf(part, f"{path}.k{position}", depth + 1, "kmeans"))

    # ---------------------------------------------------------- post-process
    def merge_small(self) -> None:
        """Merge leaves below n_min into the nearest leaf (centroid cosine) that stays <= n_max."""
        p = self.params
        device = self.unit.device
        sums = torch.stack([self.unit[torch.as_tensor(leaf.indices, device=device)].double().sum(0) for leaf in self.leaves])
        sizes = np.asarray([len(leaf.indices) for leaf in self.leaves], dtype=np.int64)
        alive = np.ones(len(self.leaves), dtype=bool)
        merged = np.zeros(len(self.leaves), dtype=bool)
        members: list[list[np.ndarray]] = [[leaf.indices] for leaf in self.leaves]
        extra_leaves: list[Leaf] = []

        def queue_order() -> list[int]:
            small = np.flatnonzero(alive & (sizes < p.n_min))
            return sorted(small.tolist(), key=lambda i: (int(sizes[i]), int(self.leaves[i].indices[0])))

        pending = queue_order()
        while pending:
            source = pending[0]
            centroids = torch.nn.functional.normalize(sums, dim=1)
            similarity = (centroids @ centroids[source]).cpu().numpy()
            similarity[~alive] = -np.inf
            similarity[source] = -np.inf
            order = np.lexsort((np.arange(len(sizes)), -similarity))
            order = order[np.isfinite(similarity[order])]
            fits = order[sizes[order] + sizes[source] <= p.n_max]
            target = int(fits[0]) if len(fits) else int(order[0])
            members[target].extend(members[source])
            sums[target] += sums[source]
            sizes[target] += sizes[source]
            alive[source] = False
            merged[target] = True
            if sizes[target] > p.n_max:  # no leaf had room: bisect the merged result
                combined = np.sort(np.concatenate(members[target]))
                left, right = self.bisect(combined, p.seed + int(combined[0]))
                leaf = self.leaves[target]
                alive[target] = False
                for half in (left, right):
                    extra_leaves.append(Leaf(half, leaf.path, leaf.depth, leaf.split + "+merged"))
            pending = queue_order()
        final: list[Leaf] = []
        for position, leaf in enumerate(self.leaves):
            if not alive[position]:
                continue
            combined = np.sort(np.concatenate(members[position]))
            final.append(Leaf(combined, leaf.path, leaf.depth, leaf.split + ("+merged" if merged[position] else "")))
        final.extend(extra_leaves)
        self.leaves = final

    def run(self) -> None:
        count = self.unit.shape[0]
        everything = np.arange(count, dtype=np.int64)
        if self.params.mode == "kmeans_control":
            k = round(count / (self.params.control_fill * self.params.n_max))
            labels = self.spherical_kmeans(everything, k, self.params.seed)
            for label in np.unique(labels):
                part = everything[labels == label]
                for position, piece in enumerate(
                    self.kmeans_parts(part, 1, self.params.seed + int(label)) if len(part) > self.params.n_max else [part]
                ):
                    self.leaves.append(Leaf(piece, f"{int(label)}" + (f".b{position}" if len(part) > self.params.n_max else ""), 1, "kmeans"))
        else:
            self.node(everything, self.params.eps0, 0, "0")
        self.origin_split = np.empty(count, dtype=object)
        for leaf in self.leaves:
            self.origin_split[leaf.indices] = leaf.split
        self.merge_small()


def groups_table(grouper: Grouper, window_ids: np.ndarray) -> pa.Table:
    """Canonical group ids: groups sorted by their smallest window_id."""
    leaves = grouper.leaves
    keys = [min(window_ids[leaf.indices].tolist()) for leaf in leaves]
    order = sorted(range(len(leaves)), key=keys.__getitem__)
    count = len(window_ids)
    group_id = np.full(count, -1, dtype=np.int32)
    path = np.empty(count, dtype=object)
    depth = np.zeros(count, dtype=np.int16)
    split = np.empty(count, dtype=object)
    size = np.zeros(count, dtype=np.int32)
    for canonical, position in enumerate(order):
        leaf = leaves[position]
        group_id[leaf.indices] = canonical
        path[leaf.indices] = leaf.path
        depth[leaf.indices] = leaf.depth
        split[leaf.indices] = leaf.split
        size[leaf.indices] = len(leaf.indices)
    if (group_id < 0).any():
        raise RuntimeError("some rows are not in any group")
    return pa.table(
        {
            "row_id": pa.array(window_ids.tolist(), pa.string()),
            "group_id": pa.array(group_id, pa.int32()),
            "group_path": pa.array(path.tolist(), pa.string()),
            "group_depth": pa.array(depth, pa.int16()),
            "group_split": pa.array(split.tolist(), pa.string()),
            "group_size": pa.array(size, pa.int32()),
            "attached": pa.array(grouper.attached, pa.bool_()),
        }
    )


def metrics(grouper: Grouper, table: pa.Table, seconds: float, peak_mib: int) -> dict[str, Any]:
    sizes = np.asarray([len(leaf.indices) for leaf in grouper.leaves])
    depths = table["group_depth"].to_numpy()
    origin = grouper.origin_split
    qs = lambda values: {f"p{q}": float(np.percentile(values, q)) for q in (0, 5, 50, 95, 100)}  # noqa: E731
    return {
        "dbscan_row_frac": float(np.mean([value in ("root", "dbscan") for value in origin])),
        "final_dbscan_group_row_frac": float(
            np.mean([value.split("+")[0] in ("root", "dbscan") for value in table["group_split"].to_pylist()])
        ),
        "attached_frac": float(grouper.attached.mean()),
        "groups": int(len(sizes)),
        "group_size": qs(sizes),
        "group_depth": qs(depths),
        "dbscan_calls": grouper.stats.dbscan_calls,
        "dbscan_cache_hits": grouper.stats.dbscan_cache_hits,
        "dbscan_seconds": grouper.stats.dbscan_seconds,
        "attach_seconds": grouper.stats.attach_seconds,
        "kmeans_fallback_nodes": grouper.stats.kmeans_fallback_nodes,
        "pruned_levels": grouper.stats.pruned_levels,
        "prune_seconds": grouper.stats.prune_seconds,
        "wall_seconds": seconds,
        "peak_gpu_mib": peak_mib,
        "merged_group_frac": float(np.mean(["+merged" in value for value in table["group_split"].to_pylist()])),
        "dbscan_call_log": grouper.stats.calls,
    }


def check(table: pa.Table, params: Params, expected_rows: int) -> None:
    group_id = table["group_id"].to_numpy()
    sizes = np.bincount(group_id)
    if int(sizes.sum()) != expected_rows or table.num_rows != expected_rows:
        raise AssertionError("group sizes do not sum to the population")
    if len(set(table["row_id"].to_pylist())) != expected_rows:
        raise AssertionError("row ids are not unique")
    if not (sizes.min() >= params.n_min and sizes.max() <= params.n_max):
        raise AssertionError(f"group size out of [{params.n_min}, {params.n_max}]: {sizes.min()}..{sizes.max()}")
    if not np.array_equal(sizes[group_id], table["group_size"].to_numpy()):
        raise AssertionError("group_size column disagrees with membership")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True, help="source_meta.parquet (window_id in embedding order)")
    parser.add_argument("--params", type=str, required=True, help="JSON Params")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--root-reach", type=Path, default=None, help="precomputed full-population reach_distance_m*.npy")
    args = parser.parse_args()
    params = Params(**json.loads(args.params))
    # Canonical partitions must be bit-reproducible: CUDA index_add_ (k-means centroid sums) is
    # otherwise atomic and nondeterministic, which flips near-tie assignments between reruns.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    log_handle = (args.output / "grouping.log").open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    window_ids = np.asarray(pq.read_table(args.meta, columns=["window_id"])["window_id"].to_pylist(), dtype=object)
    unit = torch.as_tensor(np.load(args.embeddings), device="cuda")
    start = time.time()
    with GpuMemoryPeak() as peak:
        root_reach = None if args.root_reach is None else np.load(args.root_reach)
        grouper = Grouper(unit, params, args.cache_dir, log, root_reach)
        grouper.run()
        table = groups_table(grouper, window_ids)
    seconds = time.time() - start
    check(table, params, len(window_ids))
    pq.write_table(table, args.output / "groups.parquet", compression="zstd")
    result = {"params": asdict(params), **metrics(grouper, table, seconds, peak.peak_mib)}
    (args.output / "grouping_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(json.dumps({k: v for k, v in result.items() if k != "dbscan_call_log"}))


if __name__ == "__main__":
    main()
