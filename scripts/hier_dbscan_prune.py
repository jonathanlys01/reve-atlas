#!/usr/bin/env python
"""Exact DBSCAN clustered-fraction curve at the root, for pruning the Gate 1b eps ladder.

With cuML/sklearn semantics (min_samples counts the point itself), row i is core at eps iff
cd_m(i) <= eps, where cd_m is the distance to its m-th nearest row counting itself; it is in
some cluster (core or border) iff b_m(i) = min(cd_m(i), min_{j != i} max(cd_m(j), d(i, j))) <= eps.
So the fraction of rows DBSCAN can place in any cluster at eps is mean(b_m <= eps), exactly, for
every eps at once. A node split needs viable rows >= viable_frac * n, and viable rows are a subset
of clustered rows, so any eps with mean(b_m <= eps) < viable_frac is rejected without a call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

M_VALUES = (5, 10, 25)
QUERY_CHUNK = 4096
TARGET_CHUNK = 262144


def distances(unit: torch.Tensor, queries: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(2.0 - 2.0 * (unit[queries] @ unit[targets].T), min=0.0))


def reach_distances(unit: torch.Tensor, indices: np.ndarray, m: int) -> np.ndarray:
    """b_m for the rows ``indices`` computed within that subset only (same semantics as main)."""
    rows = torch.as_tensor(indices, device=unit.device)
    count = len(indices)
    local = torch.arange(count, device=unit.device)
    core = torch.empty(count, device=unit.device)
    for q0 in range(0, count, QUERY_CHUNK):
        queries = rows[q0 : q0 + QUERY_CHUNK]
        best = torch.full((len(queries), m), torch.inf, device=unit.device)
        for t0 in range(0, count, TARGET_CHUNK):
            block = distances(unit, queries, rows[t0 : t0 + TARGET_CHUNK])
            best = torch.topk(torch.cat([best, block], dim=1), m, dim=1, largest=False).values
        core[q0 : q0 + QUERY_CHUNK] = best[:, m - 1]
    reach = core.clone()
    for q0 in range(0, count, QUERY_CHUNK):
        queries = rows[q0 : q0 + QUERY_CHUNK]
        for t0 in range(0, count, TARGET_CHUNK):
            block = distances(unit, queries, rows[t0 : t0 + TARGET_CHUNK])
            block = block.masked_fill(local[q0 : q0 + QUERY_CHUNK, None] == local[None, t0 : t0 + TARGET_CHUNK], torch.inf)
            value = torch.maximum(block, core[t0 : t0 + TARGET_CHUNK][None, :]).min(dim=1).values
            reach[q0 : q0 + QUERY_CHUNK] = torch.minimum(reach[q0 : q0 + QUERY_CHUNK], value)
    return reach.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    unit = torch.as_tensor(np.load(args.embeddings), device="cuda")
    count = unit.shape[0]
    everything = torch.arange(count, device="cuda")
    kmax = max(M_VALUES)

    # Pass 1: m-th nearest distance counting self (column m-1 of the ascending list incl. self).
    nearest = torch.empty((count, kmax), device="cuda")
    for q0 in range(0, count, QUERY_CHUNK):
        queries = everything[q0 : q0 + QUERY_CHUNK]
        best = torch.full((len(queries), kmax), torch.inf, device="cuda")
        for t0 in range(0, count, TARGET_CHUNK):
            block = distances(unit, queries, everything[t0 : t0 + TARGET_CHUNK])
            best = torch.topk(torch.cat([best, block], dim=1), kmax, dim=1, largest=False).values
        nearest[q0 : q0 + QUERY_CHUNK] = best
        if q0 % (QUERY_CHUNK * 50) == 0:
            print(f"pass1 {q0}/{count}", flush=True)
    core = {m: nearest[:, m - 1].contiguous() for m in M_VALUES}

    # Pass 2: b_m(i) = min over j != i of max(cd_m(j), d(i, j)), then min with cd_m(i).
    reach = {m: core[m].clone() for m in M_VALUES}
    for q0 in range(0, count, QUERY_CHUNK):
        queries = everything[q0 : q0 + QUERY_CHUNK]
        for t0 in range(0, count, TARGET_CHUNK):
            targets = everything[t0 : t0 + TARGET_CHUNK]
            block = distances(unit, queries, targets)
            self_mask = queries[:, None] == targets[None, :]
            block = block.masked_fill(self_mask, torch.inf)
            for m in M_VALUES:
                value = torch.maximum(block, core[m][targets][None, :]).min(dim=1).values
                reach[m][q0 : q0 + QUERY_CHUNK] = torch.minimum(reach[m][q0 : q0 + QUERY_CHUNK], value)
        if q0 % (QUERY_CHUNK * 50) == 0:
            print(f"pass2 {q0}/{count}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    grid = np.geomspace(1e-3, 1.0, 400)
    summary = {}
    for m in M_VALUES:
        cd = core[m].cpu().numpy()
        b = reach[m].cpu().numpy()
        np.save(args.output / f"core_distance_m{m}.npy", cd)
        np.save(args.output / f"reach_distance_m{m}.npy", b)
        summary[str(m)] = {
            "eps_grid": grid.tolist(),
            "core_frac": [float(np.mean(cd <= e)) for e in grid],
            "clustered_frac": [float(np.mean(b <= e)) for e in grid],
            "eps_at_clustered_half": float(np.quantile(b, 0.5)),
        }
        print(m, "eps where clustered fraction = 0.5:", summary[str(m)]["eps_at_clustered_half"], flush=True)
    (args.output / "prune_summary.json").write_text(json.dumps(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
