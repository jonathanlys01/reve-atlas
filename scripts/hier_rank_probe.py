#!/usr/bin/env python
"""Gate 1a: operational rank of the multiplicative (w=0) DPP kernel on cosine kNN balls.

For seeded rows, take the s cosine-nearest neighbours, run greedy MAP (multiplicative, w=0,
i.e. pivoted Cholesky of the cosine Gram + epsilon*I) to k=min(s, 600), and count the steps
whose pivot^2 exceeds the threshold. Also reports the participation ratio of X^T X.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

SIZES = (1024, 2048, 3072, 4096, 6144, 8192)


def pivot_sequence(unit: torch.Tensor, steps: int, epsilon: float) -> np.ndarray:
    """Pivot^2 sequence of greedy MAP on cosine_gram(unit) at multiplicative w=0 (fp64)."""
    gram = unit @ unit.T
    gram = (gram + gram.T) * 0.5
    gram.diagonal().add_(epsilon)
    count = gram.shape[0]
    residual = gram.diagonal().clone()
    factors = torch.zeros((steps, count), dtype=gram.dtype, device=gram.device)
    selected = torch.zeros(count, dtype=torch.bool, device=gram.device)
    pivots = []
    for step in range(steps):
        winner = int(torch.argmax(residual.masked_fill(selected, -torch.inf)).item())
        pivot_sq = float(residual[winner].clamp(min=epsilon).item())
        row = gram[winner].clone()
        if step:
            row -= factors[:step].T @ factors[:step, winner]
        row /= math.sqrt(pivot_sq)
        factors[step] = row
        residual -= row.square()
        selected[winner] = True
        pivots.append(pivot_sq)
    return np.asarray(pivots)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="L2-normalized float32 .npy (N x 512)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--threshold", type=float, default=1.0e-4)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    args = parser.parse_args()

    unit = np.load(args.embeddings, mmap_mode="r")
    seeds = np.sort(np.random.default_rng(args.seed).choice(unit.shape[0], size=args.seeds, replace=False))
    device = torch.device("cuda")
    all_unit = torch.as_tensor(np.ascontiguousarray(unit), device=device)
    records = []
    for seed_row in seeds.tolist():
        similarity = all_unit @ all_unit[seed_row]
        neighbours = torch.topk(similarity, max(SIZES)).indices  # sorted by descending cosine
        for size in SIZES:
            ball = all_unit[neighbours[:size]].double()
            steps = min(size, args.max_steps)
            pivots = pivot_sequence(ball, steps, args.epsilon)
            eigenvalues = torch.linalg.eigvalsh(ball.T @ ball).clamp(min=0)
            participation = float(eigenvalues.sum() ** 2 / eigenvalues.square().sum())
            records.append(
                {
                    "seed_row": seed_row,
                    "size": size,
                    "r_op": int(np.sum(pivots > args.threshold)),
                    "first_below": int(np.argmax(pivots <= args.threshold)) if np.any(pivots <= args.threshold) else steps,
                    "participation_ratio": participation,
                    "kth_cosine": float(similarity[neighbours[size - 1]].item()),
                }
            )
        print(seed_row, [(r["size"], r["r_op"]) for r in records[-len(SIZES):]], flush=True)
    summary = []
    for size in SIZES:
        rows = [r for r in records if r["size"] == size]
        r_op = np.asarray([r["r_op"] for r in rows])
        summary.append(
            {
                "size": size,
                "request_k_eta010": math.ceil(0.10 * size),
                "r_op_p5": float(np.percentile(r_op, 5)),
                "r_op_median": float(np.median(r_op)),
                "r_op_min": int(r_op.min()),
                "r_op_max": int(r_op.max()),
                "participation_median": float(np.median([r["participation_ratio"] for r in rows])),
                "participation_p5": float(np.percentile([r["participation_ratio"] for r in rows], 5)),
                "kth_cosine_median": float(np.median([r["kth_cosine"] for r in rows])),
                "passes_rule": math.ceil(0.10 * size) <= 0.9 * float(np.percentile(r_op, 5)),
                "steps_capped": min(size, args.max_steps),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "records": records, "args": vars(args) | {"embeddings": str(args.embeddings), "output": str(args.output)}}, indent=2))
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
