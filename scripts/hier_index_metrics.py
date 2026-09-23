#!/usr/bin/env python
"""Gate 4: index-level metrics on the union over groups for every (method, direction, kernel, w, eta).

Vendi of the union via the 512x512 dual, mean pairwise cosine from the sum vector, coverage@tau
on a fixed display sample, density flattening against the largest decile of groups, recon-loss
quantiles, dataset shares, recordings represented, Jaccard against the published per-recording
DPP, and rank_limited counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

TAUS = (0.8, 0.9)
UNION_KEYS = ["method", "direction", "kernel_method", "w_interaction", "selection_eta"]


def union_memberships(directory: Path) -> tuple[list[dict], dict[tuple, np.ndarray]]:
    runs = pq.read_table(
        directory / "selection_runs.parquet",
        columns=["run_id", *UNION_KEYS, *(c for c in ("rank_limited",) if c in pq.read_schema(directory / "selection_runs.parquet").names)],
    ).to_pylist()
    key_of_run = {r["run_id"]: tuple(r[k] for k in UNION_KEYS) for r in runs}
    selections = pq.read_table(directory / "selections.parquet", columns=["run_id", "row_id"])
    run_ids = selections["run_id"].to_numpy(zero_copy_only=False)
    row_ids = selections["row_id"].to_numpy(zero_copy_only=False)
    order = np.argsort(run_ids, kind="stable")
    run_sorted = run_ids[order]
    starts = np.flatnonzero(np.r_[True, run_sorted[1:] != run_sorted[:-1]])
    ends = np.r_[starts[1:], len(run_sorted)]
    unions: dict[tuple, list[np.ndarray]] = {}
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        unions.setdefault(key_of_run[run_sorted[start]], []).append(row_ids[order[start:end]])
    return runs, {key: np.concatenate(parts) for key, parts in unions.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--set-name", required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True, help="source_meta.parquet (embedding order)")
    parser.add_argument("--display", type=Path, required=True)
    parser.add_argument("--groups", type=Path, default=None, help="groups.parquet; default: recordings")
    parser.add_argument("--published", type=Path, default=None, help="published data dir for Jaccard")
    parser.add_argument("--coverage-sample", type=Path, required=True, help=".npy of embedding-order rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")

    meta = pq.read_table(args.meta, columns=["window_id", "big_recording_index", "recon_loss"])
    window_ids = np.asarray(meta["window_id"].to_pylist(), dtype=object)
    position = {w: i for i, w in enumerate(window_ids.tolist())}
    recording = meta["big_recording_index"].to_numpy()
    losses = meta["recon_loss"].to_numpy().astype(np.float64)
    display = pq.read_table(args.display, columns=["window_id", "dataset"])
    dataset_by_window = dict(zip(display["window_id"].to_pylist(), display["dataset"].to_pylist(), strict=True))
    dataset = np.asarray([dataset_by_window[w] for w in window_ids.tolist()], dtype=object)
    if args.groups is not None:
        groups = pq.read_table(args.groups, columns=["row_id", "group_id"])
        gid = dict(zip(groups["row_id"].to_pylist(), groups["group_id"].to_pylist(), strict=True))
        group = np.asarray([gid[w] for w in window_ids.tolist()], dtype=np.int64)
    else:
        group = recording.astype(np.int64)
    group_sizes = np.bincount(group - group.min())
    group_index = group - group.min()
    decile_cut = np.quantile(group_sizes[group_sizes > 0], 0.9)
    in_top_decile = (group_sizes >= decile_cut)[group_index]
    population_top_decile_share = float(in_top_decile.mean())

    unit = torch.as_tensor(np.load(args.embeddings), device=device)
    sample = torch.as_tensor(np.load(args.coverage_sample), device=device)
    sample_unit = unit[sample]

    runs, unions = union_memberships(args.set_dir)
    rank_limited: dict[tuple, int] = {}
    for run in runs:
        key = tuple(run[k] for k in UNION_KEYS)
        rank_limited[key] = rank_limited.get(key, 0) + int(bool(run.get("rank_limited")))
    published_unions: dict[tuple, set] = {}
    if args.published is not None:
        _, published = union_memberships(args.published)
        published_unions = {key: set(rows.tolist()) for key, rows in published.items() if key[0] == "dpp"}

    rows = []
    for key in sorted(unions, key=lambda k: tuple("" if v is None else v for v in k)):
        members = unions[key]
        indices = np.asarray([position[w] for w in members.tolist()], dtype=np.int64)
        if len(np.unique(indices)) != len(indices):
            raise RuntimeError(f"duplicate rows in union {key}")
        k = len(indices)
        selected = unit[torch.as_tensor(indices, device=device)].double()
        eigenvalues = torch.linalg.eigvalsh(selected.T @ selected / k).clamp(min=0)
        probabilities = eigenvalues / eigenvalues.sum()
        probabilities = probabilities[probabilities > 0]
        vendi = float(torch.exp(-(probabilities * probabilities.log()).sum()))
        total = selected.sum(0)
        mean_cosine = float((total @ total - k) / (k * (k - 1))) if k > 1 else 1.0
        best = torch.full((len(sample),), -torch.inf, device=device)
        selected32 = selected.float()
        for start in range(0, k, 65536):
            block = selected32[start : start + 65536]
            for q0 in range(0, len(sample), 16384):
                sims = sample_unit[q0 : q0 + 16384] @ block.T
                best[q0 : q0 + 16384] = torch.maximum(best[q0 : q0 + 16384], sims.max(dim=1).values)
        coverage = {f"coverage_{tau}": float((best >= tau).float().mean()) for tau in TAUS}
        selected_losses = losses[indices]
        datasets, counts = np.unique(dataset[indices], return_counts=True)
        top = np.argsort(-counts, kind="stable")[:5]
        row = {
            "set": args.set_name,
            **dict(zip(UNION_KEYS, key, strict=True)),
            "k": k,
            "vendi_union": vendi,
            "vendi_union_normalized": vendi / k,
            "mean_pairwise_cosine_union": mean_cosine,
            **coverage,
            "top_decile_group_share": float(in_top_decile[indices].mean()),
            "population_top_decile_share": population_top_decile_share,
            "recon_loss_mean": float(selected_losses.mean()),
            **{f"recon_loss_q{q}": float(np.quantile(selected_losses, q / 100)) for q in (5, 25, 50, 75, 95)},
            "top5_datasets": json.dumps({str(datasets[i]): round(float(counts[i] / k), 4) for i in top}),
            "recordings_represented": int(len(np.unique(recording[indices]))),
            "groups_represented": int(len(np.unique(group[indices]))),
            "rank_limited_runs": rank_limited.get(key, 0),
            "jaccard_vs_published": None,
        }
        if key[0] == "dpp" and key in published_unions:
            mine = set(members.tolist())
            theirs = published_unions[key]
            row["jaccard_vs_published"] = len(mine & theirs) / len(mine | theirs)
        rows.append(row)
        print(key, k, f"vendi={vendi:.2f} cov0.9={coverage['coverage_0.9']:.3f}", flush=True)
    table = pa.Table.from_pylist(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd")
    print(f"wrote {len(rows)} rows to {args.output}; population top-decile share {population_top_decile_share:.4f}")


if __name__ == "__main__":
    main()
