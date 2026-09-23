#!/usr/bin/env python
"""Add a map-coloring ``group_colour`` column to a set's display.parquet.

The atlas colors points by a small categorical column (embedding-atlas stores the category as
UTINYINT and folds rare values into "(other)"), so ~1,000 group ids cannot each get a color.
Instead, groups that touch on the 2-D projection get different colors: adjacency comes from
k-nearest neighbours of the display points in projection space, colored greedily (DSatur,
largest-degree ties by group id). Leftover same-color contacts are reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from scripts.real_atlas import atomic_write_table


def dsatur(adjacency: list[set[int]], weights: dict[tuple[int, int], int], palette: int) -> np.ndarray:
    """Greedy DSatur; when all palette colors are taken, reuse the least-conflicting one."""
    count = len(adjacency)
    colors = np.full(count, -1, dtype=np.int64)
    saturation = [set() for _ in range(count)]
    degree = np.asarray([len(n) for n in adjacency])
    for _ in range(count):
        uncolored = np.flatnonzero(colors < 0)
        node = int(max(uncolored.tolist(), key=lambda i: (len(saturation[i]), degree[i], -i)))
        free = [c for c in range(palette) if c not in saturation[node]]
        if free:
            color = free[0]
        else:
            cost = np.zeros(palette)
            for other in adjacency[node]:
                if colors[other] >= 0:
                    cost[colors[other]] += weights[(min(node, other), max(node, other))]
            color = int(np.argmin(cost))
        colors[node] = color
        for other in adjacency[node]:
            saturation[other].add(color)
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--palette", type=int, default=12)
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--min-contacts", type=int, default=20, help="neighbour pairs needed to call two groups adjacent")
    args = parser.parse_args()
    path = args.set_dir / "display.parquet"
    table = pq.read_table(path)
    group = table["group_id"].to_numpy().astype(np.int64)
    xy = np.column_stack([table["projection_x"].to_numpy(), table["projection_y"].to_numpy()]).astype(np.float64)
    _, neighbours = cKDTree(xy).query(xy, k=args.neighbours + 1, workers=16)
    left = np.repeat(group, args.neighbours)
    right = group[neighbours[:, 1:].reshape(-1)]
    cross = left != right
    pairs = np.column_stack([np.minimum(left[cross], right[cross]), np.maximum(left[cross], right[cross])])
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    groups = int(group.max()) + 1
    adjacency: list[set[int]] = [set() for _ in range(groups)]
    weights: dict[tuple[int, int], int] = {}
    for (a, b), c in zip(unique.tolist(), counts.tolist(), strict=True):
        if c >= args.min_contacts:
            adjacency[a].add(b)
            adjacency[b].add(a)
            weights[(a, b)] = c
    colors = dsatur(adjacency, weights, args.palette)
    same = colors[pairs[:, 0]] == colors[pairs[:, 1]]
    stats = {
        "groups": groups,
        "palette": args.palette,
        "adjacent_group_pairs": len(weights),
        "max_degree": max(len(n) for n in adjacency),
        "same_colour_adjacent_pairs": int(sum(1 for (a, b) in weights if colors[a] == colors[b])),
        "cross_group_neighbour_contacts": int(len(pairs)),
        "same_colour_contact_frac": float(same.mean()) if len(pairs) else 0.0,
    }
    labels = np.asarray([f"colour {c + 1:02d}" for c in range(args.palette)], dtype=object)[colors[group]]
    if "group_colour" in table.column_names:
        table = table.drop(["group_colour"])
    table = table.append_column("group_colour", pa.array(labels.tolist(), pa.string()))
    atomic_write_table(table, path)
    (args.set_dir / "group_colour_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(args.set_dir.name, stats)


if __name__ == "__main__":
    main()
