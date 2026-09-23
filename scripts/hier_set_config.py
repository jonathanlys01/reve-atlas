#!/usr/bin/env python
"""Write the Gate 3 sweep config for one {grouping} x {allocation} artifact set."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ETAS = (0.01, 0.03, 0.10)
KERNELS = {
    "multiplicative": {"w_interaction": [0, 0.125, 0.25, 0.5, 1, 2, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64]},
    "additive": {"w_interaction": [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30, 64]},
}


def eta_tag(eta: float) -> str:
    return f"{round(eta * 100):03d}"


def set_config(embeddings: str, groups: str, grouping: str, allocation: str, output: str, backend: str) -> dict:
    rankings = [
        {
            "id": f"{direction}_eta_{eta_tag(eta)}_per_group",
            "direction": direction,
            "scope": "per_group",
            "selection_eta": eta,
            "allocation": allocation,
        }
        for eta in ETAS
        for direction in ("top", "bottom")
    ]
    return {
        "input": {"embeddings": embeddings, "population": "display"},
        "output": {"directory": output, "atlas_artifact": "display.parquet"},
        "grouping": {"name": grouping, "source": groups, "column": "group_id"},
        "rankings": rankings,
        "dpp": [
            {
                "candidates": ranking["id"],
                "selection_eta": ranking["selection_eta"],
                "full_population": True,
                "allocation": allocation,
                "kernels": KERNELS,
            }
            for ranking in rankings
        ],
        "baselines": [
            {
                "id": f"random_stratified_eta_{eta_tag(eta)}_per_group",
                "scope": "per_group",
                "selection_eta": eta,
                "seed": 20260914,
                "allocation": allocation,
            }
            for eta in ETAS
        ],
        "numerics": {"epsilon": 1.0e-6, "max_workspace_gib": 8, "tie_tolerance": 1.0e-9, "dpp_backend": backend},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--grouping", required=True)
    parser.add_argument("--allocation", choices=("proportional", "flat"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", default="numpy", choices=("auto", "numpy", "torch"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = set_config(args.embeddings, args.groups, args.grouping, args.allocation, args.output_dir, args.backend)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(args.config)


if __name__ == "__main__":
    main()
