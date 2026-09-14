#!/usr/bin/env python
"""Publish the synthetic atlas data and dataset card to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="jonathan-lys/reve-atlas")
    parser.add_argument("--parquet", type=Path, default=Path("data/demo.parquet"))
    parser.add_argument("--card", type=Path, default=Path("hf/README.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.is_file():
        raise FileNotFoundError(f"Generate the Parquet file first: {args.parquet}")
    if not args.card.is_file():
        raise FileNotFoundError(f"Dataset card not found: {args.card}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=args.parquet,
        path_in_repo="data/demo.parquet",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add deterministic REVE Atlas demo data",
    )
    api.upload_file(
        path_or_fileobj=args.card,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add REVE Atlas dataset card",
    )
    print(f"Published https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
