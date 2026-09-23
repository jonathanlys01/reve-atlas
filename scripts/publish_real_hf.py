#!/usr/bin/env python
"""Publish the validated real-data atlas tables to a Hugging Face dataset."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="jonathan-lys/reve-atlas")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--card", type=Path, default=Path("hf/README.real.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = {
        "display.parquet": ["data/display.parquet"],
        "atlas.parquet": ["data/atlas.parquet"],
        "selections.parquet": ["data/selections.parquet"],
        # The frontend fetches this table itself (not via DuckDB's direct URL read) to work
        # around an HF LFS/Xet redirect Content-Length bug, and only knows the "_web" name.
        "selection_runs.parquet": ["data/selection_runs.parquet", "data/selection_runs_web.parquet"],
        "curve_summary.parquet": ["data/curve_summary.parquet"],
        "manifest.json": ["data/manifest.json"],
    }
    for filename in files:
        path = args.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Validated real-data artifact not found: {path}")
    if not args.card.is_file():
        raise FileNotFoundError(f"Dataset card not found: {args.card}")

    api = HfApi()
    try:
        api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    except HfHubHTTPError as error:
        if error.response.status_code == HTTPStatus.FORBIDDEN:
            raise PermissionError("The configured Hugging Face token cannot write this dataset.") from error
        raise

    for filename, remote_paths in files.items():
        for remote_path in remote_paths:
            api.upload_file(
                path_or_fileobj=args.data_dir / filename,
                path_in_repo=remote_path,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Add real REVE Atlas {remote_path}",
            )
    api.upload_file(
        path_or_fileobj=args.card,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Document real REVE Atlas data",
    )
    print(f"Published https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
