#!/usr/bin/env python
"""Resumable three-stage launcher for the real REVE atlas build."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(path: Path, **updates: Any) -> None:
    status = {}
    if path.exists():
        status = json.loads(path.read_text(encoding="utf-8"))
    status.update(updates)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(json.dumps(__import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))))
    output = Path(str(config.get("output", {}).get("directory", "data/real"))).expanduser().resolve()
    status_path = (args.status or output / "status.json").resolve()
    status_path.parent.mkdir(parents=True, exist_ok=True)
    _write_status(
        status_path,
        state="running",
        current_run="real-atlas/projection-selection-validation",
        completed_runs=[],
        last_progress_at=_now(),
        last_result_at=None,
        message="launcher started",
        config=str(args.config.resolve()),
        output_root=str(output),
        pid=os.getpid(),
    )
    stages = [
        ("projection", "scripts.build_real_atlas"),
        ("selection", "scripts.build_selections"),
        ("validation", "scripts.validate_real_artifacts"),
    ]
    try:
        for stage, module in stages:
            _write_status(status_path, current_run=f"real-atlas/{stage}", last_progress_at=_now(), message=f"starting {stage}")
            result = subprocess.run([sys.executable, "-m", module, "--config", str(args.config.resolve())], check=False)
            if result.returncode:
                raise RuntimeError(f"{stage} failed with exit code {result.returncode}")
            completed = json.loads(status_path.read_text(encoding="utf-8")).get("completed_runs", [])
            completed.append(stage)
            _write_status(
                status_path,
                completed_runs=completed,
                last_progress_at=_now(),
                last_result_at=_now(),
                message=f"completed {stage}",
            )
        _write_status(status_path, state="completed", current_run=None, last_progress_at=_now(), message="all stages completed")
    except Exception as error:
        _write_status(status_path, state="failed", last_progress_at=_now(), message=str(error))
        raise


if __name__ == "__main__":
    main()
