"""
Capture auto_pipeline outputs as golden baseline fixtures.

Usage (from repo root):
    python scripts/capture_golden_baseline.py

Prerequisites:
    - Step 6 inputs exist under poseestimate_mediapipe/out/ and frame_viewer_tool/
    - Run auto_pipeline first if outputs are stale:
        PYTHONIOENCODING=utf-8 python -m poseestimate_mediapipe.process.auto_pipeline --skip-until 4
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden"
PIPELINE_DIR = GOLDEN_DIR / "pipeline"
REPS_DIR = GOLDEN_DIR / "processed_reps"

PIPELINE_SOURCES = {
    "lac_dataset_full.csv": REPO_ROOT / "poseestimate_mediapipe/out/work_calculated/lac_dataset_full.csv",
    "input_database_dataset.csv": REPO_ROOT
    / "poseestimate_mediapipe/out/work_calculated/input_database_dataset.csv",
    "REP_DATABASE.csv": REPO_ROOT / "poseestimate_mediapipe/out/work_calculated/REP_DATABASE.csv",
    "TOTAL_WORK_DATABASE.csv": REPO_ROOT
    / "poseestimate_mediapipe/out/work_calculated/TOTAL_WORK_DATABASE.csv",
    "REP_POSTURE_DATABASE.csv": REPO_ROOT
    / "poseestimate_mediapipe/out/rep_posture/REP_POSTURE_DATABASE.csv",
}

REP_SOURCE = REPO_ROOT / "frame_viewer_tool/reps/processed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _copy_pipeline_artifacts() -> dict[str, dict]:
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}

    for name, src in PIPELINE_SOURCES.items():
        if not src.is_file():
            raise FileNotFoundError(f"Missing pipeline artifact: {src}")
        dst = PIPELINE_DIR / name
        shutil.copy2(src, dst)
        df = pd.read_csv(dst, encoding="utf-8-sig")
        entries[name] = {
            "sha256": _sha256(dst),
            "rows": len(df),
            "columns": len(df.columns),
            "bytes": dst.stat().st_size,
        }
    return entries


def _copy_rep_csvs() -> dict[str, int]:
    if not REP_SOURCE.is_dir():
        raise FileNotFoundError(f"Missing rep CSV directory: {REP_SOURCE}")

    if REPS_DIR.exists():
        shutil.rmtree(REPS_DIR)
    REPS_DIR.mkdir(parents=True)

    count = 0
    total_bytes = 0
    for src in sorted(REP_SOURCE.glob("*_rep.csv")):
        dst = REPS_DIR / src.name
        shutil.copy2(src, dst)
        count += 1
        total_bytes += dst.stat().st_size
    return {"count": count, "bytes": total_bytes}


def main() -> None:
    pipeline = _copy_pipeline_artifacts()
    reps = _copy_rep_csvs()

    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "pipeline": "auto_pipeline step 6 (lac_dataset_full.csv)",
        "pipeline_command": "python -m poseestimate_mediapipe.process.auto_pipeline --skip-until 4",
        "artifacts": pipeline,
        "processed_reps": reps,
    }

    manifest_path = GOLDEN_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Golden baseline captured under {GOLDEN_DIR}")
    print(f"  lac_dataset_full.csv: {pipeline['lac_dataset_full.csv']['rows']} rows")
    print(f"  processed_reps: {reps['count']} files")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
