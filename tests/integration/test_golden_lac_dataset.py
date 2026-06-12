"""Golden test: auto_pipeline Step 6 reproduces lac_dataset_full.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "golden"
PIPELINE_DIR = GOLDEN_DIR / "pipeline"
REPS_DIR = GOLDEN_DIR / "processed_reps"


@pytest.fixture
def golden_lac_dataset() -> pd.DataFrame:
    path = PIPELINE_DIR / "lac_dataset_full.csv"
    if not path.is_file():
        pytest.skip(f"Golden fixture not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def test_golden_manifest_exists() -> None:
    manifest = GOLDEN_DIR / "manifest.json"
    assert manifest.is_file(), "Run scripts/capture_golden_baseline.py first"


@pytest.mark.integration
def test_step6_matches_golden(tmp_path: Path, golden_lac_dataset: pd.DataFrame) -> None:
    """Re-run Step 6 build logic against fixture inputs; output must match golden CSV."""
    from poseestimate_mediapipe.process import auto_pipeline as ap

    required = [
        PIPELINE_DIR / "input_database_dataset.csv",
        PIPELINE_DIR / "REP_DATABASE.csv",
        PIPELINE_DIR / "REP_POSTURE_DATABASE.csv",
    ]
    if not all(p.is_file() for p in required) or not REPS_DIR.is_dir():
        pytest.skip("Golden pipeline inputs incomplete; run capture_golden_baseline.py")

    out_path = tmp_path / "lac_dataset_full.csv"

    ap.BASE_DATASET_PATH = str(PIPELINE_DIR / "input_database_dataset.csv")
    ap.REP_DATABASE_PATH = str(PIPELINE_DIR / "REP_DATABASE.csv")
    ap.REP_POSTURE_PATH = str(PIPELINE_DIR / "REP_POSTURE_DATABASE.csv")
    ap.PROCESSED_REP_DIR = str(REPS_DIR)
    ap.OUTPUT_DATASET_PATH = str(out_path)

    ap.step6_build_dataset()

    assert out_path.is_file(), "step6_build_dataset did not produce output"
    actual = pd.read_csv(out_path, encoding="utf-8-sig")

    pd.testing.assert_frame_equal(
        actual,
        golden_lac_dataset,
        check_dtype=False,
        rtol=1e-9,
        atol=1e-9,
    )
