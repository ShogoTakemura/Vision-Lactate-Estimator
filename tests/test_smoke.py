"""Smoke tests to verify the test harness and package layout."""

from __future__ import annotations

from pathlib import Path


def test_repo_root_exists(repo_root: Path) -> None:
    assert repo_root.is_dir()
    assert (repo_root / "poseestimate_mediapipe").is_dir()


def test_fixtures_dir_ready(fixtures_dir: Path) -> None:
    assert fixtures_dir.is_dir()


def test_poseestimate_mediapipe_importable() -> None:
    import poseestimate_mediapipe  # noqa: F401

    assert poseestimate_mediapipe.__name__ == "poseestimate_mediapipe"
