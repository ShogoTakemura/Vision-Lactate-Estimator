"""Shared pytest configuration for squat_analyze."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "fixtures"


def scipy_signal_available() -> bool:
    try:
        from scipy.signal import butter  # noqa: F401

        return True
    except ImportError:
        return False


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def hr_estimation():
    if not scipy_signal_available():
        pytest.skip("scipy.signal is unavailable (HR signal tests skipped)")
    return load_module_from_path(
        "hr_estimation",
        ROOT / "frame_viewer_tool" / "hr_estimation.py",
    )
