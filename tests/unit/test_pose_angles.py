"""Characterization tests for posture angle helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poseestimate_mediapipe.process.pose_analyze_process import (
    _inclination_angle,
    _three_point_angle,
)


def test_three_point_angle_right_angle() -> None:
    # A=(0,1), B=(0,0), C=(1,0) → 90° at B
    angle = _three_point_angle(
        pd.Series([0.0]),
        pd.Series([1.0]),
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([1.0]),
        pd.Series([0.0]),
    )
    assert angle[0] == pytest.approx(90.0)


def test_three_point_angle_collinear() -> None:
    angle = _three_point_angle(
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([0.5]),
        pd.Series([0.0]),
        pd.Series([1.0]),
    )
    assert angle[0] == pytest.approx(180.0, abs=0.01)


def test_three_point_angle_array_broadcast() -> None:
    n = 5
    y = pd.Series(np.linspace(0.0, 1.0, n))
    z = pd.Series(np.zeros(n))
    angles = _three_point_angle(y, z, y * 0, z, y, z + 1.0)
    assert len(angles) == n
    assert np.all(np.isfinite(angles))


def test_inclination_angle_vertical() -> None:
    incl = _inclination_angle(
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([1.0]),
        pd.Series([0.0]),
    )
    assert incl[0] == pytest.approx(0.0)


def test_inclination_angle_45_degrees() -> None:
    incl = _inclination_angle(
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([1.0]),
        pd.Series([1.0]),
    )
    assert incl[0] == pytest.approx(45.0)
