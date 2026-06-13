"""Characterization tests for segment COM calculations."""

from __future__ import annotations

import pytest
from poseestimate_mediapipe.module.com.segment import (
    BothEndSegment,
    FootSegment,
    HandSegment,
    HeadSegment,
)


def test_both_end_segment_com_linear_interpolation() -> None:
    seg = BothEndSegment(
        length=0.3,
        ratio=0.4,
        root=(0.0, 0.0, 0.0),
        end=(1.0, 2.0, 3.0),
    )
    assert seg.segment_com() == pytest.approx((0.6, 1.2, 1.8))


def test_hand_segment_com_along_unit_vector() -> None:
    seg = HandSegment(
        length=300.0,  # mm
        ratio=0.5,
        root=(0.0, 0.0, 0.0),
        end=(0.0, 1.0, 0.0),
    )
    assert seg.segment_com() == pytest.approx((0.0, 0.15, 0.0))


def test_head_segment_com_keeps_root_x_and_z() -> None:
    seg = HeadSegment(
        length=200.0,
        ratio=0.5,
        root=(1.0, 2.0, 3.0),
        end=(1.0, 5.0, 3.0),
    )
    com = seg.segment_com()
    assert com[0] == pytest.approx(1.0)
    assert com[2] == pytest.approx(3.0)
    assert com[1] == pytest.approx(1.1)


def test_foot_segment_com_on_xz_plane() -> None:
    seg = FootSegment(
        length=250.0,
        ratio=0.4,
        root=(0.0, 0.5, 0.0),
        end=(0.0, 0.0, 1.0),
    )
    com = seg.segment_com()
    assert com[1] == pytest.approx(0.0)
    assert com[2] == pytest.approx(0.9)
