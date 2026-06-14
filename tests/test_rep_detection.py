"""Unit tests for squat_core.rep_detection."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from squat_core.rep_detection import detect_reps_from_com, save_rep_csv

vec3d = tuple[float, float, float]


def _make_comlist(y_values: list[float]) -> list[vec3d]:
    """y軸のみ変化する最小 comlist を生成する。"""
    return [(0.0, y, 0.0) for y in y_values]


def _synthetic_squat(
    fps: float = 30.0,
    n_reps: int = 3,
    rep_frames: int = 60,  # 1 rep = 2 s at 30 Hz
    y_stand: float = 0.5,
    y_bottom: float = 1.2,
) -> list[vec3d]:
    """立位→底部→立位 を n_reps 繰り返す合成軌跡を生成する。"""
    y = []
    for _ in range(n_reps):
        # 降下: y_stand → y_bottom
        half = rep_frames // 2
        y.extend(np.linspace(y_stand, y_bottom, half).tolist())
        # 上昇: y_bottom → y_stand
        y.extend(np.linspace(y_bottom, y_stand, half).tolist())
    return _make_comlist(y)


# ---------------------------------------------------------------------------
# 基本的なレップ検出
# ---------------------------------------------------------------------------


def test_detect_correct_rep_count() -> None:
    comlist = _synthetic_squat(n_reps=3)
    reps = detect_reps_from_com(comlist, fps=30.0)
    assert len(reps) == 3


def test_rep_numbering_starts_at_1() -> None:
    comlist = _synthetic_squat(n_reps=2)
    reps = detect_reps_from_com(comlist, fps=30.0)
    assert reps[0][0] == 1
    assert reps[1][0] == 2


def test_bottom_frame_has_highest_y() -> None:
    """bottom_frame が各レップ内で y が最大のフレームであること。"""
    comlist = _synthetic_squat(n_reps=2, fps=30.0)
    y = [c[1] for c in comlist]
    reps = detect_reps_from_com(comlist, fps=30.0)
    for rep, start, bottom, end in reps:
        seg_y = y[start : end + 1]
        local_max_idx = start + int(np.argmax(seg_y))
        assert abs(local_max_idx - bottom) <= 1, (
            f"rep {rep}: bottom={bottom} but argmax={local_max_idx}"
        )


def test_start_le_bottom_le_end() -> None:
    comlist = _synthetic_squat(n_reps=4)
    reps = detect_reps_from_com(comlist, fps=30.0)
    for rep, start, bottom, end in reps:
        assert start <= bottom, f"rep {rep}: start {start} > bottom {bottom}"
        assert bottom <= end, f"rep {rep}: bottom {bottom} > end {end}"


def test_empty_for_flat_trajectory() -> None:
    """y が一定なら検出 0 件。"""
    comlist = _make_comlist([0.5] * 90)
    reps = detect_reps_from_com(comlist, fps=30.0)
    assert reps == []


def test_single_rep() -> None:
    comlist = _synthetic_squat(n_reps=1)
    reps = detect_reps_from_com(comlist, fps=30.0)
    assert len(reps) == 1


# ---------------------------------------------------------------------------
# save_rep_csv
# ---------------------------------------------------------------------------


def test_save_rep_csv(tmp_path: Path) -> None:
    comlist = _synthetic_squat(n_reps=2)
    reps = detect_reps_from_com(comlist, fps=30.0)
    out = tmp_path / "test_rep.csv"
    save_rep_csv(reps, out)

    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["rep", "start_frame", "bottom_frame", "end_frame"]
    assert len(rows) == 1 + len(reps)


def test_save_rep_csv_creates_parent(tmp_path: Path) -> None:
    comlist = _synthetic_squat(n_reps=1)
    reps = detect_reps_from_com(comlist, fps=30.0)
    out = tmp_path / "nested" / "dir" / "test_rep.csv"
    save_rep_csv(reps, out)
    assert out.exists()
