"""squat_core/rep_detection.py
-------------------------------
CoM y軌跡からスクワットのレップ区間を自動検出する。

MediaPipe の y 座標は下向き正なので、スクワット最下点 = y の局所最大値 (local maximum)。
各レップの start / bottom / end フレームを検出し、_rep.csv スキーマで書き出す。

Usage (as a library):
    from squat_core.rep_detection import detect_reps_from_com, save_rep_csv
    reps = detect_reps_from_com(comlist, fps=30.0)
    save_rep_csv(reps, "output/squat01_rep.csv")

Usage (CLI):
    python -m squat_core.rep_detection \\
        --com-pickle out/bodycompickle/squat01_modelbase.pickle \\
        --output poseestimate_mediapipe/frame_viewer_tool/reps/processed/squat01_rep.csv
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

vec3d = tuple[float, float, float]

# MediaPipe y convention: y increases downward
_Y = 1


def detect_reps_from_com(
    comlist: list[vec3d],
    fps: float = 30.0,
    *,
    min_rep_duration_s: float = 0.8,
    prominence_fraction: float = 0.04,
) -> list[tuple[int, int, int, int]]:
    """CoM y軌跡の局所最大点からスクワットレップ区間を検出する。

    Parameters
    ----------
    comlist              : CoM座標リスト [(x, y, z), ...]
    fps                  : フレームレート [Hz]
    min_rep_duration_s   : レップ最小間隔 [s]（ノイズ起因の短いピークを除外）
    prominence_fraction  : プロミネンス閾値 = y振幅 × この値

    Returns
    -------
    list of (rep_num, start_frame, bottom_frame, end_frame)  1-indexed
    """
    y = np.array([com[_Y] for com in comlist], dtype=float)
    y_range = float(y.max() - y.min())
    if y_range == 0.0:
        return []

    prominence = y_range * prominence_fraction
    distance = max(1, int(fps * min_rep_duration_s))

    bottoms, _ = find_peaks(y, distance=distance, prominence=prominence)
    if len(bottoms) == 0:
        return []

    def _valley(lo: int, hi: int) -> int:
        """[lo, hi] 区間内で y が最小になるフレームを返す（= 立位位置）。"""
        return int(np.argmin(y[lo : hi + 1])) + lo

    reps: list[tuple[int, int, int, int]] = []
    for i, bottom in enumerate(bottoms):
        prev = int(bottoms[i - 1]) if i > 0 else 0
        nxt = int(bottoms[i + 1]) if i < len(bottoms) - 1 else len(y) - 1

        start = _valley(prev, int(bottom))
        end = _valley(int(bottom), nxt)
        reps.append((i + 1, start, int(bottom), end))

    return reps


def save_rep_csv(
    reps: list[tuple[int, int, int, int]],
    output_path: str | Path,
) -> None:
    """レップ区間リストを _rep.csv スキーマで保存する。

    スキーマ: rep, start_frame, bottom_frame, end_frame
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rep", "start_frame", "bottom_frame", "end_frame"])
        writer.writerows(reps)


def _cli() -> None:
    from squat_core.paths import FRAME_VIEWER_ROOT

    parser = argparse.ArgumentParser(
        description="CoM pickle からレップ区間を自動検出して _rep.csv を生成する"
    )
    parser.add_argument(
        "--com-pickle",
        required=True,
        help="bodycompickle/xxx.pickle のパス",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="出力 _rep.csv のパス（省略時: poseestimate_mediapipe/frame_viewer_tool/reps/processed/<stem>_rep.csv）",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="フレームレート (default: 30)")
    parser.add_argument(
        "--min-rep-duration",
        type=float,
        default=0.8,
        help="レップ最小間隔 [s] (default: 0.8)",
    )
    parser.add_argument(
        "--prominence-fraction",
        type=float,
        default=0.04,
        help="プロミネンス閾値係数 (default: 0.04)",
    )
    args = parser.parse_args()

    com_path = Path(args.com_pickle)
    if not com_path.exists():
        parser.error(f"CoM pickle が見つかりません: {com_path}")

    with open(com_path, "rb") as f:
        comlist: list[vec3d] = pickle.load(f)

    reps = detect_reps_from_com(
        comlist,
        fps=args.fps,
        min_rep_duration_s=args.min_rep_duration,
        prominence_fraction=args.prominence_fraction,
    )

    if not reps:
        print(
            "レップが検出されませんでした。--prominence-fraction を小さくするか動画を確認してください。"
        )
        return

    if args.output:
        out_path = Path(args.output)
    else:
        stem = com_path.stem
        out_path = FRAME_VIEWER_ROOT / "reps" / "processed" / f"{stem}_rep.csv"

    save_rep_csv(reps, out_path)

    print(f"検出レップ数: {len(reps)}")
    print(f"  {'rep':>3}  {'start':>7}  {'bottom':>7}  {'end':>7}")
    for rep, start, bottom, end in reps:
        print(f"  {rep:>3}  {start:>7}  {bottom:>7}  {end:>7}")
    print(f"\n出力: {out_path}")


if __name__ == "__main__":
    _cli()
