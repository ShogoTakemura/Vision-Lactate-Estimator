"""squat_core/joint_torque.py のサニティチェック。

下肢関節トルク推定はまだ実データでの数値検証を行っていないため、
ここでは「手計算で正しさを保証できる」単純なケースのみを検証する:

  1. 全セグメントが鉛直に一直線(水平方向の力・てこの長さが0)で静止して
     いる場合、関節モーメントは全て0になるはず(モーメントを発生させる
     水平方向の偶力が存在しないため)。
  2. 足首モーメント単体は、COPが足首から水平方向にずれているときの
     「力 x 距離」の手計算と一致するはず。
"""

from __future__ import annotations

import math

import pytest
from squat_core import joint_torque as jt


def test_segment_sagittal_angle_vertical_segment_is_zero() -> None:
    # 近位(腰)が遠位(膝)の真上にある(同じZ) -> 垂直 -> theta=0
    assert jt.segment_sagittal_angle((0.0, 0.5), (1.0, 0.5)) == pytest.approx(0.0)


def test_segment_sagittal_angle_forward_lean_is_positive() -> None:
    # 遠位端がZ+(前方)にずれている -> theta > 0
    theta = jt.segment_sagittal_angle((0.0, 0.0), (1.0, 0.3))
    assert theta > 0.0


def test_unwrap_angles_removes_branch_cut_jump() -> None:
    # atan2の分枝切断: +pi付近から-pi付近へ「ジャンプ」したように見える系列
    near_pi = math.pi - 0.05
    near_neg_pi = -math.pi + 0.05
    angles = [near_pi, near_neg_pi, near_neg_pi - 0.05]
    unwrapped = jt.unwrap_angles(angles)
    # 補正後は連続的(隣接差がpiを超えない)はず
    assert unwrapped[0] == pytest.approx(near_pi)
    for a, b in zip(unwrapped, unwrapped[1:], strict=False):
        assert abs(b - a) < math.pi


def test_unwrap_angles_empty_and_single() -> None:
    assert jt.unwrap_angles([]) == []
    assert jt.unwrap_angles([1.0]) == [1.0]


def test_cross2_basic() -> None:
    assert jt.cross2(1.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)
    assert jt.cross2(0.0, 1.0, 0.0, 1.0) == pytest.approx(0.0)


def test_ankle_moment_matches_hand_calculation() -> None:
    # 足首の真上(同じY,Z)にCOPがある場合はモーメント0
    assert jt.ankle_moment_from_grf(
        ankle_yz=(0.0, 0.0), cop_yz=(0.0, 0.0), grf_yz=(700.0, 0.0)
    ) == pytest.approx(0.0)

    # COPが足首より0.1m前方(Z+)、垂直反力700Nのみ -> |M| = 700 * 0.1 = 70 N*m
    moment = jt.ankle_moment_from_grf(
        ankle_yz=(0.0, 0.0), cop_yz=(0.0, 0.1), grf_yz=(700.0, 0.0)
    )
    assert moment == pytest.approx(70.0)


def test_static_vertical_stack_produces_zero_moments() -> None:
    """全関節が一直線上(Z=0)・無加速度・水平反力0のとき、
    膝・腰モーメントは0になる(回転を生む水平方向のてこが存在しないため)。
    """
    grf = (700.0, 0.0)
    ankle_yz = (0.0, 0.0)
    knee_yz = (-0.4, 0.0)
    hip_yz = (-0.8, 0.0)
    crus_com_yz = (-0.2, 0.0)
    thigh_com_yz = (-0.6, 0.0)

    ankle_m = jt.ankle_moment_from_grf(ankle_yz, ankle_yz, grf)
    assert ankle_m == pytest.approx(0.0)

    knee_m, knee_reaction = jt.knee_moment(
        ankle_moment=ankle_m,
        ankle_yz=ankle_yz,
        knee_yz=knee_yz,
        crus_com_yz=crus_com_yz,
        crus_com_accel_yz=(0.0, 0.0),
        crus_mass=3.0,
        crus_inertia=0.05,
        crus_angular_accel=0.0,
        grf_yz=grf,
    )
    assert knee_m == pytest.approx(0.0)

    hip_m = jt.hip_moment(
        knee_moment_value=knee_m,
        knee_yz=knee_yz,
        hip_yz=hip_yz,
        thigh_com_yz=thigh_com_yz,
        thigh_com_accel_yz=(0.0, 0.0),
        thigh_mass=8.0,
        thigh_inertia=0.1,
        thigh_angular_accel=0.0,
        knee_reaction_force_yz=knee_reaction,
    )
    assert hip_m == pytest.approx(0.0)


def test_segment_moment_of_inertia_male_thigh_known_value() -> None:
    # parameter.py の回帰式: (a0 + a1*m + a2*length_cm) / 10000
    # m=80kg, thigh length=400mm(=40cm) のときの手計算値と一致することを確認
    mass = 80.0
    length_mm = 400.0
    expected = (-2043.38 + 5547.75 * mass + 10.6498 * 40.0) / 10000.0
    actual = jt.segment_moment_of_inertia('thigh', 'M', mass, length_mm)
    assert actual == pytest.approx(expected)


def test_segment_proximal_force_static_equilibrium() -> None:
    # 静止(加速度0)・遠位力0のとき、近位力は -m*g のみ(重力を支える)
    proximal = jt.segment_proximal_force(
        seg_mass=5.0, accel_yz=(0.0, 0.0), distal_force_yz=(0.0, 0.0), gravity=9.8
    )
    assert proximal == pytest.approx((-49.0, 0.0))
