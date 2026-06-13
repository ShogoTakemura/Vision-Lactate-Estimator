"""
squat_core/kinematics.py
------------------------
重心 (CoM) 運動学 + 床反力計算の正規実装。
calccomprocess / modelbasecorrect / functest の3重複を統合した単一実装。
"""

from __future__ import annotations

from poseestimate_mediapipe.config.constants import GRAVITY


def calc_velocity(
    comlist: list[float] | tuple[float, ...],
    index: int,
    timespan: float,
) -> float:
    """中心差分による速度 {f(x+h) - f(x-h)} / 2h。"""
    return (comlist[index + 1] - comlist[index - 1]) / (2.0 * timespan)


def calc_velocities(
    comlist: list[float] | tuple[float, ...],
    fps: float,
) -> tuple[float, ...]:
    """CoM 1D 時系列から速度時系列を算出する（端点は 0.0）。"""
    n = len(comlist)
    dt = 1.0 / fps
    return tuple(calc_velocity(comlist, i, dt) if i not in (0, n - 1) else 0.0 for i in range(n))


def calc_accel(left: float, center: float, right: float, timespan: float) -> float:
    """中心二階差分による加速度 {f(x+h) - 2f(x) + f(x-h)} / h^2。"""
    return (left - 2 * center + right) / (timespan * timespan)


def calc_accels(
    comlist: list[float] | tuple[float, ...],
    fps: float,
) -> tuple[float, ...]:
    """CoM 1D 時系列から加速度時系列を算出する（端点は 0.0）。"""
    n = len(comlist)
    dt = 1.0 / fps
    return tuple(
        calc_accel(comlist[i + 1], comlist[i], comlist[i - 1], dt) if i not in (0, n - 1) else 0.0
        for i in range(n)
    )


def _calc_floorforce(inertial: float, compositemass: float, gravity: float) -> float:
    """1 フレームの床反力 F = M*g - F_inertial。"""
    return compositemass * gravity - inertial


def calc_floorforces(
    inertialforce: list[float] | tuple[float, ...],
    compositemass: float,
    gravity: float = GRAVITY,
) -> tuple[float, ...]:
    """慣性力時系列から床反力時系列を算出する。"""
    return tuple(_calc_floorforce(f, compositemass, gravity) for f in inertialforce)


def calc_dist_force(
    l_foot_x: float,
    r_foot_x: float,
    com_x: float,
    force: float,
) -> tuple[float, float]:
    """左右足への床反力配分を計算する (r_force, l_force)。"""
    l_partial = abs(l_foot_x - com_x)
    r_partial = abs(r_foot_x - com_x)
    width = l_partial + r_partial
    r_force = force / width * r_partial
    l_force = force / width * l_partial
    return (r_force, l_force)


def distribute_floorforce(
    r_foot_com: tuple[float, ...],
    l_foot_com: tuple[float, ...],
    com: tuple[float, ...],
    floorforce: list[float] | tuple[float, ...],
) -> list[tuple[float, float]]:
    """フレームごとの床反力を左右足に配分する。"""
    return [
        calc_dist_force(lf, rf, c, f)
        for rf, lf, c, f in zip(r_foot_com, l_foot_com, com, floorforce, strict=False)
    ]
