"""
squat_core/joint_torque.py
----------------------------
矢状面 (Y-Z 平面, Y下向き正) における下肢関節モーメント推定。

足首 -> 膝 -> 腰(股関節) の順に Newton-Euler 逆動力学 (bottom-up
link-segment model, Winter "Biomechanics and Motor Control of Human
Movement" 参照) を適用し、各関節が支えなければならない net moment を
逐次計算する。

由来:
    squat_program/python/index_estimation/Joint_Torque_Estimation/joint_moment.py
    に同種の計算（地面反力 -> 足首モーメント -> 膝モーメント -> 腰モーメント）が
    存在するが、AlphaPose座標系・別の軸定義を前提にしており、かつ三角関数の
    abs(sin)/abs(cos) で力を分解する箇所など符号があいまいな実装になっている。

    本モジュールは同じ力学的な考え方を採用しつつ、行単位の移植ではなく
    F=ma / I*alpha=ΣM の運動方程式から本プロジェクトの座標系
    (MediaPipe, Y下向き正, squat_core.kinematics の既存の符号規約) に
    合わせて再導出したもの。

既知の単純化（要将来検証）:
    - 水平(Z)方向の地面反力は 0 と仮定し、鉛直反力のみで近似する
      (水平反力の実測・推定手段が無いため)。
    - 地面反力の作用点(COP)は、足首と足先(toe)関節の中点で近似する
      (実測の圧力分布ではない簡略化)。
    - 足部自身の質量・慣性は無視し、地面反力は足首関節を介して
      そのまま下腿に伝わるとみなす(質量の小さい遠位セグメントを無視する、
      多くの2D link-segmentモデルで採用される簡略化)。

    このモジュールはまだ実データでの数値検証(既知の静解析・先行研究との
    比較)を行っていない。研究目的で値を利用する前に、静止区間
    (加速度ゼロ)でのモーメントが手計算の静力学と一致するか確認すること。
"""

from __future__ import annotations

import math

from poseestimate_mediapipe.config.constants import GRAVITY
from poseestimate_mediapipe.module.com.aeparam import AEParam

vec2d = tuple[float, float]


def segment_sagittal_angle(proximal_yz: vec2d, distal_yz: vec2d) -> float:
    """近位端->遠位端ベクトルの、垂直線からの矢状面傾き角[rad]を返す。

    Y下向き正のため、近位->遠位ベクトルがそのまま「下向き」(dy>0)になるのが
    通常の立位姿勢(腰の真下に膝がある状態)。脚が垂直に伸びている時 theta=0,
    遠位端が前方(Z+方向)にずれるほど theta は正に増加する。
    """
    proximal_y, proximal_z = proximal_yz
    distal_y, distal_z = distal_yz
    dy = distal_y - proximal_y
    dz = distal_z - proximal_z
    return math.atan2(dz, dy)


def unwrap_angles(angles: list[float]) -> list[float]:
    """連続する角度系列の +-2pi ジャンプを補正する (atan2 の分枝切断対策)。

    squat_core.kinematics.calc_velocities/calc_accels の中心差分は値が
    連続的に変化することを前提とするため、segment_sagittal_angle() の
    出力をそのまま微分すると atan2 の +-pi 境界で実体のない巨大な
    角速度・角加速度が発生する。微分の前に必ずこの関数を通すこと。
    """
    if not angles:
        return []
    unwrapped = [angles[0]]
    offset = 0.0
    for i in range(1, len(angles)):
        diff = angles[i] - angles[i - 1]
        if diff > math.pi:
            offset -= 2.0 * math.pi
        elif diff < -math.pi:
            offset += 2.0 * math.pi
        unwrapped.append(angles[i] + offset)
    return unwrapped


def cross2(lever_y: float, lever_z: float, force_y: float, force_z: float) -> float:
    """YZ平面内の2次元クロス積 (X軸まわりのモーメントのスカラー成分)。"""
    return lever_y * force_z - lever_z * force_y


def segment_moment_of_inertia(
    segment: str,
    sex: str,
    composite_mass_kg: float,
    segment_length_mm: float,
) -> float:
    """阿江(1996)の回帰式によるセグメント慣性モーメント推定 [kg*m^2]。

    Args:
        segment: 'thigh'(大腿) または 'crus'(下腿)。
        sex: 'M' または 'F'。
        composite_mass_kg: 体重 + 負荷の合計質量[kg]。
        segment_length_mm: セグメント長[mm] (SUBJECTS_DATA.csv の単位に合わせる)。
    """
    a0 = AEParam.MOMENT_OF_INERTIA_A0[sex][segment]
    a1 = AEParam.MOMENT_OF_INERTIA_A1[sex][segment]
    a2 = AEParam.MOMENT_OF_INERTIA_A2[sex][segment]
    length_cm = segment_length_mm / 10.0
    return (a0 + a1 * composite_mass_kg + a2 * length_cm) / 10000.0


def segment_proximal_force(
    seg_mass: float,
    accel_yz: vec2d,
    distal_force_yz: vec2d,
    gravity: float = GRAVITY,
) -> vec2d:
    """セグメントの並進運動方程式から、近位関節がセグメントへ及ぼす反力を逆算する。

    F_proximal + F_distal + m*g(down) = m*a
    => F_proximal = m*a - m*g(down) - F_distal
    """
    accel_y, accel_z = accel_yz
    distal_y, distal_z = distal_force_yz
    proximal_y = seg_mass * accel_y - seg_mass * gravity - distal_y
    proximal_z = seg_mass * accel_z - distal_z
    return (proximal_y, proximal_z)


def segment_proximal_moment(
    seg_inertia: float,
    angular_accel: float,
    distal_moment: float,
    com_to_proximal_yz: vec2d,
    com_to_distal_yz: vec2d,
    proximal_force_yz: vec2d,
    distal_force_yz: vec2d,
) -> float:
    """セグメントの回転運動方程式から、近位関節モーメントを逆算する。

    I*alpha = M_proximal_on_seg + M_distal_on_seg + (r_p x F_p) + (r_d x F_d)
    M_distal_on_seg = -distal_moment (作用・反作用)
    => M_proximal_on_seg = I*alpha + distal_moment - (r_p x F_p) - (r_d x F_d)
    """
    cross_p = cross2(*com_to_proximal_yz, *proximal_force_yz)
    cross_d = cross2(*com_to_distal_yz, *distal_force_yz)
    return seg_inertia * angular_accel + distal_moment - cross_p - cross_d


def ankle_moment_from_grf(ankle_yz: vec2d, cop_yz: vec2d, grf_yz: vec2d) -> float:
    """足部を質量無視のリンクとみなした静的モーメント平衡から足首モーメントを算出する。

    足部にかかるモーメントの合計が 0 になるとして、
    M_ankle(crusが足部に及ぼすモーメント) + (cop-ankle) x GRF = 0
    => M_ankle = -((cop-ankle) x GRF)
    """
    lever_y = cop_yz[0] - ankle_yz[0]
    lever_z = cop_yz[1] - ankle_yz[1]
    return -cross2(lever_y, lever_z, *grf_yz)


def knee_moment(
    *,
    ankle_moment: float,
    ankle_yz: vec2d,
    knee_yz: vec2d,
    crus_com_yz: vec2d,
    crus_com_accel_yz: vec2d,
    crus_mass: float,
    crus_inertia: float,
    crus_angular_accel: float,
    grf_yz: vec2d,
) -> tuple[float, vec2d]:
    """下腿(crus)セグメントの逆動力学から膝モーメントを算出する。

    Returns:
        (膝モーメント, 膝関節が下腿へ及ぼす反力(F_proximal_on_crus))
        反力は大腿側の逆動力学にそのまま渡せる。
    """
    # 足部は質量無視のため、地面反力はそのまま足首を介して下腿へ伝わる。
    distal_force = grf_yz

    proximal_force = segment_proximal_force(
        seg_mass=crus_mass,
        accel_yz=crus_com_accel_yz,
        distal_force_yz=distal_force,
    )

    com_to_knee = (knee_yz[0] - crus_com_yz[0], knee_yz[1] - crus_com_yz[1])
    com_to_ankle = (ankle_yz[0] - crus_com_yz[0], ankle_yz[1] - crus_com_yz[1])

    moment = segment_proximal_moment(
        seg_inertia=crus_inertia,
        angular_accel=crus_angular_accel,
        distal_moment=ankle_moment,
        com_to_proximal_yz=com_to_knee,
        com_to_distal_yz=com_to_ankle,
        proximal_force_yz=proximal_force,
        distal_force_yz=distal_force,
    )
    return (moment, proximal_force)


def hip_moment(
    *,
    knee_moment_value: float,
    knee_yz: vec2d,
    hip_yz: vec2d,
    thigh_com_yz: vec2d,
    thigh_com_accel_yz: vec2d,
    thigh_mass: float,
    thigh_inertia: float,
    thigh_angular_accel: float,
    knee_reaction_force_yz: vec2d,
) -> float:
    """大腿(thigh)セグメントの逆動力学から股関節(腰)モーメントを算出する。

    Args:
        knee_reaction_force_yz: knee_moment() が返した F_proximal_on_crus
            (膝関節が下腿へ及ぼした反力)。作用・反作用で、下腿が大腿に
            及ぼす力 (= 大腿が受け取る distal force) はその反対符号になる。
    """
    distal_force = (-knee_reaction_force_yz[0], -knee_reaction_force_yz[1])

    proximal_force = segment_proximal_force(
        seg_mass=thigh_mass,
        accel_yz=thigh_com_accel_yz,
        distal_force_yz=distal_force,
    )

    com_to_hip = (hip_yz[0] - thigh_com_yz[0], hip_yz[1] - thigh_com_yz[1])
    com_to_knee = (knee_yz[0] - thigh_com_yz[0], knee_yz[1] - thigh_com_yz[1])

    return segment_proximal_moment(
        seg_inertia=thigh_inertia,
        angular_accel=thigh_angular_accel,
        distal_moment=knee_moment_value,
        com_to_proximal_yz=com_to_hip,
        com_to_distal_yz=com_to_knee,
        proximal_force_yz=proximal_force,
        distal_force_yz=distal_force,
    )
