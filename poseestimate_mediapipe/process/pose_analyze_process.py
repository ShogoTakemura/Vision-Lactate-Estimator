"""
pose_analyze_process.py
-----------------------
姿勢分析プロセス。
modelbased ディレクトリの座標CSVを読み込み、各種姿勢角度・指標を計算して
posture_analyzed ディレクトリに出力する。

trimming処理を廃止し、modelbased (非トリミング) を直接入力として使用。

【算出指標】
  1. trunk_lean_angle       : 体幹前傾角 (Trunk Lean Angle, deg)
  2. r/l_crotch_angle       : 大腿部傾斜角 (Thigh Inclination, deg)  ※旧: 股角度
  3. r/l_ankle_lean_angle   : 下腿前傾角 (Shank Lean Angle, deg)     ※旧: ankle_angle
  4. r/l_knee_angle         : 膝関節屈曲角 (Knee Flexion, deg) — 内積3点角
  5. r/l_hip_flexion_angle  : 股関節屈曲角 (Hip Flexion, deg) — 内積3点角 ★新規
  6. trunk_thigh_angle      : 体幹-大腿相対角 (Trunk-Thigh Angle, deg) ★新規
  7. shoulder_tilt          : 肩の水平傾き (Shoulder Lateral Tilt, deg) ★新規
  8. hip_tilt               : 股関節の水平傾き (Hip Lateral Tilt, deg) ★新規
  9. lateral_trunk_tilt     : 体幹側屈角 (Lateral Trunk Tilt, deg) ★新規
  10. r/l_knee_in           : ニーイン指標 (Knee-In Index, m)
  11. r/l_knee_forward      : 膝の前方突き出し (Knee Forward Displacement, m)
  12. relative_com_depth    : 相対的重心降下率 (Relative CoM Depth, %) ★新規
"""

import os
from configparser import ConfigParser
from pathlib import Path
import pandas as pd
import numpy as np
from poseestimate_mediapipe.process.csv_utils import read_pose_csv, write_pose_csv


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _three_point_angle(a_y, a_z, b_y, b_z, c_y, c_z) -> np.ndarray:
    """
    3点 A-B-C において頂点 B での内角 (deg) を計算する (YZ平面)。
    A, B, C はそれぞれ Series / ndarray。
    """
    ba_y = a_y - b_y
    ba_z = a_z - b_z
    bc_y = c_y - b_y
    bc_z = c_z - b_z

    dot = ba_y * bc_y + ba_z * bc_z
    mag = np.sqrt(ba_y**2 + ba_z**2) * np.sqrt(bc_y**2 + bc_z**2)
    cosval = np.clip(dot / (mag + 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cosval))


def _inclination_angle(p1_y, p1_z, p2_y, p2_z) -> np.ndarray:
    """
    2点 P1→P2 の傾斜角 (deg): arctan2(Δz, Δy)
    スクワット解析では「鉛直 (Y軸) に対してどれだけ前傾しているか」の近似値。
    """
    dy = np.abs(p2_y - p1_y)
    dz = np.abs(p2_z - p1_z)
    return np.degrees(np.arctan2(dz, dy + 1e-9))


# ---------------------------------------------------------------------------
# メインプロセス
# ---------------------------------------------------------------------------

def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    # ---- trimming廃止: modelbased ディレクトリを直接使用 ----
    target_dir = os.path.join(packagepath, 'out', 'modelbased')
    output_dir = os.path.join(packagepath, 'out', 'posture_analyzed')
    os.makedirs(output_dir, exist_ok=True)

    csv_files = list(Path(target_dir).glob('*.csv'))
    if not csv_files:
        print(f"⚠️ 対象CSVが見つかりません: {target_dir}")
        return

    for csv_path in csv_files:
        print(f"Analyzing Posture: {csv_path.name}")

        header_lines, df = read_pose_csv(csv_path)
        df = df.apply(pd.to_numeric, errors='coerce')

        # ================================================================
        # 1. 体幹前傾角 (Trunk Lean Angle)
        #    肩中点 → 股関節中点ベクトルの鉛直に対する前傾度
        # ================================================================
        shoulder_y = (df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) / 2.0
        shoulder_z = (df['RIGHT_SHOULDER_z'] + df['LEFT_SHOULDER_z']) / 2.0
        hip_y      = (df['RIGHT_HIP_y']      + df['LEFT_HIP_y'])      / 2.0
        hip_z      = (df['RIGHT_HIP_z']      + df['LEFT_HIP_z'])      / 2.0

        df['trunk_lean_angle'] = _inclination_angle(hip_y, hip_z, shoulder_y, shoulder_z)

        # ================================================================
        # 2. 大腿部傾斜角 (Thigh Inclination)
        #    股関節 → 膝関節ベクトルの鉛直に対する傾斜
        # ================================================================
        df['r_crotch_angle'] = _inclination_angle(
            df['RIGHT_HIP_y'], df['RIGHT_HIP_z'],
            df['RIGHT_KNEE_y'], df['RIGHT_KNEE_z']
        )
        df['l_crotch_angle'] = _inclination_angle(
            df['LEFT_HIP_y'], df['LEFT_HIP_z'],
            df['LEFT_KNEE_y'], df['LEFT_KNEE_z']
        )

        # ================================================================
        # 3. 下腿前傾角 (Shank Lean Angle)
        #    膝関節 → 足関節ベクトルの鉛直に対する傾斜
        # ================================================================
        df['r_ankle_lean_angle'] = _inclination_angle(
            df['RIGHT_KNEE_y'], df['RIGHT_KNEE_z'],
            df['RIGHT_ANKLE_y'], df['RIGHT_ANKLE_z']
        )
        df['l_ankle_lean_angle'] = _inclination_angle(
            df['LEFT_KNEE_y'], df['LEFT_KNEE_z'],
            df['LEFT_ANKLE_y'], df['LEFT_ANKLE_z']
        )

        # ================================================================
        # 4. 膝関節屈曲角 (Knee Flexion Angle) — 内積3点角
        #    頂点: 膝  / 2点: 股関節・足関節
        # ================================================================
        df['r_knee_angle'] = _three_point_angle(
            df['RIGHT_HIP_y'],   df['RIGHT_HIP_z'],
            df['RIGHT_KNEE_y'],  df['RIGHT_KNEE_z'],
            df['RIGHT_ANKLE_y'], df['RIGHT_ANKLE_z']
        )
        df['l_knee_angle'] = _three_point_angle(
            df['LEFT_HIP_y'],   df['LEFT_HIP_z'],
            df['LEFT_KNEE_y'],  df['LEFT_KNEE_z'],
            df['LEFT_ANKLE_y'], df['LEFT_ANKLE_z']
        )

        # ================================================================
        # 5. 股関節屈曲角 (Hip Flexion Angle) ★新規
        #    頂点: 股関節  / 2点: 肩（体幹）・膝
        # ================================================================
        df['r_hip_flexion_angle'] = _three_point_angle(
            df['RIGHT_SHOULDER_y'], df['RIGHT_SHOULDER_z'],
            df['RIGHT_HIP_y'],      df['RIGHT_HIP_z'],
            df['RIGHT_KNEE_y'],     df['RIGHT_KNEE_z']
        )
        df['l_hip_flexion_angle'] = _three_point_angle(
            df['LEFT_SHOULDER_y'], df['LEFT_SHOULDER_z'],
            df['LEFT_HIP_y'],      df['LEFT_HIP_z'],
            df['LEFT_KNEE_y'],     df['LEFT_KNEE_z']
        )

        # ================================================================
        # 6. 体幹-大腿相対角 (Trunk-Thigh Angle) ★新規
        #    体幹前傾角 + 大腿部傾斜角 = 実質的な股関節前面の開き角に相当
        #    (左右平均で1指標)
        # ================================================================
        avg_crotch = (df['r_crotch_angle'] + df['l_crotch_angle']) / 2.0
        df['trunk_thigh_angle'] = df['trunk_lean_angle'] + avg_crotch

        # ================================================================
        # 7. 肩の水平傾き (Shoulder Lateral Tilt) ★新規
        #    左右肩のY座標差 → 側方への傾き (deg)
        #    正値: 右肩が高い、負値: 左肩が高い
        # ================================================================
        shoulder_x_diff = df['RIGHT_SHOULDER_x'] - df['LEFT_SHOULDER_x']
        shoulder_y_diff = df['RIGHT_SHOULDER_y'] - df['LEFT_SHOULDER_y']
        df['shoulder_tilt'] = np.degrees(
            np.arctan2(shoulder_y_diff, np.abs(shoulder_x_diff) + 1e-9)
        )

        # ================================================================
        # 8. 股関節の水平傾き (Hip Lateral Tilt) ★新規
        #    左右股関節のY座標差
        # ================================================================
        hip_x_diff = df['RIGHT_HIP_x'] - df['LEFT_HIP_x']
        hip_y_diff = df['RIGHT_HIP_y'] - df['LEFT_HIP_y']
        df['hip_tilt'] = np.degrees(
            np.arctan2(hip_y_diff, np.abs(hip_x_diff) + 1e-9)
        )

        # ================================================================
        # 9. 体幹側屈角 (Lateral Trunk Tilt) ★新規
        #    肩中点 → 股関節中点ベクトルの水平(X方向)傾き
        # ================================================================
        shoulder_x = (df['RIGHT_SHOULDER_x'] + df['LEFT_SHOULDER_x']) / 2.0
        hip_x      = (df['RIGHT_HIP_x']      + df['LEFT_HIP_x'])      / 2.0
        lat_dx = shoulder_x - hip_x
        lat_dy = shoulder_y - hip_y  # Y下向き座標系
        df['lateral_trunk_tilt'] = np.degrees(
            np.arctan2(lat_dx, np.abs(lat_dy) + 1e-9)
        )

        # ================================================================
        # 10. ニーイン指標 (Knee-In Index)
        #     正値: 膝が足先より内側 → ニーイン傾向
        # ================================================================
        df['r_knee_in'] = df['RIGHT_KNEE_x'] - df['RIGHT_ANKLE_x']
        df['l_knee_in'] = df['LEFT_KNEE_x']  - df['LEFT_ANKLE_x']

        # ================================================================
        # 11. 膝の前方突き出し (Knee Forward Displacement)
        #     膝のZ座標 - 足先Z座標 (前方正)
        # ================================================================
        df['r_knee_forward'] = df['RIGHT_KNEE_z'] - df['RIGHT_FOOT_INDEX_z']
        df['l_knee_forward'] = df['LEFT_KNEE_z']  - df['LEFT_FOOT_INDEX_z']

        # ================================================================
        # 12. 相対的重心降下率 (Relative CoM Depth) ★新規
        #     肩Y座標を重心代理として使用し、
        #     セット内最高点を 0%, 最低点を 100% として正規化
        # ================================================================
        com_proxy = (df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) / 2.0
        com_min   = com_proxy.min()  # 最高点 (Y下向きなので最小値 = 最も高い)
        com_max   = com_proxy.max()  # 最低点 (ボトム位置)
        com_range = com_max - com_min
        if com_range > 1e-6:
            df['relative_com_depth'] = (com_proxy - com_min) / com_range * 100.0
        else:
            df['relative_com_depth'] = 0.0

        # ================================================================
        # 出力
        # ================================================================
        out_path = os.path.join(output_dir, f"{csv_path.stem}_analyzed.csv")
        write_pose_csv(out_path, header_lines, df)
        print(f"  -> 出力: {out_path}")

    print(f"\n✅ 姿勢分析が完了しました。出力先: {output_dir}")