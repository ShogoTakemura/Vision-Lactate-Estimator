import os
from configparser import ConfigParser
from pathlib import Path
import pandas as pd
from poseestimate_mediapipe.process.csv_utils import read_pose_csv, write_pose_csv
import numpy as np

def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    target_dir = os.path.join(packagepath, 'out', 'modelbased_trimmed')
    output_dir = os.path.join(packagepath, 'out', 'posture_analyzed')
    os.makedirs(output_dir, exist_ok=True)

    csv_files = list(Path(target_dir).glob('*.csv'))

    for csv_path in csv_files:
        print(f"Analyzing Posture: {csv_path.name}")

        header_lines, df = read_pose_csv(csv_path)

        # --- ここから追加：全列を強制的に数値に変換（エラー値は NaN になる） ---
        df = df.apply(pd.to_numeric, errors='coerce')

        # --- 1. 体幹前傾角度 (Trunk Lean Angle) ---
        shoulder_y = (df['RIGHT_SHOULDER_y'] + df['LEFT_SHOULDER_y']) / 2.0
        shoulder_z = (df['RIGHT_SHOULDER_z'] + df['LEFT_SHOULDER_z']) / 2.0
        hip_y = (df['RIGHT_HIP_y'] + df['LEFT_HIP_y']) / 2.0
        hip_z = (df['RIGHT_HIP_z'] + df['LEFT_HIP_z']) / 2.0

        t_y_magni = np.abs(shoulder_y - hip_y)
        t_z_magni = np.abs(shoulder_z - hip_z)
        df['trunk_lean_angle'] = np.degrees(np.arctan2(t_z_magni, t_y_magni))

        # --- 2. 股角度 / 大腿部傾斜角 (Crotch/Thigh Angle) ---
        r_crotch_y_magni = np.abs(df['RIGHT_HIP_y'] - df['RIGHT_KNEE_y'])
        r_crotch_z_magni = np.abs(df['RIGHT_HIP_z'] - df['RIGHT_KNEE_z'])
        df['r_crotch_angle'] = np.degrees(np.arctan2(r_crotch_z_magni, r_crotch_y_magni))

        l_crotch_y_magni = np.abs(df['LEFT_HIP_y'] - df['LEFT_KNEE_y'])
        l_crotch_z_magni = np.abs(df['LEFT_HIP_z'] - df['LEFT_KNEE_z'])
        df['l_crotch_angle'] = np.degrees(np.arctan2(l_crotch_z_magni, l_crotch_y_magni))

        # --- 3. 足関節角度 / 下腿前傾角 (Ankle Angle) ---
        r_ank_y_magni = np.abs(df['RIGHT_ANKLE_y'] - df['RIGHT_KNEE_y'])
        r_ank_z_magni = np.abs(df['RIGHT_ANKLE_z'] - df['RIGHT_KNEE_z'])
        df['r_ankle_angle'] = np.degrees(np.arctan2(r_ank_z_magni, r_ank_y_magni))

        l_ank_y_magni = np.abs(df['LEFT_ANKLE_y'] - df['LEFT_KNEE_y'])
        l_ank_z_magni = np.abs(df['LEFT_ANKLE_z'] - df['LEFT_KNEE_z'])
        df['l_ankle_angle'] = np.degrees(np.arctan2(l_ank_z_magni, l_ank_y_magni))

        # --- 4. 膝関節角度 (Knee Angle) : ベクトルの内積を使用 ---
        # ここがご提示いただいたロジックをPandas用に変換して組み込んだ部分です
        # 右足
        r_hk_y = df['RIGHT_KNEE_y'] - df['RIGHT_HIP_y']
        r_hk_z = df['RIGHT_KNEE_z'] - df['RIGHT_HIP_z']
        r_ka_y = df['RIGHT_ANKLE_y'] - df['RIGHT_KNEE_y']
        r_ka_z = df['RIGHT_ANKLE_z'] - df['RIGHT_KNEE_z']

        r_dot = (r_hk_y * r_ka_y) + (r_hk_z * r_ka_z)
        r_mag = np.sqrt(r_hk_y**2 + r_hk_z**2) * np.sqrt(r_ka_y**2 + r_ka_z**2)
        r_cosval = np.clip(r_dot / r_mag, -1.0, 1.0) # 誤差エラー防止
        df['r_knee_angle'] = np.degrees(np.arccos(r_cosval))

        # 左足
        l_hk_y = df['LEFT_KNEE_y'] - df['LEFT_HIP_y']
        l_hk_z = df['LEFT_KNEE_z'] - df['LEFT_HIP_z']
        l_ka_y = df['LEFT_ANKLE_y'] - df['LEFT_KNEE_y']
        l_ka_z = df['LEFT_ANKLE_z'] - df['LEFT_KNEE_z']

        l_dot = (l_hk_y * l_ka_y) + (l_hk_z * l_ka_z)
        l_mag = np.sqrt(l_hk_y**2 + l_hk_z**2) * np.sqrt(l_ka_y**2 + l_ka_z**2)
        l_cosval = np.clip(l_dot / l_mag, -1.0, 1.0)
        df['l_knee_angle'] = np.degrees(np.arccos(l_cosval))

        # --- 5. ニーイン指標 (Knee-In Index) ---
        df['r_knee_in'] = df['RIGHT_KNEE_x'] - df['RIGHT_ANKLE_x']
        df['l_knee_in'] = df['LEFT_KNEE_x'] - df['LEFT_ANKLE_x']

        # --- 6. 膝の突き出し (Knee Forward Displacement) ---
        df['r_knee_forward'] = df['RIGHT_KNEE_z'] - df['RIGHT_FOOT_INDEX_z']
        df['l_knee_forward'] = df['LEFT_KNEE_z'] - df['LEFT_FOOT_INDEX_z']

        out_path = os.path.join(output_dir, f"{csv_path.stem}_analyzed.csv")
        write_pose_csv(out_path, header_lines, df)

    print(f"\n✅ 姿勢分析が完了しました。出力先: {output_dir}")