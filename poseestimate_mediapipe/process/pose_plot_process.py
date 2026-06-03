import os
import matplotlib.pyplot as plt
from configparser import ConfigParser
from pathlib import Path
import pandas as pd

# 既存のCSV読み込みユーティリティを利用
from poseestimate_mediapipe.process.csv_utils import read_pose_csv

def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    # 読み込み元1：姿勢分析CSV (角度データ)
    posture_dir = os.path.join(packagepath, 'out', 'posture_analyzed')
    # 読み込み元2：重心CSV (CoMデータ)
    com_dir = os.path.join(packagepath, 'out', 'bodycom_trimmed')
    
    # 出力先
    output_dir = os.path.join(packagepath, 'out', 'graphs')
    os.makedirs(output_dir, exist_ok=True)

    posture_files = list(Path(posture_dir).glob('*.csv'))

    if not posture_files:
        print(f"⚠️ 分析済みのCSVファイルが見つかりません: {posture_dir}")
        print("先に 'Analyze body angles and posture' を実行してください。")
        return

    for posture_csv in posture_files:
        print(f"Generating Graphs for: {posture_csv.name}")

        # 1. 姿勢分析CSVの読み込み
        header_lines, df_posture = read_pose_csv(posture_csv)
        frames = df_posture.index
        base_name = posture_csv.stem.replace('_analyzed', '')
        
        # 2. 対応する重心(CoM)CSVの探索と読み込み
        search_name = base_name.replace('_modelbased_trimmed', '').replace('_modelbase_trimmed', '')
        
        com_csv_path = None
        for com_file in Path(com_dir).glob('*.csv'):
            if search_name in com_file.name:
                com_csv_path = com_file
                break
        
        df_com = None
        if com_csv_path and com_csv_path.exists():
            try:
                df_com = pd.read_csv(com_csv_path)
            except Exception as e:
                print(f"CoM CSVの読み込みに失敗しました: {e}")

        # --- グラフの個別描画と保存 ---

        # 共通の設定関数
        def save_plot(filename_suffix, plot_func):
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
            plot_func(ax)
            out_path = os.path.join(output_dir, f"{base_name}_{filename_suffix}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # 1. 体幹前傾角度 (Trunk Lean Angle)
        if 'trunk_lean_angle' in df_posture.columns:
            def plot_trunk(ax):
                ax.plot(frames, df_posture['trunk_lean_angle'], label='Trunk Lean Angle', color='black', linewidth=2)
                ax.set_title(f'Trunk Lean Angle - {base_name}', fontsize=14)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Angle (degrees)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            save_plot('plot_trunk', plot_trunk)

        # 2. 股角度 (Crotch / Thigh Angle)
        if 'r_crotch_angle' in df_posture.columns and 'l_crotch_angle' in df_posture.columns:
            def plot_crotch(ax):
                ax.plot(frames, df_posture['r_crotch_angle'], label='Right Crotch', color='red', linewidth=1.5)
                ax.plot(frames, df_posture['l_crotch_angle'], label='Left Crotch', color='blue', linestyle='--', linewidth=1.5)
                ax.set_title(f'Crotch / Thigh Angle - {base_name}', fontsize=14)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Angle (degrees)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            save_plot('plot_crotch', plot_crotch)

        # 3. 膝関節角度 (Knee Angle)
        if 'r_knee_angle' in df_posture.columns and 'l_knee_angle' in df_posture.columns:
            def plot_knee(ax):
                ax.plot(frames, df_posture['r_knee_angle'], label='Right Knee', color='red', linewidth=1.5)
                ax.plot(frames, df_posture['l_knee_angle'], label='Left Knee', color='blue', linestyle='--', linewidth=1.5)
                ax.set_title(f'Knee Flexion Angle - {base_name}', fontsize=14)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Angle (degrees)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            save_plot('plot_knee', plot_knee)

        # 4. 足関節角度 (Ankle Angle)
        if 'r_ankle_angle' in df_posture.columns and 'l_ankle_angle' in df_posture.columns:
            def plot_ankle(ax):
                ax.plot(frames, df_posture['r_ankle_angle'], label='Right Ankle', color='red', linewidth=1.5)
                ax.plot(frames, df_posture['l_ankle_angle'], label='Left Ankle', color='blue', linestyle='--', linewidth=1.5)
                ax.set_title(f'Ankle / Lower Leg Angle - {base_name}', fontsize=14)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Angle (degrees)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            save_plot('plot_ankle', plot_ankle)

        # 5. 重心位置 (CoM Y)
        if df_com is not None and 'CoM_y' in df_com.columns:
            com_frames = range(len(df_com))
            def plot_com(ax):
                ax.plot(com_frames, df_com['CoM_y'], label='CoM Y (Vertical)', color='green', linewidth=2)
                ax.set_title(f'Center of Mass (Vertical Position) - {base_name}', fontsize=14)
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Position Y (m)')
                ax.legend(loc='upper right')
                ax.grid(True, linestyle='--', alpha=0.7)
            save_plot('plot_com', plot_com)

    print(f"\n✅ グラフの個別出力が完了しました！ 出力先: {output_dir}")