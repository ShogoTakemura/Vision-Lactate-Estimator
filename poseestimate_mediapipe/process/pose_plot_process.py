"""
pose_plot_process.py
--------------------
姿勢分析CSVのグラフ描画プロセス。
posture_analyzed ディレクトリの *_analyzed.csv を読み込み、
各指標の時系列グラフを out/graphs に保存する。

trimming処理廃止に伴い、CoM参照先を bodycom (非トリミング) に変更。
新規指標 (股関節屈曲角, 体幹-大腿相対角, 肩/股傾き, 側屈, CoM深度) のグラフを追加。
"""

import os
import matplotlib.pyplot as plt
from configparser import ConfigParser
from pathlib import Path
import pandas as pd

from poseestimate_mediapipe.process.csv_utils import read_pose_csv


def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    posture_dir = os.path.join(packagepath, 'out', 'posture_analyzed')
    # trimming廃止: bodycom (非トリミング) を参照
    com_dir     = os.path.join(packagepath, 'out', 'bodycom')
    output_dir  = os.path.join(packagepath, 'out', 'graphs')
    os.makedirs(output_dir, exist_ok=True)

    posture_files = list(Path(posture_dir).glob('*.csv'))
    if not posture_files:
        print(f"⚠️ 分析済みCSVが見つかりません: {posture_dir}")
        print("先に 'Analyze body angles and posture' を実行してください。")
        return

    for posture_csv in posture_files:
        print(f"Generating Graphs for: {posture_csv.name}")

        header_lines, df = read_pose_csv(posture_csv)
        frames    = df.index
        base_name = posture_csv.stem.replace('_analyzed', '')

        # 対応 CoM CSV の探索
        search_name  = base_name.replace('_modelbased', '').replace('_modelbase', '')
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
                print(f"  CoM CSV 読み込み失敗: {e}")

        # ----------------------------------------------------------------
        # 共通保存ヘルパー
        # ----------------------------------------------------------------
        def save_plot(suffix, plot_func):
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
            plot_func(ax)
            out_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # ================================================================
        # 1. 体幹前傾角
        # ================================================================
        if 'trunk_lean_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['trunk_lean_angle'], color='black', linewidth=2, label='Trunk Lean')
                ax.set_title(f'Trunk Lean Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_trunk', _p)

        # ================================================================
        # 2. 大腿部傾斜角
        # ================================================================
        if 'r_crotch_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_crotch_angle'], color='red',  linewidth=1.5, label='Right Thigh')
                ax.plot(frames, df['l_crotch_angle'], color='blue', linewidth=1.5, linestyle='--', label='Left Thigh')
                ax.set_title(f'Thigh Inclination Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_thigh', _p)

        # ================================================================
        # 3. 膝関節屈曲角
        # ================================================================
        if 'r_knee_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_knee_angle'], color='red',  linewidth=1.5, label='Right Knee')
                ax.plot(frames, df['l_knee_angle'], color='blue', linewidth=1.5, linestyle='--', label='Left Knee')
                ax.set_title(f'Knee Flexion Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_knee', _p)

        # ================================================================
        # 4. 下腿前傾角
        # ================================================================
        if 'r_ankle_lean_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_ankle_lean_angle'], color='red',  linewidth=1.5, label='Right Shank')
                ax.plot(frames, df['l_ankle_lean_angle'], color='blue', linewidth=1.5, linestyle='--', label='Left Shank')
                ax.set_title(f'Shank Lean Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_shank', _p)

        # ================================================================
        # 5. 股関節屈曲角 ★新規
        # ================================================================
        if 'r_hip_flexion_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_hip_flexion_angle'], color='darkred',  linewidth=1.5, label='Right Hip')
                ax.plot(frames, df['l_hip_flexion_angle'], color='darkblue', linewidth=1.5, linestyle='--', label='Left Hip')
                ax.set_title(f'Hip Flexion Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_hip_flexion', _p)

        # ================================================================
        # 6. 体幹-大腿相対角 ★新規
        # ================================================================
        if 'trunk_thigh_angle' in df.columns:
            def _p(ax):
                ax.plot(frames, df['trunk_thigh_angle'], color='purple', linewidth=2, label='Trunk-Thigh')
                ax.set_title(f'Trunk-Thigh Relative Angle — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_trunk_thigh', _p)

        # ================================================================
        # 7. 肩・股関節の水平傾き + 体幹側屈 (1図にまとめる) ★新規
        # ================================================================
        tilt_cols = ['shoulder_tilt', 'hip_tilt', 'lateral_trunk_tilt']
        if all(c in df.columns for c in tilt_cols):
            def _p(ax):
                ax.plot(frames, df['shoulder_tilt'],       color='orange', linewidth=1.5, label='Shoulder Tilt')
                ax.plot(frames, df['hip_tilt'],            color='green',  linewidth=1.5, linestyle='--', label='Hip Tilt')
                ax.plot(frames, df['lateral_trunk_tilt'],  color='purple', linewidth=1.5, linestyle=':', label='Lateral Trunk Tilt')
                ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')
                ax.set_title(f'Lateral Tilt Angles — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_lateral_tilt', _p)

        # ================================================================
        # 8. ニーイン指標
        # ================================================================
        if 'r_knee_in' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_knee_in'], color='red',  linewidth=1.5, label='Right Knee-In')
                ax.plot(frames, df['l_knee_in'], color='blue', linewidth=1.5, linestyle='--', label='Left Knee-In')
                ax.axhline(0, color='gray', linewidth=0.8)
                ax.set_title(f'Knee-In Index — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (m)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_knee_in', _p)

        # ================================================================
        # 9. 膝の前方突き出し
        # ================================================================
        if 'r_knee_forward' in df.columns:
            def _p(ax):
                ax.plot(frames, df['r_knee_forward'], color='red',  linewidth=1.5, label='Right')
                ax.plot(frames, df['l_knee_forward'], color='blue', linewidth=1.5, linestyle='--', label='Left')
                ax.set_title(f'Knee Forward Displacement — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Displacement (m)')
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_knee_fwd', _p)

        # ================================================================
        # 10. 相対的重心降下率 ★新規
        # ================================================================
        if 'relative_com_depth' in df.columns:
            def _p(ax):
                ax.plot(frames, df['relative_com_depth'], color='teal', linewidth=2, label='Relative CoM Depth')
                ax.set_title(f'Relative CoM Depth (%) — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Depth (%)')
                ax.set_ylim(-5, 105)
                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_com_depth', _p)

        # ================================================================
        # 11. 主要角度サマリーグラフ (全部1枚)
        # ================================================================
        summary_cols = [
            ('trunk_lean_angle',   'Trunk Lean',     'black',   '-'),
            ('r_knee_angle',       'R Knee Flex',    'red',     '-'),
            ('l_knee_angle',       'L Knee Flex',    'blue',    '--'),
            ('r_hip_flexion_angle','R Hip Flex',     'darkred', '-'),
            ('l_hip_flexion_angle','L Hip Flex',     'darkblue','--'),
        ]
        available = [(col, lbl, c, ls) for col, lbl, c, ls in summary_cols if col in df.columns]
        if available:
            def _p(ax):
                for col, lbl, c, ls in available:
                    ax.plot(frames, df[col], color=c, linestyle=ls, linewidth=1.5, label=lbl)
                ax.set_title(f'Joint Angles Summary — {base_name}')
                ax.set_xlabel('Frame'); ax.set_ylabel('Angle (deg)')
                ax.legend(loc='upper right', fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.6)
            save_plot('plot_summary', _p)

        print(f"  -> グラフ出力完了: {base_name}")

    print(f"\n✅ グラフ描画が完了しました。出力先: {output_dir}")
