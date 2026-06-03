"""
calculate_work_process.py
--------------------------
スクワット仕事量計算プロセス。

rep CSV の読み込み元:
  C:\\Users\\ironm\\squat_analyze\\frame_viewer_tool\\reps\\processed\\
  ファイル形式: {stem}_rep.csv
  列: rep, start_frame, bottom_frame, end_frame

座標 CSV の読み込み元:
  out/modelbased/  (trimming廃止)

出力:
  out/work_calculated/{stem}_squat_work_data.csv  (レップ別)
  out/work_calculated/TOTAL_WORK_DATABASE.csv     (セット単位DB)
  out/work_calculated/REP_DATABASE.csv            (全rep単位DB ← 追加)
  out/graphs/peak/{stem}_peaks.png
"""

import os
import re
import pandas as pd
import numpy as np
from configparser import ConfigParser
from pathlib import Path
import matplotlib.pyplot as plt

from poseestimate_mediapipe.process.csv_utils import read_pose_csv

# ----------------------------------------------------------------
# rep CSVのフォルダパス (環境に合わせて変更)
# ----------------------------------------------------------------
REP_CSV_DIR = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\reps\processed"

_SUFFIX_RE = re.compile(r'(_correct|_modelbased|_modelbase|_analyzed)+$')

def _stem_to_base(stem: str) -> str:
    return _SUFFIX_RE.sub('', stem)


def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    target_dir            = os.path.join(packagepath, 'out', 'modelbased')
    com_workset_path      = os.path.join(packagepath, 'config', 'COM_WORKSET.csv')
    modelbased_workset_path = os.path.join(packagepath, 'config', 'MODELBASED_WORKSET.csv')
    subjects_data_path    = os.path.join(packagepath, 'config', 'SUBJECTS_DATA.csv')

    output_dir     = os.path.join(packagepath, 'out', 'work_calculated')
    peak_graph_dir = os.path.join(packagepath, 'out', 'graphs', 'peak')
    os.makedirs(output_dir,     exist_ok=True)
    os.makedirs(peak_graph_dir, exist_ok=True)

    csv_files = list(Path(target_dir).glob('*.csv'))
    if not csv_files:
        print(f"⚠️ 対象の座標CSVが見つかりません: {target_dir}")
        return

    df_com      = pd.read_csv(com_workset_path)       if os.path.exists(com_workset_path)       else pd.DataFrame()
    df_model    = pd.read_csv(modelbased_workset_path) if os.path.exists(modelbased_workset_path) else pd.DataFrame()
    df_subjects = pd.read_csv(subjects_data_path)     if os.path.exists(subjects_data_path)     else pd.DataFrame()

    g   = 9.80665
    fps = 30.0

    total_work_db_path = os.path.join(output_dir, 'TOTAL_WORK_DATABASE.csv')
    rep_db_path        = os.path.join(output_dir, 'REP_DATABASE.csv')

    # 実行開始時に既存DBをリセット
    for p in [total_work_db_path, rep_db_path]:
        if os.path.exists(p):
            os.remove(p)

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path.name}")

        base_name    = _stem_to_base(csv_path.stem)
        subject_name = base_name.split('_')[1] if '_' in base_name else base_name

        # ---- rep CSV 探索 ----
        rep_csv_path = os.path.join(REP_CSV_DIR, f"{base_name}_rep.csv")
        if not os.path.exists(rep_csv_path):
            print(f"  ⚠️ rep CSV が見つかりません: {base_name}_rep.csv  スキップします。")
            continue

        # ---- 荷重 解決 ----
        weight_kg = None
        mass_kg   = None

        if not df_com.empty:
            match = df_com[df_com['filename'].str.contains(base_name, na=False)]
            if not match.empty:
                weight_kg    = match.iloc[0]['load']
                subject_name = match.iloc[0].get('subject', subject_name)

        if weight_kg is None and not df_model.empty and not df_subjects.empty:
            match_model = df_model[df_model['filename'].str.contains(base_name, na=False)]
            if not match_model.empty:
                s_id = match_model.iloc[0]['subject_id']
                match_subject = df_subjects[df_subjects['segment_id'] == s_id]
                if not match_subject.empty:
                    weight_kg    = match_subject.iloc[0]['load']
                    mass_kg      = match_subject.iloc[0].get('mass', None)
                    subject_name = match_subject.iloc[0]['name']

        if weight_kg is None or pd.isna(weight_kg):
            weight_kg = 0.0
            print(f"  ⚠️ 荷重が特定できないため 0 kg で計算します。")

        print(f"  Subject: {subject_name}  |  Load: {weight_kg} kg")

        # ---- 座標 CSV 読み込み ----
        try:
            header_lines, df = read_pose_csv(csv_path)
            df = df.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            print(f"  ⚠️ 座標CSV読み込みエラー: {e}")
            continue

        dumbbell_y = ((df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0).values
        frames     = df['count'].values if 'count' in df.columns else df.index.values

        # ---- rep CSV 読み込み ----
        try:
            df_reps = pd.read_csv(rep_csv_path)
            df_reps = df_reps.dropna(subset=['start_frame', 'end_frame'])
        except Exception as e:
            print(f"  ⚠️ rep CSV読み込みエラー: {e}")
            continue

        col_start  = 'start_frame'  if 'start_frame'  in df_reps.columns else 'Start_Frame'
        col_bottom = 'bottom_frame' if 'bottom_frame' in df_reps.columns else 'Top_Frame'
        col_end    = 'end_frame'    if 'end_frame'    in df_reps.columns else 'End_Frame'
        col_rep    = 'rep'          if 'rep'          in df_reps.columns else 'Rep'

        rep_results    = []
        set_total_work = 0.0
        set_total_ke   = 0.0
        plot_starts    = []
        plot_bottoms   = []
        plot_ends      = []

        for i, row in df_reps.iterrows():
            abs_start  = int(row[col_start])
            abs_bottom = int(row[col_bottom]) if col_bottom in df_reps.columns and not pd.isna(row[col_bottom]) else -1
            abs_end    = int(row[col_end])

            if 'count' in df.columns:
                start_matches = df.index[df['count'] >= abs_start]
                end_matches   = df.index[df['count'] <= abs_end]
                if len(start_matches) == 0 or len(end_matches) == 0:
                    continue
                start_idx = int(start_matches.min())
                end_idx   = int(end_matches.max())
                if abs_bottom >= 0:
                    bot_matches = df.index[df['count'] == abs_bottom]
                    top_idx = int(bot_matches[0]) if len(bot_matches) > 0 else start_idx + int(np.argmax(dumbbell_y[start_idx:end_idx+1]))
                else:
                    top_idx = start_idx + int(np.argmax(dumbbell_y[start_idx:end_idx+1]))
            else:
                start_idx = max(0, abs_start)
                end_idx   = min(len(dumbbell_y) - 1, abs_end)
                top_idx   = max(start_idx, min(end_idx, abs_bottom)) if abs_bottom >= 0 else start_idx + int(np.argmax(dumbbell_y[start_idx:end_idx+1]))

            if start_idx >= end_idx:
                continue

            plot_starts.append(start_idx)
            plot_bottoms.append(top_idx)
            plot_ends.append(end_idx)

            start_y = dumbbell_y[start_idx]
            top_y   = dumbbell_y[top_idx]
            end_y   = dumbbell_y[end_idx]

            h1 = abs(top_y - start_y)
            h2 = abs(end_y - top_y)
            total_displacement = h1 + h2

            work_down  = weight_kg * g * h1
            work_up    = weight_kg * g * h2
            total_work = work_down + work_up

            time_eccentric   = (top_idx   - start_idx) / fps
            time_concentric  = (end_idx   - top_idx)   / fps
            velocity_ecc     = h1 / time_eccentric  if time_eccentric  > 0 else 0.0
            velocity_con     = h2 / time_concentric if time_concentric > 0 else 0.0
            kinetic_energy   = 0.5 * weight_kg * (velocity_con ** 2)

            set_total_work += total_work
            set_total_ke   += kinetic_energy

            rep_num = int(row[col_rep]) if col_rep in df_reps.columns else i + 1

            rep_results.append({
                'File':                     base_name,
                'Subject':                  subject_name,
                'Weight(kg)':               weight_kg,
                'Rep':                      rep_num,
                'Start_Frame':              frames[start_idx],
                'Top_Frame':                frames[top_idx],
                'End_Frame':                frames[end_idx],
                'Start_Y(m)':               round(start_y, 4),
                'Top_Y(m)':                 round(top_y,   4),
                'End_Y(m)':                 round(end_y,   4),
                'Displacement_Down_h1(m)':  round(h1,    4),
                'Displacement_Up_h2(m)':    round(h2,    4),
                'Total_Displacement(m)':    round(total_displacement, 4),
                'Time_Eccentric(s)':        round(time_eccentric,  4),
                'Time_Concentric(s)':       round(time_concentric, 4),
                'Velocity_Eccentric(m/s)':  round(velocity_ecc, 4),
                'Velocity_Concentric(m/s)': round(velocity_con, 4),
                'Work_Down_U1(J)':          round(work_down,  2),
                'Work_Up_U2(J)':            round(work_up,    2),
                'Total_Work_U(J)':          round(total_work, 2),
                'Kinetic_Energy_Up(J)':     round(kinetic_energy, 2),
            })

        if not rep_results:
            print(f"  ⚠️ 有効なレップデータがありませんでした。")
            continue

        # ---- ピーク検出グラフ ----
        fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
        ax.plot(frames, -dumbbell_y, color='gray', label='Shoulder Y (inv)')
        ax.plot(frames[plot_starts],  -dumbbell_y[plot_starts],  'go', markersize=8,  label='Start')
        ax.plot(frames[plot_bottoms], -dumbbell_y[plot_bottoms], 'rx', markersize=10, markeredgewidth=2, label='Bottom')
        ax.plot(frames[plot_ends],    -dumbbell_y[plot_ends],    'bo', markersize=8,  label='End')
        ax.set_title(f'Squat Rep Detection: {base_name}', fontsize=14)
        ax.set_xlabel('Frame'); ax.set_ylabel('Shoulder Y (inverted)')
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(peak_graph_dir, f"{base_name}_peaks.png"), dpi=150)
        plt.close(fig)

        # ---- 個別 CSV ----
        df_rep = pd.DataFrame(rep_results)
        df_rep.to_csv(os.path.join(output_dir, f"{base_name}_squat_work_data.csv"),
                      index=False, encoding='utf-8-sig')

        # ---- REP_DATABASE 追記（rep単位・全ファイル結合）----
        rep_db_cols = [
            'File', 'Subject', 'Weight(kg)', 'Rep',
            'Start_Frame', 'Top_Frame', 'End_Frame',
            'Displacement_Down_h1(m)', 'Displacement_Up_h2(m)', 'Total_Displacement(m)',
            'Time_Eccentric(s)', 'Time_Concentric(s)',
            'Velocity_Eccentric(m/s)', 'Velocity_Concentric(m/s)',
            'Work_Down_U1(J)', 'Work_Up_U2(J)', 'Total_Work_U(J)', 'Kinetic_Energy_Up(J)',
        ]
        df_rep_db = df_rep[rep_db_cols]
        if os.path.exists(rep_db_path):
            df_rep_db.to_csv(rep_db_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_rep_db.to_csv(rep_db_path, mode='w', header=True,  index=False, encoding='utf-8-sig')

        # ---- TOTAL_WORK_DATABASE 追記（セット単位）----
        summary = pd.DataFrame([{
            'File_Name':                   base_name,
            'Subject':                     subject_name,
            'Weight(kg)':                  weight_kg,
            'Total_Reps':                  len(rep_results),
            'Set_Total_Work(J)':           round(set_total_work, 2),
            'Set_Total_Kinetic_Energy(J)': round(set_total_ke,   2),
        }])
        if os.path.exists(total_work_db_path):
            summary.to_csv(total_work_db_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            summary.to_csv(total_work_db_path, mode='w', header=True,  index=False, encoding='utf-8-sig')

        print(f"  -> {len(rep_results)} レップ完了")

    print(f"\n✅ 全処理完了！")
    print(f"   TOTAL_WORK_DATABASE : {total_work_db_path}")
    print(f"   REP_DATABASE        : {rep_db_path}")