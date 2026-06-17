"""
rep_posture_analyze_process.py
-------------------------------
レップごとの姿勢分析プロセス。

rep CSV (processed フォルダ) の start_frame / bottom_frame / end_frame を使い、
posture_analyzed/*_analyzed.csv の角度データをレップ単位でスライスして
各レップの要約統計を出力する。

rep CSV 列名: rep, start_frame, bottom_frame, end_frame
  ※ bottom_frame が直接利用できるため top_rel_idx の推定は不要

【出力】
  out/rep_posture/{stem}_rep_posture.csv      : ファイル別レップ統計
  out/rep_posture/REP_POSTURE_DATABASE.csv    : 全ファイル結合DB
  out/rep_posture/graphs/{stem}_rep_angles_bottom.png
"""

import os
import re
from configparser import ConfigParser
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from poseestimate_mediapipe.process.csv_utils import read_pose_csv
from squat_core.paths import FRAME_VIEWER_ROOT

# ----------------------------------------------------------------
# rep CSVのフォルダパス
# ----------------------------------------------------------------
REP_CSV_DIR = str(FRAME_VIEWER_ROOT / "reps" / "processed")


# 座標CSV / posture CSV の stem から付加サフィックスを除去して base_name を得る
# _correct_modelbased / _modelbased_correct 等、順序不問で全除去
_SUFFIX_RE = re.compile(r'(_correct|_modelbased|_modelbase|_analyzed)+$')

def _stem_to_base(stem: str) -> str:
    """座標CSV / posture CSV の stem から rep CSV 名の基底部分を返す。"""
    return _SUFFIX_RE.sub('', stem)




# ----------------------------------------------------------------
# 分析対象の角度・指標列
# ----------------------------------------------------------------
ANGLE_COLS = [
    'trunk_lean_angle',
    'r_knee_angle',
    'l_knee_angle',
    'r_hip_flexion_angle',
    'l_hip_flexion_angle',
    'r_crotch_angle',
    'l_crotch_angle',
    'r_ankle_lean_angle',
    'l_ankle_lean_angle',
    'trunk_thigh_angle',
    'shoulder_tilt',
    'hip_tilt',
    'lateral_trunk_tilt',
    'r_knee_in',
    'l_knee_in',
    'r_knee_forward',
    'l_knee_forward',
    'relative_com_depth',
]


def _summarize_rep(df_rep: pd.DataFrame, bottom_rel_idx: int, col: str) -> dict:
    """
    1レップ分の列データから要約統計を計算する。

    Parameters
    ----------
    df_rep          : レップ区間の DataFrame (start〜end, reset_index済み)
    bottom_rel_idx  : df_rep 内でのボトムフレームの相対インデックス
    col             : 集計対象の列名

    Returns
    -------
    dict: mean / max / min / bottom / descent_mean / ascent_mean
    """
    if col not in df_rep.columns:
        return {}

    series = df_rep[col].dropna().values
    if len(series) == 0:
        nan = np.nan
        return {
            f"{col}_mean":         nan, f"{col}_max":          nan,
            f"{col}_min":          nan, f"{col}_bottom":       nan,
            f"{col}_descent_mean": nan, f"{col}_ascent_mean":  nan,
        }

    bottom_val = df_rep[col].iloc[bottom_rel_idx] if bottom_rel_idx < len(df_rep) else np.nan
    descent    = df_rep[col].iloc[:bottom_rel_idx + 1].values
    ascent     = df_rep[col].iloc[bottom_rel_idx:].values

    def _safe_mean(arr):
        return round(float(np.nanmean(arr)), 4) if len(arr) > 0 else np.nan

    return {
        f"{col}_mean":         round(float(np.nanmean(series)), 4),
        f"{col}_max":          round(float(np.nanmax(series)),  4),
        f"{col}_min":          round(float(np.nanmin(series)),  4),
        f"{col}_bottom":       round(float(bottom_val), 4) if not (isinstance(bottom_val, float) and np.isnan(bottom_val)) else np.nan,
        f"{col}_descent_mean": _safe_mean(descent),
        f"{col}_ascent_mean":  _safe_mean(ascent),
    }


def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    posture_dir = os.path.join(packagepath, 'out', 'posture_analyzed')
    output_dir  = os.path.join(packagepath, 'out', 'rep_posture')
    graph_dir   = os.path.join(output_dir,  'graphs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(graph_dir,  exist_ok=True)

    db_path = os.path.join(output_dir, 'REP_POSTURE_DATABASE.csv')
    if os.path.exists(db_path):
        os.remove(db_path)

    # subjects情報 (荷重・被験者名取得用)
    subjects_path   = os.path.join(packagepath, 'config', 'SUBJECTS_DATA.csv')
    modelbased_path = os.path.join(packagepath, 'config', 'MODELBASED_WORKSET.csv')
    df_subjects = pd.read_csv(subjects_path)   if os.path.exists(subjects_path)   else pd.DataFrame()
    df_model    = pd.read_csv(modelbased_path) if os.path.exists(modelbased_path) else pd.DataFrame()

    posture_files = list(Path(posture_dir).glob('*_analyzed.csv'))
    if not posture_files:
        print(f"警告: posture_analyzed CSVが見つかりません: {posture_dir}")
        print("先に 'Analyze body angles and posture' を実行してください。")
        return

    for posture_csv in posture_files:
        # posture CSV の stem から base_name を解決
        # 例: 20250121_uchino-RIR1-nonmax-set3_correct_analyzed
        #   → _analyzed 除去 → _correct/_modelbased 等除去 → base_name
        base_name    = _stem_to_base(posture_csv.stem)
        subject_name = base_name.split('_')[1] if '_' in base_name else base_name

        print(f"\nRep Posture Analysis: {base_name}")

        # ---- rep CSV 探索 ----
        rep_csv_name = f"{base_name}_rep.csv"
        rep_csv_path = os.path.join(REP_CSV_DIR, rep_csv_name)
        if not os.path.exists(rep_csv_path):
            print(f"  警告: rep CSV が見つかりません: {rep_csv_name}  スキップします。")
            continue

        try:
            df_reps = pd.read_csv(rep_csv_path)
        except Exception as e:
            print(f"  警告: rep CSV 読み込み失敗: {e}")
            continue

        # 列名を柔軟に解決 (新形式: start_frame/bottom_frame, 旧形式: Start_Frame/Top_Frame)
        col_rep    = 'rep'          if 'rep'          in df_reps.columns else 'Rep'
        col_start  = 'start_frame'  if 'start_frame'  in df_reps.columns else 'Start_Frame'
        col_bottom = 'bottom_frame' if 'bottom_frame' in df_reps.columns else 'Top_Frame'
        col_end    = 'end_frame'    if 'end_frame'    in df_reps.columns else 'End_Frame'

        df_reps = df_reps.dropna(subset=[col_start, col_end])

        # ---- posture CSV 読み込み ----
        try:
            header_lines, df_pose = read_pose_csv(posture_csv)
            df_pose = df_pose.apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            print(f"  警告: posture CSV 読み込み失敗: {e}")
            continue

        # フレーム番号列を特定
        if 'count' in df_pose.columns:
            frame_col = 'count'
        else:
            df_pose = df_pose.reset_index()
            df_pose.rename(columns={'index': 'count'}, inplace=True)
            frame_col = 'count'

        # ---- 被験者情報・荷重 ----
        load_kg  = np.nan
        mass_kg  = np.nan
        if not df_model.empty and not df_subjects.empty:
            match_model = df_model[df_model['filename'].str.contains(base_name, na=False)]
            if not match_model.empty:
                s_id = match_model.iloc[0]['subject_id']
                match_subject = df_subjects[df_subjects['segment_id'] == s_id]
                if not match_subject.empty:
                    load_kg      = match_subject.iloc[0]['load']
                    subject_name = match_subject.iloc[0]['name']
                    mass_kg      = match_subject.iloc[0].get('mass', np.nan)

        # ---- 対象列を存在するものに絞る ----
        target_cols = [c for c in ANGLE_COLS if c in df_pose.columns]
        if not target_cols:
            print(f"  警告: 対象角度列が見つかりません。")
            continue

        rep_records = []

        for i, row in df_reps.iterrows():
            rep_num  = int(row[col_rep]) if col_rep in df_reps.columns else i + 1

            # ボトムフレームが欠損している場合はスキップして警告を出す
            if pd.isna(row[col_bottom]):
                print(f"    Rep {rep_num}: ボトムフレームのデータが欠損しているためスキップします。")
                continue

            start_f  = int(row[col_start])
            bottom_f = int(row[col_bottom])
            end_f    = int(row[col_end])

            # フレーム番号でスライス
            mask   = (df_pose[frame_col] >= start_f) & (df_pose[frame_col] <= end_f)
            df_rep = df_pose[mask].reset_index(drop=True)

            if len(df_rep) == 0:
                print(f"    Rep {rep_num}: フレーム {start_f}~{end_f} がCSV範囲外 -> スキップ")
                continue

            # ボトムフレームの相対インデックス (bottom_frame が直接指定されているので推定不要)
            bot_matches    = df_rep.index[df_rep[frame_col] == bottom_f]
            bottom_rel_idx = int(bot_matches[0]) if len(bot_matches) > 0 else len(df_rep) // 2

            n_descent = bottom_rel_idx + 1
            n_ascent  = len(df_rep) - bottom_rel_idx

            rec = {
                'File':           base_name,
                'Subject':        subject_name,
                'Load(kg)':       load_kg,
                'Mass(kg)':       mass_kg,
                'Rep':            rep_num,
                'Start_Frame':    start_f,
                'Bottom_Frame':   bottom_f,
                'End_Frame':      end_f,
                'Rep_Frames':     len(df_rep),
                'Descent_Frames': n_descent,
                'Ascent_Frames':  n_ascent,
                'Rep_Duration_s': round(len(df_rep) / 30.0, 3),
                'Descent_Time_s': round(n_descent   / 30.0, 3),
                'Ascent_Time_s':  round(n_ascent    / 30.0, 3),
            }

            for col in target_cols:
                rec.update(_summarize_rep(df_rep, bottom_rel_idx, col))

            rep_records.append(rec)

        if not rep_records:
            print(f"  警告: 有効なレップデータがありませんでした。")
            continue

        df_result = pd.DataFrame(rep_records)

        # ---- 個別CSV出力 ----
        out_csv = os.path.join(output_dir, f"{base_name}_rep_posture.csv")
        df_result.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"  -> {len(rep_records)} レップ出力: {os.path.basename(out_csv)}")

        # ---- DB追記 ----
        if os.path.exists(db_path):
            df_result.to_csv(db_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df_result.to_csv(db_path, mode='w', header=True,  index=False, encoding='utf-8-sig')

        # ---- グラフ出力 ----
        _plot_rep_bottom_angles(df_result, base_name, graph_dir)

    print(f"\n完了: レップ別姿勢分析が完了しました。出力先: {output_dir}")


def _plot_rep_bottom_angles(df_result: pd.DataFrame, base_name: str, graph_dir: str):
    """
    レップ×ボトム時の主要関節角度をバーグラフで可視化する。
    """
    plot_targets = [
        ('trunk_lean_angle_bottom',    'Trunk Lean',   'black'),
        ('r_knee_angle_bottom',        'R Knee Flex',  'red'),
        ('l_knee_angle_bottom',        'L Knee Flex',  'blue'),
        ('r_hip_flexion_angle_bottom', 'R Hip Flex',   'darkred'),
        ('l_hip_flexion_angle_bottom', 'L Hip Flex',   'darkblue'),
    ]
    available = [(c, lbl, clr) for c, lbl, clr in plot_targets if c in df_result.columns]
    if not available:
        return

    reps  = df_result['Rep'].values
    x     = np.arange(len(reps))
    n     = len(available)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(8, len(reps) * 1.5), 5), tight_layout=True)
    for j, (col, lbl, clr) in enumerate(available):
        offset = (j - n / 2 + 0.5) * width
        ax.bar(x + offset, df_result[col], width=width, label=lbl, color=clr, alpha=0.78)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Rep {r}" for r in reps])
    ax.set_xlabel('Rep')
    ax.set_ylabel('Angle at Bottom (deg)')
    ax.set_title(f'Joint Angles at Bottom Position - {base_name}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    out_path = os.path.join(graph_dir, f"{base_name}_rep_angles_bottom.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> グラフ: {os.path.basename(out_path)}")