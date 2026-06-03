"""
auto_pipeline.py
----------------
BASEFRAMES.csv が生成済みの状態から，乳酸推定用統合データセット(lac_dataset_full.csv)
までの全処理を自動実行するスクリプト。

【実行される処理順序】
  Step 1: MODELBASED_WORKSET.csv の basedframe_id を BASEFRAMES.csv から自動更新
  Step 2: model based correct pose      (modelbasecorrect.process)
  Step 3: calculate com location        (calccomprocess.process)
  Step 4: Calculate Squat Work          (calculate_work_process.process)
  Step 5: build_lac_dataset             (統合データセット生成)

【前提条件】
  - BASEFRAMES.csv が config/ に存在すること
  - MODELBASED_WORKSET.csv / COM_WORKSET.csv に filename/subject_id/load/mass が記入済みであること
    （basedframe_id のみ自動更新。その他は手動入力が必要）
  - correctpickle/ に pickle ファイルが存在すること

【使い方】
  python auto_pipeline.py [--skip-until STEP番号]

  例: Step3 から再開する場合
      python auto_pipeline.py --skip-until 2
"""

import os
import sys
import csv
import glob
import argparse
import pathlib
import configparser
import pandas as pd
from unittest.mock import patch

# ============================================================
# ★ パス設定（環境に合わせて変更してください）
# ============================================================
PACKAGE_DIR      = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe"
PROCESSED_REP_DIR = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\reps\processed"
FPS              = 30.0

# ============================================================
# 設定ファイル・ワークセットのパス（PACKAGE_DIR 配下）
# ============================================================
def _p(*parts):
    return os.path.join(PACKAGE_DIR, *parts)

CONFIG_PATH          = _p('config', 'config.ini')
BASEFRAMES_PATH      = _p('config', 'BASEFRAMES.csv')
MODELBASED_WS_PATH   = _p('config', 'MODELBASED_WORKSET.csv')
COM_WS_PATH          = _p('config', 'COM_WORKSET.csv')
SUBJECTS_DATA_PATH   = _p('config', 'SUBJECTS_DATA.csv')
REP_DATABASE_PATH    = _p('out', 'work_calculated', 'REP_DATABASE.csv')
REP_POSTURE_PATH     = _p('out', 'rep_posture', 'REP_POSTURE_DATABASE.csv')
BASE_DATASET_PATH    = _p('out', 'work_calculated', 'input_database_dataset.csv')
OUTPUT_DATASET_PATH  = _p('out', 'work_calculated', 'lac_dataset_full.csv')


# ============================================================
# ユーティリティ
# ============================================================
def section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def check_file(path: str, label: str) -> bool:
    if not os.path.exists(path):
        print(f"  ❌ 見つかりません [{label}]: {path}")
        return False
    print(f"  ✅ 確認済み [{label}]")
    return True


# ============================================================
# Step 1: MODELBASED_WORKSET.csv の basedframe_id 自動更新
# ============================================================
def step1_update_basedframe_id():
    section("Step 1: MODELBASED_WORKSET basedframe_id 自動更新")

    df_base  = pd.read_csv(BASEFRAMES_PATH,    encoding='utf-8-sig')
    df_model = pd.read_csv(MODELBASED_WS_PATH, encoding='utf-8-sig')

    # BASEFRAMES: filename から _rep.csv を除去してキー生成
    df_base['match_key'] = df_base['filename'].str.replace('_rep.csv', '', regex=False)

    # MODELBASED_WORKSET: filename から _correct を除去してキー生成
    df_model['match_key'] = df_model['filename'].str.replace('_correct', '', regex=False)

    id_map = df_base.set_index('match_key')['baseframe id'].to_dict()
    df_model['basedframe_id'] = df_model['match_key'].map(id_map)

    miss = df_model['basedframe_id'].isna().sum()
    if miss > 0:
        print(f"  ⚠️  basedframe_id が見つからない行: {miss} 件")
        print(df_model[df_model['basedframe_id'].isna()][['filename']].to_string())

    df_model = df_model.drop(columns=['match_key'])
    df_model.to_csv(MODELBASED_WS_PATH, index=False, encoding='utf-8-sig')
    print(f"  → {len(df_model)}件更新完了: {MODELBASED_WS_PATH}")


# ============================================================
# Step 2: modelbasecorrect（モデルベース姿勢補正）
# input() を 'N' で自動応答してワークセットチェックモードで実行
# ============================================================
def step3_modelbasecorrect(config):
    section("Step 2: model based correct pose")
    from poseestimate_mediapipe.process import modelbasecorrect

    # input() を 'N' で自動応答（ワークセットの更新をスキップ→チェックモード）
    with patch('builtins.input', return_value='N'):
        modelbasecorrect.process(config)
    print("  → modelbasecorrect 完了")


# ============================================================
# Step 4: calccomprocess（重心計算）
# input() を 'N' で自動応答
# ============================================================
def step4_calccom(config):
    section("Step 3: calculate com location")
    from poseestimate_mediapipe.process import calccomprocess

    with patch('builtins.input', return_value='N'):
        calccomprocess.process(config)
    print("  → calccom 完了")


# ============================================================
# Step 5: calculate_work_process（仕事量・REP_DATABASE 生成）
# ============================================================
def step5_calculate_work(config):
    section("Step 4: Calculate Squat Work & Database Export")
    from poseestimate_mediapipe.process import calculate_work_process
    calculate_work_process.process(config)
    print("  → calculate_work 完了")


# ============================================================
# Step 5: 統合データセット生成
# ============================================================

# 姿勢角度の基底列名・統計量サフィックス
_ANGLE_BASE_COLS = [
    'trunk_lean_angle', 'r_knee_angle', 'l_knee_angle',
    'r_hip_flexion_angle', 'l_hip_flexion_angle',
    'r_crotch_angle', 'l_crotch_angle',
    'r_ankle_lean_angle', 'l_ankle_lean_angle',
    'trunk_thigh_angle', 'shoulder_tilt', 'hip_tilt',
    'lateral_trunk_tilt', 'r_knee_in', 'l_knee_in',
    'r_knee_forward', 'l_knee_forward', 'relative_com_depth',
]
_STAT_SUFFIXES = ['_mean', '_bottom', '_descent_mean', '_ascent_mean']
_TIME_COLS     = ['Rep_Duration_s', 'Descent_Time_s', 'Ascent_Time_s']


def _agg_rep_database(path: str) -> pd.DataFrame:
    """REP_DATABASE → セット単位（速度・速度低下率）"""
    import numpy as np
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['match_key'] = df['File'].str.replace('_correct', '', regex=False)
    rows = []
    for mk, grp in df.groupby('match_key'):
        vels  = grp.sort_values('Rep')['Velocity_Concentric(m/s)'].values
        avg_v = float(np.mean(vels))
        valid = vels[vels > 0]
        vdrop = float((1.0 - valid[-1] / valid[0]) * 100.0) if len(valid) >= 2 else float('nan')
        rows.append({'match_key': mk,
                     'Avg_Velocity_Concentric(m/s)': round(avg_v, 4),
                     'Vel_Drop_pct': round(vdrop, 2) if not np.isnan(vdrop) else float('nan')})
    return pd.DataFrame(rows)


def _agg_rep_csvs(processed_dir: str) -> pd.DataFrame:
    """*_rep.csv → セット単位（rep所要時間・休息時間）"""
    import numpy as np
    rows = []
    for fpath in sorted(glob.glob(os.path.join(processed_dir, '*_rep.csv'))):
        mk = os.path.basename(fpath).replace('_rep.csv', '')
        try:
            df_r     = pd.read_csv(fpath).sort_values('rep').reset_index(drop=True)
            avg_dur  = float(((df_r['end_frame'] - df_r['start_frame']) / FPS).mean())
            if len(df_r) >= 2:
                rests    = (df_r['start_frame'].iloc[1:].values - df_r['end_frame'].iloc[:-1].values) / FPS
                avg_rest = float(np.mean(rests[rests >= 0])) if len(rests[rests >= 0]) > 0 else float('nan')
            else:
                avg_rest = float('nan')
            rows.append({'match_key': mk,
                         'Avg_Rep_Duration(s)':   round(avg_dur,  4),
                         'Avg_Inter_Rep_Rest(s)': round(avg_rest, 4) if not np.isnan(avg_rest) else float('nan')})
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(fpath)}: {e}")
    return pd.DataFrame(rows)


def _agg_posture_database(path: str) -> pd.DataFrame:
    """REP_POSTURE_DATABASE → セット単位（角度×統計量の平均・変化率）"""
    import numpy as np
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['match_key'] = df['File'].str.replace('_correct', '', regex=False)

    target_cols = [f"{b}{s}" for b in _ANGLE_BASE_COLS for s in _STAT_SUFFIXES if f"{b}{s}" in df.columns]
    target_cols += [c for c in _TIME_COLS if c in df.columns]

    rows = []
    for mk, grp in df.groupby('match_key'):
        rec = {'match_key': mk}
        for col in target_cols:
            vals = grp.sort_values('Rep')[col].dropna().values
            if len(vals) == 0:
                rec[f"{col}_set_mean"]  = float('nan')
                rec[f"{col}_drop_rate"] = float('nan')
            else:
                rec[f"{col}_set_mean"]  = round(float(np.mean(vals)), 4)
                rec[f"{col}_drop_rate"] = round(float((vals[-1] - vals[0]) / abs(vals[0]) * 100.0), 4) \
                                          if vals[0] != 0 and not np.isnan(vals[0]) else float('nan')
        rows.append(rec)
    return pd.DataFrame(rows)


def step6_build_dataset():
    section("Step 5: 統合データセット生成 (lac_dataset_full.csv)")

    # ── 前提ファイル確認 ──────────────────────────────
    ok = True
    ok &= check_file(BASE_DATASET_PATH,  "input_database_dataset.csv")
    ok &= check_file(REP_DATABASE_PATH,  "REP_DATABASE.csv")
    ok &= check_file(REP_POSTURE_PATH,   "REP_POSTURE_DATABASE.csv")
    ok &= check_file(PROCESSED_REP_DIR,  "processed rep dir")
    if not ok:
        print("  ❌ 必要ファイルが不足しています。Step4 の出力を確認してください。")
        return

    # ── 読み込み・結合 ──────────────────────────────
    print("  [1] input_database_dataset 読み込み中...")
    df = pd.read_csv(BASE_DATASET_PATH, encoding='utf-8-sig')
    df['match_key'] = df['File_Name'].str.replace('_correct', '', regex=False)

    print("  [2] REP_DATABASE 集計中...")
    df = df.merge(_agg_rep_database(REP_DATABASE_PATH), on='match_key', how='left')

    print("  [3] *_rep.csv 集計中...")
    df = df.merge(_agg_rep_csvs(PROCESSED_REP_DIR), on='match_key', how='left')

    print("  [4] REP_POSTURE_DATABASE 集計中...")
    df_posture = _agg_posture_database(REP_POSTURE_PATH)
    posture_cols = sorted([c for c in df_posture.columns if c != 'match_key'])
    df = df.merge(df_posture, on='match_key', how='left')

    # ── 派生特徴量 ───────────────────────────────────
    if 'Set_Total_Kinetic_Energy(J)' in df.columns:
        df = df.rename(columns={'Set_Total_Kinetic_Energy(J)': 'Set_Total_KE(J)'})
    df['Work_per_rep(J)'] = (df['Set_Total_Work(J)'] / df['Total_Reps']).round(2)
    df['Relative_Load']   = (df['Weight(kg)']        / df['mass']).round(4)

    # ── 出力列構成 ───────────────────────────────────
    base_cols = [
        'File_Name', 'lac', 'after_lac',
        'Set_Total_Work(J)', 'Set_Total_KE(J)', 'Work_per_rep(J)',
        'Avg_Velocity_Concentric(m/s)', 'Vel_Drop_pct',
        'Total_Reps', 'set', 'Weight(kg)', 'Relative_Load',
        'Avg_Rep_Duration(s)', 'Avg_Inter_Rep_Rest(s)',
        'mass', 'height', 'gender', 'age',
    ]
    out_cols = base_cols + posture_cols
    for col in out_cols:
        if col not in df.columns:
            df[col] = float('nan')
            print(f"  ⚠️  列なし・NaN補完: {col}")

    df_out = df[out_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_DATASET_PATH, index=False, encoding='utf-8-sig')

    print(f"\n  ✅ 完了: {len(df_out)}行 × {len(df_out.columns)}列")
    print(f"     基本特徴量: {len(base_cols)}列")
    print(f"     姿勢角度特徴量: {len(posture_cols)}列")
    print(f"  出力先: {OUTPUT_DATASET_PATH}")
    miss = df_out.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        print("  --- 欠損値サマリ ---")
        for col, n in miss.items():
            print(f"    {col}: {n}件")
    else:
        print("  欠損なし ✅")


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="auto_pipeline: BASEFRAMES → lac_dataset_full")
    parser.add_argument('--skip-until', type=int, default=0,
                        help='このSTEP番号まで処理をスキップして再開 (例: --skip-until 2 → Step3から実行)')
    args = parser.parse_args()
    skip_until = args.skip_until

    # ── 前提ファイル確認 ──────────────────────────────
    section("前提ファイル確認")
    all_ok = True
    all_ok &= check_file(CONFIG_PATH,        "config.ini")
    all_ok &= check_file(BASEFRAMES_PATH,    "BASEFRAMES.csv")
    all_ok &= check_file(MODELBASED_WS_PATH, "MODELBASED_WORKSET.csv")
    all_ok &= check_file(COM_WS_PATH,        "COM_WORKSET.csv")
    all_ok &= check_file(SUBJECTS_DATA_PATH, "SUBJECTS_DATA.csv")
    if not all_ok:
        print("\n前提ファイルが不足しています。確認して再実行してください。")
        sys.exit(1)

    # ── config 読み込み ──────────────────────────────
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH, encoding='utf-8')

    # ── 各ステップ実行 ───────────────────────────────
    steps = [
        (1, "MODELBASED_WORKSET basedframe_id 更新", lambda: step1_update_basedframe_id()),
        (2, "model based correct pose",              lambda: step3_modelbasecorrect(config)),
        (3, "calculate com location",                lambda: step4_calccom(config)),
        (4, "Calculate Squat Work",                  lambda: step5_calculate_work(config)),
        (5, "統合データセット生成",                    lambda: step6_build_dataset()),
    ]

    for step_num, step_name, func in steps:
        if step_num <= skip_until:
            print(f"\n  [SKIP] Step {step_num}: {step_name}")
            continue
        try:
            func()
        except Exception as e:
            print(f"\n  ❌ Step {step_num} でエラーが発生しました: {e}")
            print(f"     --skip-until {step_num} オプションで {step_num+1} 以降から再開できます。")
            import traceback; traceback.print_exc()
            sys.exit(1)

    section("✅ 全処理完了")
    print(f"  出力: {OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    main()
