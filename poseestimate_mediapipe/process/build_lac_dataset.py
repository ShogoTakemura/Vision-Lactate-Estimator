"""
build_lac_dataset.py
--------------------
乳酸推定用統合データセットを生成するスクリプト。

【入力ファイル】
  1. input_database_dataset.csv   : 既存データセット（lac/after_lac/身体特性等）
  2. REP_DATABASE.csv             : rep単位の速度・仕事データ
  3. REP_POSTURE_DATABASE.csv     : rep単位の姿勢角度データ
  4. processed/*_rep.csv          : rep区間データ（休息時間・rep所要時間）

【出力列】
  --- 識別・目標 ---
  File_Name / lac / after_lac
  --- 仕事・エネルギー ---
  Set_Total_Work(J) / Set_Total_KE(J) / Work_per_rep(J)
  --- 速度 ---
  Avg_Velocity_Concentric(m/s) / Vel_Drop_pct
  --- セット構成 ---
  Total_Reps / set / Weight(kg) / Relative_Load
  Avg_Rep_Duration(s) / Avg_Inter_Rep_Rest(s)
  --- 個人特性 ---
  mass / height / gender / age
  --- 姿勢角度（各角度×4統計量×2集計 = 150列）---
  {angle}_{stat}_set_mean   : セット内rep平均
  {angle}_{stat}_drop_rate  : 1rep→最終rep変化率(%)
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd

from squat_core.paths import PKG_ROOT, FRAME_VIEWER_ROOT

# ============================================================
# パスのデフォルト値（PROJECT_ROOT から自動計算。手動編集不要）
# ============================================================
_DEFAULT_BASE_DATASET = str(PKG_ROOT / "out" / "estimate_blc_data" / "input_database_dataset.csv")
_DEFAULT_REP_DATABASE = str(PKG_ROOT / "out" / "work_calculated" / "REP_DATABASE.csv")
_DEFAULT_REP_POSTURE  = str(PKG_ROOT / "out" / "rep_posture" / "REP_POSTURE_DATABASE.csv")
_DEFAULT_REP_DIR      = str(FRAME_VIEWER_ROOT / "reps" / "processed")
_DEFAULT_OUTPUT       = str(PKG_ROOT / "out" / "work_calculated" / "lac_dataset_full.csv")

# モジュールとして import された場合のデフォルト（後方互換）
BASE_DATASET_PATH = _DEFAULT_BASE_DATASET
REP_DATABASE_PATH = _DEFAULT_REP_DATABASE
REP_POSTURE_PATH  = _DEFAULT_REP_POSTURE
PROCESSED_REP_DIR = _DEFAULT_REP_DIR
OUTPUT_PATH       = _DEFAULT_OUTPUT

FPS = 30.0

# 姿勢角度の基底列名
ANGLE_BASE_COLS = [
    'trunk_lean_angle', 'r_knee_angle', 'l_knee_angle',
    'r_hip_flexion_angle', 'l_hip_flexion_angle',
    'r_crotch_angle', 'l_crotch_angle',
    'r_ankle_lean_angle', 'l_ankle_lean_angle',
    'trunk_thigh_angle', 'shoulder_tilt', 'hip_tilt',
    'lateral_trunk_tilt', 'r_knee_in', 'l_knee_in',
    'r_knee_forward', 'l_knee_forward', 'relative_com_depth',
]
STAT_SUFFIXES = ['_mean', '_bottom', '_descent_mean', '_ascent_mean']
TIME_COLS     = ['Rep_Duration_s', 'Descent_Time_s', 'Ascent_Time_s']


# ============================================================
# Step 1: 既存データセット
# ============================================================
def load_base(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['match_key'] = df['File_Name'].str.replace('_correct', '', regex=False)
    print(f"[1] input_database_dataset: {len(df)}行")
    return df


# ============================================================
# Step 2: REP_DATABASE → セット単位集計
#   Avg_Velocity_Concentric / Vel_Drop_pct
# ============================================================
def aggregate_rep_database(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['match_key'] = df['File'].str.replace('_correct', '', regex=False)
    print(f"[2] REP_DATABASE: {len(df)}行 ({df['match_key'].nunique()}セット)")

    rows = []
    for mk, grp in df.groupby('match_key'):
        grp_s = grp.sort_values('Rep')
        vels  = grp_s['Velocity_Concentric(m/s)'].values
        avg_v = float(np.mean(vels))
        valid = vels[vels > 0]
        vel_drop = float((1.0 - valid[-1] / valid[0]) * 100.0) if len(valid) >= 2 else np.nan
        rows.append({
            'match_key':                     mk,
            'Avg_Velocity_Concentric(m/s)': round(avg_v,    4),
            'Vel_Drop_pct':                 round(vel_drop, 2) if not np.isnan(vel_drop) else np.nan,
        })

    result = pd.DataFrame(rows)
    print(f"     → {len(result)}セット集計完了")
    return result


# ============================================================
# Step 3: *_rep.csv → セット単位集計
#   Avg_Rep_Duration / Avg_Inter_Rep_Rest
# ============================================================
def aggregate_rep_csvs(processed_dir):
    rep_files = sorted(glob.glob(os.path.join(processed_dir, '*_rep.csv')))
    if not rep_files:
        print(f"[3] WARNING: *_rep.csv が見つかりません: {processed_dir}")
        return pd.DataFrame()
    print(f"[3] *_rep.csv: {len(rep_files)}ファイル")

    rows = []
    for fpath in rep_files:
        mk = os.path.basename(fpath).replace('_rep.csv', '')
        try:
            df = pd.read_csv(fpath).sort_values('rep').reset_index(drop=True)
            durations = (df['end_frame'] - df['start_frame']) / FPS
            avg_dur   = float(durations.mean())
            if len(df) >= 2:
                rests    = (df['start_frame'].iloc[1:].values - df['end_frame'].iloc[:-1].values) / FPS
                rests    = rests[rests >= 0]
                avg_rest = float(np.mean(rests)) if len(rests) > 0 else np.nan
            else:
                avg_rest = np.nan
            rows.append({
                'match_key':            mk,
                'Avg_Rep_Duration(s)':  round(avg_dur,  4),
                'Avg_Inter_Rep_Rest(s)':round(avg_rest, 4) if not np.isnan(avg_rest) else np.nan,
            })
        except Exception as e:
            print(f"     WARNING: {os.path.basename(fpath)}: {e}")

    result = pd.DataFrame(rows)
    print(f"     → {len(result)}セット集計完了")
    return result


# ============================================================
# Step 4: REP_POSTURE_DATABASE → セット単位集計
#   各角度×統計量 の set_mean / drop_rate
# ============================================================
def aggregate_posture_database(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df['match_key'] = df['File'].str.replace('_correct', '', regex=False)
    print(f"[4] REP_POSTURE_DATABASE: {len(df)}行 ({df['match_key'].nunique()}セット)")

    # 集計対象列を決定（存在するもののみ）
    target_cols = []
    for base in ANGLE_BASE_COLS:
        for suf in STAT_SUFFIXES:
            col = f"{base}{suf}"
            if col in df.columns:
                target_cols.append(col)
    for c in TIME_COLS:
        if c in df.columns:
            target_cols.append(c)

    rows = []
    for mk, grp in df.groupby('match_key'):
        grp_s = grp.sort_values('Rep')
        rec   = {'match_key': mk}

        for col in target_cols:
            vals = grp_s[col].dropna().values
            if len(vals) == 0:
                rec[f"{col}_set_mean"]  = np.nan
                rec[f"{col}_drop_rate"] = np.nan
                continue
            rec[f"{col}_set_mean"] = round(float(np.mean(vals)), 4)
            if vals[0] != 0 and not np.isnan(vals[0]):
                rec[f"{col}_drop_rate"] = round(float((vals[-1] - vals[0]) / abs(vals[0]) * 100.0), 4)
            else:
                rec[f"{col}_drop_rate"] = np.nan

        rows.append(rec)

    result = pd.DataFrame(rows)
    posture_cols = [c for c in result.columns if c != 'match_key']
    print(f"     → {len(result)}セット集計完了 ({len(posture_cols)}列追加)")
    return result


# ============================================================
# Step 5: 派生特徴量
# ============================================================
def add_derived(df):
    df = df.copy()
    if 'Set_Total_Kinetic_Energy(J)' in df.columns:
        df = df.rename(columns={'Set_Total_Kinetic_Energy(J)': 'Set_Total_KE(J)'})
    df['Work_per_rep(J)'] = (df['Set_Total_Work(J)'] / df['Total_Reps']).round(2)
    df['Relative_Load']   = (df['Weight(kg)']        / df['mass']).round(4)
    return df


# ============================================================
# メイン
# ============================================================
def run(
    base_dataset: str = _DEFAULT_BASE_DATASET,
    rep_database: str = _DEFAULT_REP_DATABASE,
    rep_posture:  str = _DEFAULT_REP_POSTURE,
    rep_dir:      str = _DEFAULT_REP_DIR,
    output:       str = _DEFAULT_OUTPUT,
) -> None:
    print("=" * 60)
    print("乳酸推定用統合データセット生成")
    print("=" * 60)

    df = load_base(base_dataset)
    df = df.merge(aggregate_rep_database(rep_database), on='match_key', how='left')
    df = df.merge(aggregate_rep_csvs(rep_dir),          on='match_key', how='left')
    df = df.merge(aggregate_posture_database(rep_posture), on='match_key', how='left')
    df = add_derived(df)

    base_cols = [
        'File_Name', 'lac', 'after_lac',
        'Set_Total_Work(J)', 'Set_Total_KE(J)', 'Work_per_rep(J)',
        'Avg_Velocity_Concentric(m/s)', 'Vel_Drop_pct',
        'Total_Reps', 'set', 'Weight(kg)', 'Relative_Load',
        'Avg_Rep_Duration(s)', 'Avg_Inter_Rep_Rest(s)',
        'mass', 'height', 'gender', 'age',
    ]
    posture_cols = sorted([c for c in df.columns
                           if c.endswith('_set_mean') or c.endswith('_drop_rate')])
    out_cols = base_cols + posture_cols

    for col in out_cols:
        if col not in df.columns:
            df[col] = np.nan
            print(f"  WARNING: 列なし・NaN補完: {col}")

    df_out = df[out_cols].reset_index(drop=True)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df_out.to_csv(output, index=False, encoding='utf-8-sig')

    print()
    print("=" * 60)
    print(f"完了: {len(df_out)}行 x {len(df_out.columns)}列")
    print(f"   基本特徴量: {len(base_cols)}列")
    print(f"   姿勢角度特徴量: {len(posture_cols)}列")
    print(f"   出力先: {output}")
    miss = df_out.isnull().sum()
    miss = miss[miss > 0]
    if len(miss) > 0:
        print("\n--- 欠損値サマリ ---")
        for col, n in miss.items():
            print(f"  {col}: {n}件")
    else:
        print("欠損なし")


def main() -> None:
    parser = argparse.ArgumentParser(description="乳酸推定用統合データセット生成")
    parser.add_argument("--base_dataset", default=_DEFAULT_BASE_DATASET,
                        help="input_database_dataset.csv のパス")
    parser.add_argument("--rep_database", default=_DEFAULT_REP_DATABASE,
                        help="REP_DATABASE.csv のパス")
    parser.add_argument("--rep_posture",  default=_DEFAULT_REP_POSTURE,
                        help="REP_POSTURE_DATABASE.csv のパス")
    parser.add_argument("--rep_dir",      default=_DEFAULT_REP_DIR,
                        help="*_rep.csv が入ったディレクトリ")
    parser.add_argument("--output",       default=_DEFAULT_OUTPUT,
                        help="出力 CSV パス")
    args = parser.parse_args()
    run(
        base_dataset=args.base_dataset,
        rep_database=args.rep_database,
        rep_posture=args.rep_posture,
        rep_dir=args.rep_dir,
        output=args.output,
    )


if __name__ == "__main__":
    main()
