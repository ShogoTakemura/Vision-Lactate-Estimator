import os
import re
import pandas as pd
import numpy as np
from configparser import ConfigParser
from pathlib import Path
import matplotlib.pyplot as plt

from poseestimate_mediapipe.process.csv_utils import read_pose_csv


# ============================================================
# ユーティリティ
# ============================================================

def _load_config_csvs(packagepath: str):
    """MODELBASED_WORKSET と SUBJECTS_DATA を読み込む"""
    modelbased_workset_path = os.path.join(packagepath, 'config', 'MODELBASED_WORKSET.csv')
    subjects_data_path      = os.path.join(packagepath, 'config', 'SUBJECTS_DATA.csv')

    df_model    = pd.read_csv(modelbased_workset_path) if os.path.exists(modelbased_workset_path) else pd.DataFrame()
    df_subjects = pd.read_csv(subjects_data_path)      if os.path.exists(subjects_data_path)      else pd.DataFrame()
    return df_model, df_subjects


def _lookup_subject(base_name: str, df_model: pd.DataFrame, df_subjects: pd.DataFrame):
    """
    ファイル名から被験者名・実使用重量・体重を検索する。
    MODELBASED_WORKSET の filename 列（_correct サフィックスあり）と
    base_name（_correct なし）を部分一致で照合。

    Returns
    -------
    subject_name : str | None
    load_kg      : float | None  -- SUBJECTS_DATA の load 列（実使用重量 kg）
    body_mass    : float | None  -- SUBJECTS_DATA の mass 列（体重 kg）
    """
    if df_model.empty or df_subjects.empty:
        return None, None, None

    key = base_name.replace('_correct', '')
    mask = df_model['filename'].str.replace('_correct', '', regex=False).str.contains(
        re.escape(key), na=False
    )
    match_model = df_model[mask]
    if match_model.empty:
        return None, None, None

    s_id = match_model.iloc[0]['subject_id']
    match_subject = df_subjects[df_subjects['segment_id'] == s_id]
    if match_subject.empty:
        return None, None, None

    row = match_subject.iloc[0]
    return str(row['name']), float(row['load']), float(row['mass'])


# ============================================================
# 新機能：_results.csv から直接データベースを構築
# ============================================================

def process_from_peak_results(config: ConfigParser):
    """
    peak_results_dir にある *_results.csv を走査し、
    レップごとの速度・仕事・エネルギーデータベースを作成する。

    ポーズ座標 CSV（modelbased_trimmed）は不要。
    変位は *_modelbased_trimmed.csv の肩座標（左右平均）から算出する。
    ・下降変位：|shoulder_y[target_frame] - shoulder_y[start_frame]|
    ・挙上変位：|shoulder_y[end_frame]   - shoulder_y[target_frame]|
    ※ shoulder_y = (LEFT_SHOULDER_y + RIGHT_SHOULDER_y) / 2

    重量は SUBJECTS_DATA.load をそのまま使用（実使用重量）。

    運動エネルギーは挙上フェーズ（target_frame → end_frame）のみで算出し、
    セットの総エネルギーとしてレップ分を合算する。

    出力
    ----
    out/work_calculated/REP_DATABASE.csv        : 全レップ詳細
    out/work_calculated/SET_SUMMARY_DATABASE.csv: セット集計
    """
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    peak_results_dir  = config.get('paths', 'peak_results_dir',
                                   fallback=r"C:\Users\ironm\Desktop\peak\results")
    trimmed_dir       = os.path.join(packagepath, 'out', 'modelbased_trimmed')
    output_dir        = os.path.join(packagepath, 'out', 'work_calculated')
    os.makedirs(output_dir, exist_ok=True)

    G   = 9.80665  # 重力加速度 [m/s²]
    FPS = 30.0

    df_model, df_subjects = _load_config_csvs(packagepath)

    result_files = sorted(Path(peak_results_dir).glob('*_results.csv'))
    if not result_files:
        print(f"⚠️  _results.csv が見つかりません: {peak_results_dir}")
        return

    all_reps:      list[dict] = []
    set_summaries: list[dict] = []

    for csv_path in result_files:
        print(f"Processing: {csv_path.name}")

        base_name = csv_path.stem.replace('_results', '')

        # ── 被験者情報（load = 実使用重量） ──────────
        subject_name, load_kg, body_mass = _lookup_subject(base_name, df_model, df_subjects)

        if load_kg is None:
            load_kg = 0.0
            print(f"  ⚠️  被験者データが見つかりません: {base_name}")

        # ── modelbased_trimmed CSV を探して読み込む ──
        # MODELBASED_WORKSET で base_name に対応する _correct ファイル名を取得
        trimmed_df = None
        if not df_model.empty:
            key  = base_name.replace('_correct', '')
            mask = df_model['filename'].str.replace('_correct', '', regex=False).str.contains(
                re.escape(key), na=False
            )
            match_model = df_model[mask]
            if not match_model.empty:
                workset_fname  = match_model.iloc[0]['filename']          # e.g. ..._correct
                trimmed_fname  = f"{workset_fname}_modelbased_trimmed.csv"
                trimmed_path   = os.path.join(trimmed_dir, trimmed_fname)
                if os.path.exists(trimmed_path):
                    # 1行目=メタデータ、2行目=列名（header=1 で読む）
                    trimmed_df = pd.read_csv(trimmed_path, header=1)
                    trimmed_df = trimmed_df.apply(pd.to_numeric, errors='coerce')
                    # 左右肩Y座標の平均を事前計算
                    trimmed_df['shoulder_y'] = (
                        trimmed_df['LEFT_SHOULDER_y'] + trimmed_df['RIGHT_SHOULDER_y']
                    ) / 2.0
                else:
                    print(f"  ⚠️  trimmed CSV が見つかりません: {trimmed_path}")

        if trimmed_df is None:
            print(f"  ⚠️  座標データが取得できないためスキップ: {base_name}")
            continue

        # ── レップ CSV 読み込み ───────────────────────
        try:
            df_reps = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ⚠️  読み込みエラー: {e}")
            continue

        required_cols = {'レップ_number', 'start_frame', 'target_frame', 'end_frame', 'duration_sec'}
        missing = required_cols - set(df_reps.columns)
        if missing:
            print(f"  ⚠️  必要な列が不足しています: {missing}")
            continue

        def get_shoulder_y(frame: int) -> float:
            """count 列でフレームを検索し、最近傍の shoulder_y を返す"""
            exact = trimmed_df[trimmed_df['count'] == frame]
            if not exact.empty:
                return float(exact.iloc[0]['shoulder_y'])
            # 完全一致がない場合は最近傍フレームを使用
            idx = (trimmed_df['count'] - frame).abs().idxmin()
            return float(trimmed_df.loc[idx, 'shoulder_y'])

        set_reps_data: list[dict] = []

        for _, row in df_reps.iterrows():
            rep_num      = int(row['レップ_number'])
            start_frame  = int(row['start_frame'])
            target_frame = int(row['target_frame'])   # 最下点フレーム
            end_frame    = int(row['end_frame'])
            duration_sec = float(row['duration_sec'])

            # ── 肩座標の取得 ──────────────────────────
            y_start  = get_shoulder_y(start_frame)   # 開始（立位）
            y_target = get_shoulder_y(target_frame)  # 最下点
            y_end    = get_shoulder_y(end_frame)      # 終了（立位）

            # ── 変位（左右肩Y平均値の変化量を絶対値で取得） ──
            disp_eccentric  = abs(y_target - y_start)   # 下降：開始→最下点
            disp_concentric = abs(y_end    - y_target)  # 挙上：最下点→終了
            disp_total      = disp_eccentric + disp_concentric

            # ── 時間 ──────────────────────────────────
            time_eccentric  = (target_frame - start_frame) / FPS   # 下降
            time_concentric = (end_frame   - target_frame) / FPS   # 挙上

            # ── 挙上速度（変位 ÷ 挙上時間） ──────────
            vel_eccentric  = disp_eccentric  / time_eccentric  if time_eccentric  > 0 else 0.0
            vel_concentric = disp_concentric / time_concentric if time_concentric > 0 else 0.0

            # ── 位置エネルギー（W = m × g × d） ───────
            # 下降・挙上それぞれの変位から算出し，レップ内で合算
            pe_eccentric  = load_kg * G * disp_eccentric
            pe_concentric = load_kg * G * disp_concentric
            pe_total_rep  = pe_eccentric + pe_concentric

            # ── 運動エネルギー（挙上フェーズのみ）────
            # KE = 1/2 × m × v²  （v = 挙上平均速度）
            ke_concentric = 0.5 * load_kg * (vel_concentric ** 2)

            rep_row = {
                'File':                     base_name,
                'Subject':                  subject_name,
                'Load(kg)':                 load_kg,
                'Body_Mass(kg)':            body_mass,
                'Rep':                      rep_num,
                'Start_Frame':              start_frame,
                'Target_Frame':             target_frame,
                'End_Frame':                end_frame,
                'Duration_Total(s)':        round(duration_sec, 4),
                'Time_Eccentric(s)':        round(time_eccentric,  4),
                'Time_Concentric(s)':       round(time_concentric, 4),
                'Shoulder_Y_Start(m)':      round(y_start,  4),
                'Shoulder_Y_Target(m)':     round(y_target, 4),
                'Shoulder_Y_End(m)':        round(y_end,    4),
                'Disp_Eccentric(m)':        round(disp_eccentric,  4),
                'Disp_Concentric(m)':       round(disp_concentric, 4),
                'Disp_Total_Rep(m)':        round(disp_total,      4),
                'Velocity_Eccentric(m/s)':  round(vel_eccentric,   4),
                'Velocity_Concentric(m/s)': round(vel_concentric,  4),
                'PE_Eccentric(J)':          round(pe_eccentric,    2),
                'PE_Concentric(J)':         round(pe_concentric,   2),
                'PE_Total_Rep(J)':          round(pe_total_rep,    2),
                'KE_Concentric(J)':         round(ke_concentric,   2),
            }
            set_reps_data.append(rep_row)
            all_reps.append(rep_row)

        # ── セット集計 ────────────────────────────────
        if set_reps_data:
            df_set = pd.DataFrame(set_reps_data)

            set_total_pe   = df_set['PE_Total_Rep(J)'].sum()        # 位置エネルギー総和
            set_total_ke   = df_set['KE_Concentric(J)'].sum()       # 運動エネルギー総和（挙上）
            avg_vel_con    = df_set['Velocity_Concentric(m/s)'].mean()
            avg_vel_ecc    = df_set['Velocity_Eccentric(m/s)'].mean()
            avg_disp_ecc   = df_set['Disp_Eccentric(m)'].mean()
            avg_disp_con   = df_set['Disp_Concentric(m)'].mean()

            # 速度低下率（%）: 第1レップ → 最終レップ
            vel_first = df_set['Velocity_Concentric(m/s)'].iloc[0]
            vel_last  = df_set['Velocity_Concentric(m/s)'].iloc[-1]
            vel_drop  = (1 - vel_last / vel_first) * 100 if vel_first > 0 else None

            set_summaries.append({
                'File':                         base_name,
                'Subject':                      subject_name,
                'Load(kg)':                     load_kg,
                'Body_Mass(kg)':                body_mass,
                'Total_Reps':                   len(set_reps_data),
                'Avg_Disp_Eccentric(m)':        round(avg_disp_ecc, 4),
                'Avg_Disp_Concentric(m)':       round(avg_disp_con, 4),
                'Avg_Velocity_Eccentric(m/s)':  round(avg_vel_ecc,  4),
                'Avg_Velocity_Concentric(m/s)': round(avg_vel_con,  4),
                'Set_Total_PE(J)':              round(set_total_pe, 2),   # 位置エネルギー総和
                'Set_Total_KE_Concentric(J)':   round(set_total_ke, 2),   # 運動エネルギー総和（挙上）
                'Vel_Drop_First_to_Last(%)':    round(vel_drop, 2) if vel_drop is not None else None,
            })

            print(f"  -> {len(set_reps_data)} レップ計算完了 "
                  f"| 平均挙上速度: {avg_vel_con:.3f} m/s "
                  f"| 位置エネルギー総和: {set_total_pe:.1f} J "
                  f"| 運動エネルギー総和: {set_total_ke:.1f} J")

    # ── 全データ書き出し ──────────────────────────────
    if all_reps:
        out_rep = os.path.join(output_dir, 'REP_DATABASE.csv')
        pd.DataFrame(all_reps).to_csv(out_rep, index=False, encoding='utf-8-sig')
        print(f"\n📄 レップ別データベース: {out_rep}")

    if set_summaries:
        out_set = os.path.join(output_dir, 'SET_SUMMARY_DATABASE.csv')
        pd.DataFrame(set_summaries).to_csv(out_set, index=False, encoding='utf-8-sig')
        print(f"📄 セット集計データベース: {out_set}")

    print("\n✅ process_from_peak_results 完了")


# ============================================================
# 既存処理（ポーズ座標 CSV から計算）
# ============================================================

def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    target_dir       = os.path.join(packagepath, 'out', 'modelbased_trimmed')
    peak_results_dir = config.get('paths', 'peak_results_dir',
                                  fallback=r"C:\Users\ironm\squat_analyze\frame_viewer_tool\reps\processed")

    com_workset_path = os.path.join(packagepath, 'config', 'COM_WORKSET.csv')

    output_dir = os.path.join(packagepath, 'out', 'work_calculated')
    os.makedirs(output_dir, exist_ok=True)

    peak_graph_dir = os.path.join(packagepath, 'out', 'graphs', 'peak')
    os.makedirs(peak_graph_dir, exist_ok=True)

    csv_files = list(Path(target_dir).glob('*.csv'))
    if not csv_files:
        print(f"⚠️ 対象の座標CSVファイルが見つかりません: {target_dir}")
        return

    df_com = pd.read_csv(com_workset_path) if os.path.exists(com_workset_path) else pd.DataFrame()
    df_model, df_subjects = _load_config_csvs(packagepath)

    G   = 9.80665
    FPS = 30.0

    total_work_database_path = os.path.join(output_dir, 'TOTAL_WORK_DATABASE.csv')
    if os.path.exists(total_work_database_path):
        os.remove(total_work_database_path)

    for csv_path in csv_files:
        print(f"Processing: {csv_path.name}")

        base_name      = csv_path.name.replace('_modelbased_trimmed.csv', '').replace('_modelbase_trimmed.csv', '')
        search_filename = base_name
        subject_name   = base_name.split('_')[0]
        weight_kg      = None

        if not df_com.empty:
            match = df_com[df_com['filename'].str.contains(search_filename, na=False)]
            if not match.empty:
                weight_kg = match.iloc[0]['load']

        if weight_kg is None and not df_model.empty and not df_subjects.empty:
            match_model = df_model[df_model['filename'].str.contains(search_filename, na=False)]
            if not match_model.empty:
                s_id = match_model.iloc[0]['subject_id']
                match_subject = df_subjects[df_subjects['segment_id'] == s_id]
                if not match_subject.empty:
                    weight_kg    = match_subject.iloc[0]['load']
                    subject_name = match_subject.iloc[0]['name']

        if pd.isna(weight_kg) or weight_kg is None:
            weight_kg = 0

        base_name_clean = base_name.replace('_correct', '')
        rep_csv_name    = f"{base_name_clean}_rep.csv"
        rep_csv_path    = os.path.join(peak_results_dir, rep_csv_name)

        if not os.path.exists(rep_csv_path):
            continue

        header_lines, df = read_pose_csv(csv_path)
        df = df.apply(pd.to_numeric, errors='coerce')

        dumbbell_y = ((df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0).values

        if 'count' in df.columns:
            frames = df['count'].values
        else:
            frames = df.index.values

        try:
            df_reps = pd.read_csv(rep_csv_path)
            df_reps = df_reps.dropna(subset=['start_frame', 'end_frame'])
        except Exception:
            continue

        # _rep.csv の列名を process() 内の変数名に統一
        # rep.csv:  rep, start_frame, bottom_frame, end_frame
        # → rep番号列を 'レップ_number' に揃える（既存ロジック流用のため）
        if 'rep' in df_reps.columns and 'レップ_number' not in df_reps.columns:
            df_reps = df_reps.rename(columns={'rep': 'レップ_number'})
        if 'bottom_frame' in df_reps.columns and 'target_frame' not in df_reps.columns:
            df_reps = df_reps.rename(columns={'bottom_frame': 'target_frame'})

        rep_results    = []
        set_total_work = 0.0
        set_total_ke   = 0.0

        plot_starts = []
        plot_tops   = []
        plot_ends   = []

        for i, row in df_reps.iterrows():
            abs_start = int(row['start_frame'])
            abs_end   = int(row['end_frame'])

            if 'count' in df.columns:
                start_matches = df.index[df['count'] >= abs_start]
                end_matches   = df.index[df['count'] <= abs_end]
                if len(start_matches) == 0 or len(end_matches) == 0:
                    continue
                start_idx = start_matches.min()
                end_idx   = end_matches.max()
            else:
                start_idx = abs_start
                end_idx   = abs_end

            start_idx = max(0, start_idx)
            end_idx   = min(len(dumbbell_y) - 1, end_idx)

            if start_idx >= end_idx:
                continue

            # 肩Y座標は下方向が正（立位: 約 -0.09, 最深部: 約 +0.45）
            # → ボトムは Y が最大（argmax）になるフレーム
            segment  = dumbbell_y[start_idx: end_idx + 1]
            top_idx  = start_idx + np.argmax(segment)

            plot_starts.append(start_idx)
            plot_tops.append(top_idx)
            plot_ends.append(end_idx)

            start_y = dumbbell_y[start_idx]
            top_y   = dumbbell_y[top_idx]
            end_y   = dumbbell_y[end_idx]

            h1 = abs(start_y - top_y)
            h2 = abs(end_y   - top_y)
            total_displacement = h1 + h2

            work_down  = weight_kg * G * h1
            work_up    = weight_kg * G * h2
            total_work = work_down + work_up

            time_up        = (end_idx - top_idx) / FPS
            velocity_up    = h2 / time_up if time_up > 0 else 0.0
            kinetic_energy = 0.5 * weight_kg * (velocity_up ** 2)

            set_total_work += total_work
            set_total_ke   += kinetic_energy

            rep_num = int(row['レップ_number']) if 'レップ_number' in row else i + 1

            rep_results.append({
                'Subject':                  subject_name,
                'Weight(kg)':               weight_kg,
                'Rep':                      rep_num,
                'Start_Frame':              frames[start_idx],
                'Top_Frame':                frames[top_idx],
                'End_Frame':                frames[end_idx],
                'Start_Y(m)':               round(start_y, 4),
                'Top_Y(m)':                 round(top_y, 4),
                'End_Y(m)':                 round(end_y, 4),
                'Displacement_Down_h1(m)':  round(h1, 4),
                'Displacement_Up_h2(m)':    round(h2, 4),
                'Total_Displacement(m)':    round(total_displacement, 4),
                'Time_Up(s)':               round(time_up, 4),
                'Velocity_Up(m/s)':         round(velocity_up, 4),
                'Work_Down_U1(J)':          round(work_down, 2),
                'Work_Up_U2(J)':            round(work_up, 2),
                'Total_Work_U(J)':          round(total_work, 2),
                'Kinetic_Energy_Up(J)':     round(kinetic_energy, 2),
            })

        if rep_results:
            # グラフ出力
            plt.figure(figsize=(12, 6))
            plt.plot(frames, -dumbbell_y, label='Shoulder Y Pos', color='gray')
            plt.plot(frames[plot_starts], -dumbbell_y[plot_starts], 'go', markersize=8, label='Start')
            plt.plot(frames[plot_tops],   -dumbbell_y[plot_tops],   'rx', markersize=10, markeredgewidth=2, label='Bottom')
            plt.plot(frames[plot_ends],   -dumbbell_y[plot_ends],   'bo', markersize=8, label='End')
            plt.title(f'Squat Rep Detection: {base_name}', fontsize=16)
            plt.xlabel('Absolute Frame', fontsize=12)
            plt.ylabel('Shoulder Y Position (inverted)', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(peak_graph_dir, f"{base_name}_peaks.png"), dpi=150)
            plt.close()

            # 個別 CSV
            df_rep = pd.DataFrame(rep_results)
            df_rep.to_csv(os.path.join(output_dir, f"{base_name}_squat_work_data.csv"),
                          index=False, encoding='utf-8-sig')

            # 統合 DB への追記
            summary = pd.DataFrame([{
                'File_Name':                 base_name,
                'Subject':                   subject_name,
                'Weight(kg)':                weight_kg,
                'Total_Reps':                len(rep_results),
                'Set_Total_Work(J)':         round(set_total_work, 2),
                'Set_Total_Kinetic_Energy(J)':round(set_total_ke, 2),
            }])
            summary.to_csv(total_work_database_path,
                           mode='a' if os.path.exists(total_work_database_path) else 'w',
                           header=not os.path.exists(total_work_database_path),
                           index=False, encoding='utf-8-sig')

            print(f"  -> {len(rep_results)} レップ計算完了")
        else:
            print(f"  ⚠️ 有効なレップデータがありませんでした。")

    print(f"\n✅ process 完了")