import os
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 大量処理時にウィンドウがポップアップして処理が止まるのを防ぐ
import matplotlib
matplotlib.use('Agg')

# =====================================================================
# 【設定エリア】各種フォルダおよびBASEFRAMES.csvのパスを指定してください
# =====================================================================
# 1. 解析対象のCSVファイル群が格納されているフォルダ
INPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\correctpose'

# 2. 解析結果（CSVと画像）を保存するフォルダ
OUTPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\rep'

# 3. 管理用ベースフレームCSVのパス
BASEFRAMES_CSV_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\BASEFRAMES.csv'
# =====================================================================


def process_file(file_path, output_dir, baseframes_df):
    """
    1つのCSVファイルを区間切り出しした上で、腰（HIP）座標を用いて解析し、
    結果のCSVとグラフ画像を生成・保存する関数
    """
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    
    # 出力ファイルパスの設定（指定の命名規則を維持）
    output_csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    output_img_path = os.path.join(output_dir, f"{base_name}_plot.png")
    
    # 1. データの読み込み
    raw_df = pd.read_csv(file_path, skiprows=2)
    df = raw_df.copy()
    
    # 左右の腰（HIP）座標の平均値を算出（バウンドノイズ対策の要）
    if 'LEFT_HIP_y' not in df.columns or 'RIGHT_HIP_y' not in df.columns:
        print(f"  [Skip] {file_name} : HIP座標（y軸）のカラムが見つかりません。")
        return
    df['hip_y_mean'] = (df['LEFT_HIP_y'] + df['RIGHT_HIP_y']) / 2.0
    
    # 2. BASEFRAMES.csv から該当ファイルの start / end フレームを取得して切り出し
    match_filename = f"{base_name}_results.csv"
    row_bf = baseframes_df[baseframes_df['filename'] == match_filename]
    
    if not row_bf.empty:
        start_val = row_bf['start'].values[0]
        end_val = row_bf['end'].values[0]
        
        # スタートフレームの前20フレームから、endフレームの後20フレームまで切り出し
        df = df[(df['count'] >= start_val - 40) & (df['count'] <= end_val + 40)].copy()
        crop_msg = f"区間切り出し成功 ({start_val-40} ～ {end_val+40})"
    else:
        crop_msg = "BASEFRAMESに未登録のため全体を解析"
    
    if len(df) == 0:
        print(f"  [Skip] {file_name} : 切り出し後のデータが空です。")
        return

    # インデックスをリセットしてローカル窓での探索を安全にする
    df = df.reset_index(drop=True)

    # 3. 動的なピーク（bottom_frame）の検出
    min_y = df['hip_y_mean'].min()
    max_y = df['hip_y_mean'].max()
    adaptive_height = min_y + 0.5 * (max_y - min_y)  # しゃがみ深さ50%以上を閾値とする
    
    # ピーク（ボトム位置）の検出
    peaks, _ = find_peaks(df['hip_y_mean'], height=adaptive_height, distance=50)
    
    if len(peaks) == 0:
        print(f"  [Skip] {file_name} : {crop_msg} 内でピークが検出されませんでした。")
        return

    # 4. 各ピークから「バウンドを排除した」開始点・終了点を探索
    rep_data = []
    for i, peak_idx in enumerate(peaks):
        # 【S: 開始フレームの探索】
        start_search = max(0, peak_idx - 60)
        if i > 0:
            start_search = max(start_search, peaks[i-1] + 1)
            
        back_window = df[(df.index >= start_search) & (df.index <= peak_idx)]
        min_idx = back_window['hip_y_mean'].idxmin()
        min_frame_val = df.loc[min_idx, 'hip_y_mean']
        max_frame_val = df.loc[peak_idx, 'hip_y_mean']
        
        # 動き出し（5%閾値）をS点とする
        thresh_s = min_frame_val + 0.05 * (max_frame_val - min_frame_val)
        s_window = back_window[back_window.index >= min_idx]
        s_frame_idx = s_window[s_window['hip_y_mean'] >= thresh_s].index.min()
        
        s_frame = int(df.loc[s_frame_idx, 'count']) if pd.notna(s_frame_idx) else int(df.loc[min_idx, 'count'])
        b_frame = int(df.loc[peak_idx, 'count'])
        
        # 【e: 終了フレームの探索】
        end_search = min(len(df) - 1, peak_idx + 60)
        if i < len(peaks) - 1:
            end_search = min(end_search, peaks[i+1] - 1)
            
        forward_window = df[(df.index >= peak_idx) & (df.index <= end_search)]
        # 立ち上がりきって最も腰が高くなった（Yが最小）位置をe点とする（バウンドを自動回避）
        e_idx = forward_window['hip_y_mean'].idxmin()
        e_frame = int(df.loc[e_idx, 'count'])
        
        rep_data.append((s_frame, b_frame, e_frame, peak_idx))
        
    # 5. 各指標の計算と出力テーブルの構築
    results = []
    
    # ベースライン（最初のレップの終了時点の腰の高さ）
    first_e_frame = rep_data[0][2]
    baseline = df.loc[df['count'] == first_e_frame, 'hip_y_mean'].values[0]
    
    first_b_idx = rep_data[0][3]
    rep1_b_val = df.loc[first_b_idx, 'hip_y_mean']
    
    # ゼロ除算を安全に回避
    denom = rep1_b_val - baseline
    if abs(denom) < 1e-5:
        denom = 1.0
    
    for i, (s_f, b_f, e_f, p_idx) in enumerate(rep_data):
        f_c = e_f - s_f
        d_s = f_c / 30.0  # 30 FPS想定
        b_v = df.loc[p_idx, 'hip_y_mean']
        r_v = ((b_v - baseline) / denom) * 100.0
        
        # ご指定の4カラム（rep, start_frame, bottom_frame, end_frame）を網羅しつつ、既存指標も内包
        results.append({
            'rep': i + 1,
            'start_frame': s_f,
            'bottom_frame': b_f,
            'end_frame': e_f,
            'frame_count': f_c,
            'duration_sec': d_s,
            'bottom_value': b_v,
            'relative_value': r_v
        })
        
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    # 6. グラフ画像の描画と保存（切り出された美しい腰の波形）
    plt.figure(figsize=(14, 6))
    
    # 腰座標の波形を描画
    plt.plot(df['count'], df['hip_y_mean'], color='black', alpha=0.4, linewidth=1.5, label='Hip Y (Average)')
    
    # レップごとのカラーパレット
    colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF', '#FF99CC', '#E0E0E0', '#FFCCFF', '#CCFFFF', '#FFFFCC']
    
    # 各レップの区間をハイライト
    for i, row in output_df.iterrows():
        s_f = int(row['start_frame'])
        b_f = int(row['bottom_frame'])
        e_f = int(row['end_frame'])
        b_v = row['bottom_value']
        rep_num = int(row['rep'])
        
        color = colors[(rep_num - 1) % len(colors)]
        plt.axvspan(s_f, e_f, color=color, alpha=0.4, label=f'Rep {rep_num}' if rep_num <= 10 else "")
        
        # ボトム位置に赤丸をプロット
        plt.scatter(b_f, b_v, color='red', edgecolors='black', s=40, zorder=5)
        # レップ番号のテキストラベル
        plt.text(b_f, b_v + (max_y - min_y) * 0.03, f"R{rep_num}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title(f"Extracted Repetition Intervals (HIP Base) - {base_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Frame Count", fontsize=10)
    plt.ylabel("Hip Y Coordinate (Mean)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_img_path, dpi=150)
    plt.close()
    
    print(f"  [Success] {crop_msg} -> 検出数: {len(peaks)} レップ 保存完了")


if __name__ == '__main__':
    print("==================================================")
    print(" スクワット波形 決定版区間切り出し一括解析スクリプト (HIP座標版)")
    print("==================================================")
    
    # 管理用CSVの読み込み
    if not os.path.exists(BASEFRAMES_CSV_PATH):
        print(f"[Error] BASEFRAMES.csv が見つかりません: {BASEFRAMES_CSV_PATH}")
        exit(1)
    baseframes_df = pd.read_csv(BASEFRAMES_CSV_PATH)
    
    # 出力フォルダがなければ作成
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)
        print(f"出力フォルダを作成しました: {OUTPUT_FOLDER_PATH}")
        
    # 入力フォルダ内のCSVファイルを取得
    csv_files = glob.glob(os.path.join(INPUT_FOLDER_PATH, "*.csv"))
    csv_files = [f for f in csv_files if not f.endswith("_results.csv")]
    
    total_files = len(csv_files)
    print(f"解析対象ファイル数: {total_files} 件\n")
    
    for idx, file_path in enumerate(csv_files, 1):
        print(f"[{idx}/{total_files}] 処理中: {os.path.basename(file_path)}")
        try:
            process_file(file_path, OUTPUT_FOLDER_PATH, baseframes_df)
        except Exception as e:
            print(f"  [Error] エラーが発生しました: {e}")
            
    print("\nすべてのファイルの切り出しおよび解析処理が完了しました。")