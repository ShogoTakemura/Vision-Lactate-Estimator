import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# =====================================================================
# 【設定エリア】ファイルパスを環境に合わせて指定してください
# =====================================================================
# 1. 解析対象の入力データ（生データCSV）
INPUT_CSV_PATH = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\correctpose\20221102_kikuchi_sayaka-80per-10rep-set1.csv"

# 2. 解析結果の出力先データ（生成されるCSV）
OUTPUT_CSV_PATH = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\rep\20221102_kikuchi_sayaka-80per-10rep-set1_results.csv"
# =====================================================================


def generate_independent_results(raw_csv_path, output_csv_path):
    """
    各レップのピークから独立して前後の極小値（谷）を探索し、
    指定された通りの不連続な運動区間を抽出するスクリプト
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"入力ファイルが見つかりません: {raw_csv_path}")
        
    print(f"データを読み込み中: {raw_csv_path}")
    
    # 1. データの読み込みとデフラグメンテーション（PerformanceWarning対策）
    raw_df = pd.read_csv(raw_csv_path, skiprows=2)
    df = raw_df.copy()  # メモリ上の配置を最適化し、警告を消去します
    
    # 肩座標（y軸の平均値）の算出
    df['shoulder_y'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0
    
    # 2. 各レップの頂点（target_frame）の検出
    peaks, _ = find_peaks(df['shoulder_y'], height=0.2, distance=50)
    
    # 3. 各ピークから独立して開始点・終了点を探索（ローカルミニマム方式）
    rep_data = []
    for peak in peaks:
        # 【開始フレームの探索】ピークから過去に遡り、値が減少しきる（極小値）までループ
        idx = peak
        while idx > 0 and df['shoulder_y'].iloc[idx-1] < df['shoulder_y'].iloc[idx]:
            idx -= 1
        start_frame = int(df['count'].iloc[idx])
        
        # 【終了フレームの探索】ピークから未来に進み、値が減少しきる（極小値）までループ
        idx = peak
        while idx < len(df)-1 and df['shoulder_y'].iloc[idx+1] < df['shoulder_y'].iloc[idx]:
            idx += 1
        end_frame = int(df['count'].iloc[idx])
        
        target_frame = int(df['count'].iloc[peak])
        rep_data.append((start_frame, target_frame, end_frame))
        
    # 4. 各指標の計算と出力テーブルの構築
    results = []
    
    # relative_value算出のためのベースライン（第1レップの終了フレーム「1363」の肩座標の値）
    baseline_frame = rep_data[0][2] 
    baseline = df.loc[df['count'] == baseline_frame, 'shoulder_y'].values[0]
    
    # 第1レップのターゲット値（分母用）
    rep1_target_frame = rep_data[0][1]
    rep1_target_val = df.loc[df['count'] == rep1_target_frame, 'shoulder_y'].values[0]
    
    for i, (s_f, t_f, e_f) in enumerate(rep_data):
        f_c = e_f - s_f
        d_s = f_c / 30.0  # FPS=30として秒数を計算
        t_v = df.loc[df['count'] == t_f, 'shoulder_y'].values[0]
        
        # 相対値の計算公式
        r_v = ((t_v - baseline) / (rep1_target_val - baseline)) * 100.0
        
        results.append({
            'レップ_number': i + 1,
            'start_frame': s_f,
            'target_frame': t_f,
            'end_frame': e_f,
            'frame_count': f_c,
            'duration_sec': d_s,
            'target_value': t_v,
            'relative_value': r_v
        })
        
    # 指定されたパスにCSVとして出力
    output_df = pd.DataFrame(results)
    
    # 出力先フォルダが存在しない場合は作成
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_df.to_csv(output_csv_path, index=False)
    
    print("\n--- 処理が完了しました ---")
    print(f"出力ファイル: {output_csv_path}")
    print(output_df.to_string(index=False))
    return output_df


if __name__ == '__main__':
    try:
        generate_independent_results(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    except Exception as e:
        print(f"エラーが発生しました: {e}")