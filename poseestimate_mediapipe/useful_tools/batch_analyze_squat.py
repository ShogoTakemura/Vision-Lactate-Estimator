import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
    1つのCSVファイルを区間切り出しした上で解析し、結果のCSVとグラフ画像を生成・保存する関数
    """
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    
    # 出力ファイルパスの設定
    output_csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
    output_img_path = os.path.join(output_dir, f"{base_name}_plot.png")
    
    # 1. データの読み込みとデフラグ対策
    raw_df = pd.read_csv(file_path, skiprows=2)
    df = raw_df.copy()
    
    # 肩座標（y軸の平均値）の算出
    df['shoulder_y'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0
    
    # 2. BASEFRAMES.csv から該当ファイルの start / end フレームを取得して切り出し
    match_filename = f"{base_name}_results.csv"
    row_bf = baseframes_df[baseframes_df['filename'] == match_filename]
    
    if not row_bf.empty:
        start_val = row_bf['start'].values[0]
        end_val = row_bf['end'].values[0]
        
        # スタートフレームの前20フレームから、endフレームの後20フレームまで切り出し
        df = df[(df['count'] >= start_val - 20) & (df['count'] <= end_val + 20)].copy()
        crop_msg = f"区間切り出し成功 ({start_val-20} ～ {end_val+20})"
    else:
        crop_msg = "BASEFRAMESに未登録のため全体を解析"
    
    if len(df) == 0:
        print(f"  [Skip] {file_name} : 切り出し後のデータが空です。")
        return

    # 3. 動的なピーク（target_frame）の検出
    # 固定値（0.2）ではなく、切り出した運動区間内の最小値と最大値から自動で閾値を設定します
    min_y = df['shoulder_y'].min()
    max_y = df['shoulder_y'].max()
    adaptive_height = min_y + 0.3 * (max_y - min_y)  # 下位30%以上の高さを閾値とする
    
    peaks, _ = find_peaks(df['shoulder_y'], height=adaptive_height, distance=50)
    
    if len(peaks) == 0:
        print(f"  [Skip] {file_name} : {crop_msg} 内でピークが検出されませんでした。")
        return

    # 4. 各ピークから独立して開始点・終了点を探索（ローカルミニマム方式）
    rep_data = []
    for peak in peaks:
        # 【開始フレームの探索】
        idx = peak
        while idx > 0 and df['shoulder_y'].iloc[idx-1] < df['shoulder_y'].iloc[idx]:
            idx -= 1
        start_frame = int(df['count'].iloc[idx])
        
        # 【終了フレームの探索】
        idx = peak
        while idx < len(df)-1 and df['shoulder_y'].iloc[idx+1] < df['shoulder_y'].iloc[idx]:
            idx += 1
        end_frame = int(df['count'].iloc[idx])
        
        target_frame = int(df['count'].iloc[peak])
        rep_data.append((start_frame, target_frame, end_frame))
        
    # 5. 各指標の計算と出力テーブルの構築
    results = []
    
    # ベースラインとターゲット（分母用）
    baseline_frame = rep_data[0][2] 
    baseline = df.loc[df['count'] == baseline_frame, 'shoulder_y'].values[0]
    
    rep1_target_frame = rep_data[0][1]
    rep1_target_val = df.loc[df['count'] == rep1_target_frame, 'shoulder_y'].values[0]
    
    # 万が一のゼロ除算（RuntimeWarning）を安全に回避
    denom = rep1_target_val - baseline
    if abs(denom) < 1e-5:
        denom = 1.0
    
    for i, (s_f, t_f, e_f) in enumerate(rep_data):
        f_c = e_f - s_f
        d_s = f_c / 30.0
        t_v = df.loc[df['count'] == t_f, 'shoulder_y'].values[0]
        r_v = ((t_v - baseline) / denom) * 100.0
        
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
        
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    # 6. グラフ画像の描画と保存（切り出された美しい波形）
    plt.figure(figsize=(14, 6))
    
    # 切り出された肩座標の波形を描画
    plt.plot(df['count'], df['shoulder_y'], color='black', alpha=0.4, linewidth=1.5, label='Shoulder Y (Average)')
    
    # レップごとのカラーパレット
    colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF', '#FF99CC', '#E0E0E0', '#FFCCFF', '#CCFFFF', '#FFFFCC']
    
    # 各レップの区間をハイライト
    for i, row in output_df.iterrows():
        s_f = int(row['start_frame'])
        t_f = int(row['target_frame'])
        e_f = int(row['end_frame'])
        t_v = row['target_value']
        rep_num = int(row['レップ_number'])
        
        color = colors[(rep_num - 1) % len(colors)]
        plt.axvspan(s_f, e_f, color=color, alpha=0.4, label=f'Rep {rep_num}' if rep_num <= 10 else "")
        
        # ピーク位置に赤丸をプロット
        plt.scatter(t_f, t_v, color='red', edgecolors='black', s=40, zorder=5)
        # レップ番号のテキストラベル
        plt.text(t_f, t_v + (max_y - min_y) * 0.03, f"R{rep_num}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title(f"Extracted Repetition Intervals - {base_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Frame Count", fontsize=10)
    plt.ylabel("Shoulder Y Coordinate", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    
    plt.savefig(output_img_path, dpi=150)
    plt.close()
    
    print(f"  [Success] {crop_msg} -> 検出数: {len(peaks)} レップ 保存完了")


if __name__ == '__main__':
    print("==================================================")
    print(" スクワット波形 決定版区間切り出し一括解析スクリプト")
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