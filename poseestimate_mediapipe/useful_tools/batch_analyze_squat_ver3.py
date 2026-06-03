import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================================
# 【設定エリア】各種フォルダおよびBASEFRAMES.csvのパスを指定してください
# =====================================================================
INPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\posture_analyzed'
OUTPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\rep'
BASEFRAMES_CSV_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\BASEFRAMES.csv'
# =====================================================================


def process_file(file_path, output_dir, baseframes_df):
    """
    肩の移動速度（物理特徴）と多関節角度をブレンドして、
    個人差やノイズに影響されない高精度なレップ切り出しを行う関数
    """
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    core_name = base_name.replace('_correct_modelbased_trimmed_analyzed', '').replace('_trimmed', '')
    
    output_csv_path = os.path.join(output_dir, f"{core_name}_results.csv")
    output_img_path = os.path.join(output_dir, f"{core_name}_plot.png")
    
    # 1. データの読み込み
    raw_df = pd.read_csv(file_path, skiprows=2)
    df = raw_df.copy()
    
    # 2. データの平滑化（ノイズ除去）とマルチ関節・物理特徴量の算出
    # カタつきを抑えるための移動平均（ウインドウサイズ7）
    window_size = 7
    df['shoulder_y_raw'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0
    df['shoulder_y'] = df['shoulder_y_raw'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    # 【最重要】肩の移動速度（1階微分）と加速度（2階微分）を計算
    # ※MediaPipeは下方向がプラスなので、しゃがむ＝速度プラス、立ち上がる＝速度マイナス
    df['velocity'] = np.gradient(df['shoulder_y'])
    df['acceleration'] = np.gradient(df['velocity'])
    
    # マルチ関節（膝と股関節）の角度もノイズを除去して確保
    df['knee_angle'] = ((df['l_knee_angle'] + df['r_knee_angle']) / 2.0).rolling(window=window_size, center=True, min_periods=1).mean()
    df['hip_angle'] = ((df['l_crotch_angle'] + df['r_crotch_angle']) / 2.0).rolling(window=window_size, center=True, min_periods=1).mean()

    # 3. BASEFRAMES.csv による区間切り出し
    match_filename = f"{core_name}_results.csv"
    row_bf = baseframes_df[baseframes_df['filename'] == match_filename]
    if not row_bf.empty:
        start_val, end_val = row_bf['start'].values[0], row_bf['end'].values[0]
        df = df[(df['count'] >= start_val - 20) & (df['count'] <= end_val + 20)].copy()
        crop_msg = f"区間切り出し成功 ({start_val-20} ～ {end_val+20})"
    else:
        crop_msg = "BASEFRAMESに未登録のため全体を解析"
    
    if len(df) == 0:
        return

    # 4. ボトム（最下点）の検出
    # 最も波形がはっきり出る「膝の屈曲（または肩の最大沈み込み）」で中央の山を捉える
    min_knee, max_knee = df['knee_angle'].min(), df['knee_angle'].max()
    adaptive_height = min_knee + 0.4 * (max_knee - min_knee)
    peaks, _ = find_peaks(df['knee_angle'], height=max(40.0, adaptive_height), distance=50)
    
    if len(peaks) == 0:
        print(f"  [Skip] {file_name} : ピーク未検出")
        return

    # 5. 【新規ロジック】速度プロファイルを用いた「開始・終了」の探索
    rep_data = []
    for peak in peaks:
        # --- 【開始フレーム（スタート）の探索】 ---
        # ピークから過去へ遡り、まず「最大の下降速度（velocityの最大値）」を見つけ、
        # さらに過去へ遡って、その速度が「ほぼ0（静止）」になった瞬間をスタートとする。
        idx = peak
        max_v_down = 0.0
        # 下降中の最大速度を探索
        while idx > 0:
            if df['velocity'].iloc[idx] > max_v_down:
                max_v_down = df['velocity'].iloc[idx]
            # 過去に遡りすぎて、前のレップの上昇フェーズ（速度マイナス）に入ったらストップ
            if df['velocity'].iloc[idx] < -0.005 and idx < peak - 15:
                break
            idx -= 1
        
        # 最大下降速度の10%を「動き出し」の閾値とする（体格や速度の個人差を自動吸収）
        start_threshold = max_v_down * 0.10
        
        idx = peak
        while idx > 0:
            if df['velocity'].iloc[idx] < start_threshold:
                break
            idx -= 1
        start_frame = int(df['count'].iloc[idx])
        
        # --- 【終了フレーム（エンド）の探索】 ---
        # ピークから未来へ進み、まず「最大の Witten（上昇）速度（velocityの負の最大値）」を見つけ、
        # さらに進んで、速度が0付近（立ち上がり完了）に戻った瞬間をエンドとする。
        idx = peak
        max_v_up = 0.0
        while idx < len(df) - 1:
            if df['velocity'].iloc[idx] < max_v_up:
                max_v_up = df['velocity'].iloc[idx]
            if df['velocity'].iloc[idx] > 0.005 and idx > peak + 15:
                break
            idx += 1
            
        end_threshold = max_v_up * 0.10  # 負の閾値
        
        idx = peak
        while idx < len(df) - 1:
            if df['velocity'].iloc[idx] > end_threshold:  # 負の値から0に向かって越えた瞬間
                break
            idx += 1
        end_frame = int(df['count'].iloc[idx])
        
        target_frame = int(df['count'].iloc[peak])
        
        # 【多関節バリデーション】
        # 膝だけでなく股関節（hip_angle）も一定以上（例:15度以上）連動して曲がっている場合のみ正とみなす
        # これにより、手だけを動かしたり、お辞儀だけの誤検知を除外
        if (df.loc[df['count'] == target_frame, 'hip_angle'].values[0] - df['hip_angle'].min()) > 15.0:
            rep_data.append((start_frame, target_frame, end_frame))
        
    # 6. 指標計算とCSV出力
    results = []
    if not rep_data:
        return
        
    baseline_frame = rep_data[0][2] 
    baseline_knee = df.loc[df['count'] == baseline_frame, 'knee_angle'].values[0]
    rep1_target_val = df.loc[df['count'] == rep_data[0][1], 'knee_angle'].values[0]
    denom = rep1_target_val - baseline_knee
    if abs(denom) < 1e-5: denom = 1.0
    
    for i, (s_f, t_f, e_f) in enumerate(rep_data):
        f_c = e_f - s_f
        d_s = f_c / 30.0
        t_v = df.loc[df['count'] == t_f, 'knee_angle'].values[0]
        r_v = ((t_v - baseline_knee) / denom) * 100.0
        
        results.append({
            'レップ_number': i + 1, 'start_frame': s_f, 'target_frame': t_f, 'end_frame': e_f,
            'frame_count': f_c, 'duration_sec': d_s, 'target_value': t_v, 'relative_value': r_v
        })
        
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    # 7. 2軸プロットグラフの生成（上が角度、下が速度）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # 上段：膝と股関節の角度推移
    ax1.plot(df['count'], df['knee_angle'], color='blue', linewidth=2, label='Knee Angle')
    ax1.plot(df['count'], df['hip_angle'], color='green', linewidth=1.5, linestyle=':', label='Hip Angle')
    ax1.set_ylabel("Angle (Degrees)")
    ax1.grid(True, linestyle='--')
    
    # 下段：肩の移動速度（これが判定の心臓部）
    ax2.plot(df['count'], df['velocity'], color='black', linewidth=1.5, label='Shoulder Velocity (dy/dt)')
    ax2.axhline(0, color='red', linestyle='--', alpha=0.5)  # 速度0の基準線
    ax2.set_ylabel("Velocity (pixel/frame)")
    ax2.set_xlabel("Frame Count")
    ax2.grid(True, linestyle='--')
    
    # レップエリアのハイライト
    colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#CC99FF']
    for i, row in output_df.iterrows():
        s_f, t_f, e_f = int(row['start_frame']), int(row['target_frame']), int(row['end_frame'])
        c = colors[i % len(colors)]
        ax1.axvspan(s_f, e_f, color=c, alpha=0.3)
        ax2.axvspan(s_f, e_f, color=c, alpha=0.3)
        ax1.scatter(t_f, row['target_value'], color='red', zorder=5)
        ax1.text(t_f, row['target_value']+5, f"R{int(row['レップ_number'])}", ha='center', fontweight='bold')

    ax1.set_title(f"Multi-Modal Repetition Detection - {core_name}", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=150); plt.close()
    print(f"  [Success] -> 検出数: {len(output_df)} レップ")


if __name__ == '__main__':
    print("==================================================")
    print(" 物理特徴（速度プロファイル）＆多関節ハイブリッド解析")
    print("==================================================")
    if not os.path.exists(BASEFRAMES_CSV_PATH): exit(1)
    baseframes_df = pd.read_csv(BASEFRAMES_CSV_PATH)
    if not os.path.exists(OUTPUT_FOLDER_PATH): os.makedirs(OUTPUT_FOLDER_PATH)
    csv_files = glob.glob(os.path.join(INPUT_FOLDER_PATH, "*_analyzed.csv"))
    
    for idx, file_path in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] 処理中: {os.path.basename(file_path)}")
        try: process_file(file_path, OUTPUT_FOLDER_PATH, baseframes_df)
        except Exception as e: print(f"  [Error] {e}")