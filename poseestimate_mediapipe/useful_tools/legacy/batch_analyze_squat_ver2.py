import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# =====================================================================
# 【設定エリア】各種フォルダおよびBASEFRAMES.csvのパスを指定してください
# =====================================================================
# 1. 解析対象のCSVファイル群（角度データが含まれるposture_analyzed）が格納されているフォルダ
INPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\posture_analyzed'

# 2. 解析結果（CSVと画像）を保存するフォルダ
OUTPUT_FOLDER_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\rep'

# 3. 管理用ベースフレームCSVのパス
BASEFRAMES_CSV_PATH = r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\BASEFRAMES.csv'
# =====================================================================


def process_file(file_path, output_dir, baseframes_df):
    """
    1つのCSVファイルを膝角度メインで区間切り出し・解析し、結果のCSVとグラフ画像を生成・保存する関数
    """
    file_name = os.path.basename(file_path)
    base_name, _ = os.path.splitext(file_name)
    
    # ファイル名から共通の「コア名」を抽出（サフィックスの差分を吸収）
    core_name = base_name.replace('_correct_modelbased_trimmed_analyzed', '').replace('_trimmed', '')
    
    # 出力ファイルパスの設定（BASEFRAMES.csvの命名規則に統一）
    output_csv_path = os.path.join(output_dir, f"{core_name}_results.csv")
    output_img_path = os.path.join(output_dir, f"{core_name}_plot.png")
    
    # 1. データの読み込み
    raw_df = pd.read_csv(file_path, skiprows=2)
    df = raw_df.copy()
    
    # 【追加】左右の膝角度の平均値を算出してメインの指標とする（片側のブレノイズを相殺）
    df['knee_angle'] = (df['l_knee_angle'] + df['r_knee_angle']) / 2.0
    # 肩座標（参考用のグラフ描画および互換性のために維持）
    df['shoulder_y'] = (df['LEFT_SHOULDER_y'] + df['RIGHT_SHOULDER_y']) / 2.0
    
    # 2. BASEFRAMES.csv から該当ファイルの start / end フレームを取得して切り出し
    match_filename = f"{core_name}_results.csv"
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

    # 3. 動的なピーク（target_frame）の検出（膝角度ベース）
    min_angle = df['knee_angle'].min()
    max_angle = df['knee_angle'].max()
    
    # 切り出し区間内の最小と最大から、下から40%以上曲がった位置をボトム検出の閾値とする
    adaptive_height = min_angle + 0.4 * (max_angle - min_angle)
    adaptive_height = max(40.0, adaptive_height)  # 浅すぎる誤検知を防ぐため最低40度を担保
    
    peaks, _ = find_peaks(df['knee_angle'], height=adaptive_height, distance=50)
    
    if len(peaks) == 0:
        print(f"  [Skip] {file_name} : {crop_msg} 内でピークが検出されませんでした。")
        return

    # 4. 各ピークから独立して開始点・終了点を探索（閾値超過区間方式）
    rep_data = []
    
    # 直立時（しゃがんでいない状態）の角度を基準に、＋15度、または固定20度を「動きの境界」とする
    # これによりローカルミニマム方式にあった「小さなブレによる早期判定終了バグ」を完全に回避
    angle_threshold = max(20.0, min_angle + 15.0)
    
    for peak in peaks:
        # 【開始フレームの探索】ピークから過去へ遡り、膝が伸びきった（閾値を下回った）瞬間を探す
        idx = peak
        while idx > 0 and df['knee_angle'].iloc[idx] > angle_threshold:
            idx -= 1
        start_frame = int(df['count'].iloc[idx])
        
        # 【終了フレームの探索】ピークから未来へ進み、膝が再び伸びきった瞬間を探す
        idx = peak
        while idx < len(df)-1 and df['knee_angle'].iloc[idx] > angle_threshold:
            idx += 1
        end_frame = int(df['count'].iloc[idx])
        
        target_frame = int(df['count'].iloc[peak])
        rep_data.append((start_frame, target_frame, end_frame))
        
    # 5. 各指標の計算と出力テーブルの構築
    results = []
    
    # ベースライン（膝角度ベース）
    baseline_frame = rep_data[0][2] 
    baseline_angle = df.loc[df['count'] == baseline_frame, 'knee_angle'].values[0]
    
    rep1_target_frame = rep_data[0][1]
    rep1_target_val = df.loc[df['count'] == rep1_target_frame, 'knee_angle'].values[0]
    
    denom = rep1_target_val - baseline_angle
    if abs(denom) < 1e-5:
        denom = 1.0
    
    for i, (s_f, t_f, e_f) in enumerate(rep_data):
        f_c = e_f - s_f
        d_s = f_c / 30.0
        t_v = df.loc[df['count'] == t_f, 'knee_angle'].values[0] # 膝角度の最高値を記録
        r_v = ((t_v - baseline_angle) / denom) * 100.0
        
        results.append({
            'レップ_number': i + 1,
            'start_frame': s_f,
            'target_frame': t_f,
            'end_frame': e_f,
            'frame_count': f_c,
            'duration_sec': d_s,
            'target_value': t_v,       # 膝角度（度）
            'relative_value': r_v      # 1回目を100%とした相対的な屈曲深さ
        })
        
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    
    # 6. グラフ画像の描画と保存（ツインアキシス構造：角度と座標をダブル表示）
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 左軸：メインとなる「膝の角度波形」
    color_angle = '#1f77b4'
    ax1.set_xlabel("Frame Count", fontsize=10)
    ax1.set_ylabel("Knee Angle (Degrees)", color=color_angle, fontsize=10)
    ax1.plot(df['count'], df['knee_angle'], color=color_angle, linewidth=2, label='Knee Angle (Average)')
    ax1.tick_params(axis='y', labelcolor=color_angle)
    
    # 右軸：サブとしての「肩座標（従来データ）」を重ねる
    ax2 = ax1.twinx()
    color_shoulder = '#7f7f7f'
    ax2.set_ylabel("Shoulder Y Coordinate", color=color_shoulder, fontsize=10)
    ax2.plot(df['count'], df['shoulder_y'], color=color_shoulder, alpha=0.4, linestyle='--', label='Shoulder Y')
    ax2.tick_params(axis='y', labelcolor=color_shoulder)
    
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
        ax1.axvspan(s_f, e_f, color=color, alpha=0.3, label=f'Rep {rep_num}' if rep_num <= 10 else "")
        
        # 膝角度のピーク位置に赤丸をプロット
        ax1.scatter(t_f, t_v, color='red', edgecolors='black', s=40, zorder=5)
        # レップ番号のテキストラベル
        ax1.text(t_f, t_v + (max_angle - min_angle) * 0.03, f"R{rep_num}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title(f"Extracted Repetition Intervals (Knee Angle Based) - {core_name}", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 凡例をスマートに結合
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=150)
    plt.close()
    
    print(f"  [Success] {crop_msg} -> 検出数: {len(peaks)} レップ 保存完了")


if __name__ == '__main__':
    print("==================================================")
    print(" スクワット波形 角度ベース区間切り出し一括解析スクリプト")
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
        
    # 入力フォルダ（posture_analyzed）内のCSVファイルを取得
    csv_files = glob.glob(os.path.join(INPUT_FOLDER_PATH, "*_analyzed.csv"))
    
    total_files = len(csv_files)
    print(f"解析対象ファイル数: {total_files} 件\n")
    
    for idx, file_path in enumerate(csv_files, 1):
        print(f"[{idx}/{total_files}] 処理中: {os.path.basename(file_path)}")
        try:
            process_file(file_path, OUTPUT_FOLDER_PATH, baseframes_df)
        except Exception as e:
            print(f"  [Error] エラーが発生しました: {e}")
            
    print("\nすべてのファイルの切り出しおよび解析処理が完了しました。")