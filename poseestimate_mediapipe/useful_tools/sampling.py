import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os
import glob

# ==========================================
# 1. 処理するファイルのリスト（フォルダ内の全CSVを指定なども可能）
# ==========================================
# カレントディレクトリに元のCSVがある前提です。
# 別のフォルダにある場合は r"C:\path\to\folder\*.csv" のように変更してください
file_list = [
    r'C:\Users\ironm\Desktop\260428D\FP\squat_1.csv',
    r'C:\Users\ironm\Desktop\260428D\FP\squat_2.csv',
    r'C:\Users\ironm\Desktop\260428D\FP\squat_3.csv',
    r'C:\Users\ironm\Desktop\260428D\FP\squat_4.csv',
    r'C:\Users\ironm\Desktop\260428D\FP\weight_bar.csv'
]

# 出力先のフォルダ（なければ作成されます）
output_dir = r'C:\Users\ironm\Desktop\260428D\FP\resampled_30Hz'
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 30Hzへのリサンプリング処理
# ==========================================
original_hz = 1000
target_hz = 30

for filename in file_list:
    if not os.path.exists(filename):
        print(f"スキップ: {filename} が見つかりません。")
        continue

    # 先頭のメタデータ（サンプリング周波数の行など）を飛ばして6行目から読み込む
    df = pd.read_csv(filename, skiprows=5, encoding='shift_jis')
    
    # 元の時間軸 (1000Hz = 0.001秒刻み)
    time_orig = np.arange(len(df)) * (1.0 / original_hz)
    
    # 新しい時間軸 (30Hz = 約0.0333秒刻み)
    time_new = np.arange(0, time_orig[-1], 1.0 / target_hz)
    
    df_new = pd.DataFrame()
    
    # 各列の補間
    for col in df.columns:
        if df[col].dtype.kind in 'bifc': # 数値データの場合
            # 線形補間関数を作成
            f = interp1d(time_orig, df[col], kind='linear', fill_value="extrapolate")
            
            # デジタル信号(0/1)と思われる列は四捨五入して整数にする
            if col in ['同期信号', 'ExtIn1', 'ExtIn2']:
                df_new[col] = np.round(f(time_new)).astype(int)
            else:
                df_new[col] = f(time_new)

    # ==========================================
    # 3. CSVとして保存（元のフォーマットを再現）
    # ==========================================
    # 元のファイル名に「_30Hz」を付けて保存
    base_name = os.path.basename(filename)
    name_without_ext, ext = os.path.splitext(base_name)
    out_name = os.path.join(output_dir, f"{name_without_ext}_30Hz{ext}")
    
    # 1行目に新しいサンプリング周波数を書き込み、空行を挟んでからデータを追記する
    with open(out_name, 'w', encoding='shift_jis', errors='replace') as f:
        f.write(f"サンプリング周波数,{target_hz}\n\n\n\n\n")
        
    df_new.to_csv(out_name, mode='a', index=False, encoding='shift_jis')
    
    print(f"変換完了: {base_name} -> {out_name} (データ数: {len(df)}行 -> {len(df_new)}行)")

print("\nすべての処理が完了しました！")