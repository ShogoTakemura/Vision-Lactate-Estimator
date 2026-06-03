import pandas as pd
import numpy as np
from scipy import stats
import os

def remove_outliers_time_series(file_path, threshold=3.0, method='zscore'):
    """
    時系列データの連続性を保ちながら外れ値を処理する関数
    外れ値を検出し、線形補間（Interpolation）で穴埋めします。
    """
    # データの読み込み
    df = pd.read_csv(file_path)
    df_clean = df.copy()
    
    if method == 'zscore':
        # Zスコアの絶対値を計算
        z_scores = np.abs(stats.zscore(df))
        # 閾値を超える要素をNaNに置換
        df_clean[z_scores > threshold] = np.nan
        
    elif method == 'iqr':
        # IQR法 (四分位範囲) による検出
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # 範囲外の要素をNaNに置換
        df_clean[(df < lower_bound) | (df > upper_bound)] = np.nan
        
    # 外れ値（NaN）の前後を線形補間
    # 最初や最後にNaNが来たときのために bfill/ffill も合わせて行う
    df_clean = df_clean.interpolate(method='linear').bfill().ffill()
    
    return df_clean

# 処理対象のファイルリスト
target_files = [
    r"C:\Users\ironm\squat_analyze\analyze_sensor\com_features\squat01_correct_modelbase.csv",
    r"C:\Users\ironm\squat_analyze\analyze_sensor\com_features\squat02_correct_modelbase.csv",
    r"C:\Users\ironm\squat_analyze\analyze_sensor\com_features\squat03_correct_modelbase.csv"
]

# 各ファイルの処理と保存
for file in target_files:
    if os.path.exists(file):
        print(f"Processing: {file}...")
        
        # Zスコア法（閾値3）で処理
        df_cleaned = remove_outliers_time_series(file, threshold=3.0, method='zscore')
        
        # 出力ファイル名の生成
        output_file = file.replace('_correct_modelbase.csv', '_cleaned.csv')
        df_cleaned.to_csv(output_file, index=False)
        
        print(f"--> Saved to: {output_file}")
    else:
        print(f"File not found: {file}")