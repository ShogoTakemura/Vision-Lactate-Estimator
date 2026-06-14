import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os
from pathlib import Path

from squat_core.paths import PROJECT_ROOT

_ANALYZE_SENSOR = PROJECT_ROOT / "analyze_sensor"
_FP_DIR         = _ANALYZE_SENSOR / "260428D" / "FP" / "resampled_30Hz"

# 出力先フォルダの指定と自動作成
output_dir = str(_ANALYZE_SENSOR / "FP_comparison" / "squat03" / "remove_outlier")
os.makedirs(output_dir, exist_ok=True)

# 1. データの読み込み
df_sensor = pd.read_csv(
    str(_FP_DIR / "squat_3_30Hz.csv"),
    skiprows=5,
    encoding='shift_jis'
)
df_model = pd.read_csv(
    str(_ANALYZE_SENSOR / "com_features" / "squat03_cleaned.csv")
)

# ★追加：リザルト（開始・終了フレーム）の読み込み
df_results = pd.read_csv(
    str(_ANALYZE_SENSOR / "peak" / "results" / "squat03_results.csv")
)

# # ==============================================================================
# # 1.5. センサーデータ (Fz) の外れ値除去と線形補完
# # ==============================================================================
# target_sensor_col = 'Fz'
# window_size = 30
# rolling_median = df_sensor[target_sensor_col].rolling(window=window_size, center=True, min_periods=1).median()
# deviation = np.abs(df_sensor[target_sensor_col] - rolling_median)
# threshold = df_sensor[target_sensor_col].std() * 1.5
# outlier_mask = deviation > threshold
# df_sensor.loc[outlier_mask, target_sensor_col] = np.nan
# df_sensor[target_sensor_col] = df_sensor[target_sensor_col].interpolate(method='linear', limit_direction='both')

# ==============================================================================
# 1.5. センサーデータ (Fz) の外れ値除去と線形補完
# ==============================================================================

target_sensor_col = 'Fz'

# コピー
signal = df_sensor[target_sensor_col].copy()

# ------------------------------------------------------------------------------
# ① 明らかな異常値を除去（物理範囲ベース）
# ------------------------------------------------------------------------------
signal[(signal > 2500) | (signal < -100)] = np.nan

# ------------------------------------------------------------------------------
# ② Rolling Median による局所外れ値除去
# ------------------------------------------------------------------------------
window_size = 15

rolling_median = (
    signal
    .rolling(window=window_size,
             center=True,
             min_periods=1)
    .median()
)

deviation = np.abs(signal - rolling_median)

# MADベース（stdより頑健）
mad = np.nanmedian(deviation)

threshold = mad * 4

outlier_mask = deviation > threshold

signal[outlier_mask] = np.nan

# ------------------------------------------------------------------------------
# ③ 線形補間
# ------------------------------------------------------------------------------
signal = signal.interpolate(
    method='linear',
    limit_direction='both'
)

# DataFrameへ戻す
df_sensor[target_sensor_col] = signal

print(f"除去した外れ値数: {outlier_mask.sum()}")

# ==============================================================================
# 2. 解析区間（スクワット区間全体）の定義
# ==============================================================================
# df_results から全体の開始フレーム（最小値）と終了フレーム（最大値）を取得し、前後10フレームの余裕を持たせる
model_start = int(df_results['start_frame'].min()) + 10
model_end = int(df_results['end_frame'].max()) - 10

# データフレームの範囲外に出ないようにクリッピング（念のため）
model_start = max(0, model_start)
model_end = min(len(df_model) - 1, model_end)

print(f"★分析対象のモデルフレーム区間（開始前-1 ～ 終了後+1）: {model_start} 行目 ～ {model_end} 行目")

fz_raw = df_sensor['Fz'].values
model_force = df_model['Floor Force(only y)_y'].values

# 同期に用いるモデルデータを指定区間だけで切り出し
model_window = model_force[model_start : model_end + 1]

# センサーデータのアクティブな区間（Fz > 10 N）を抽出
active_idx = np.where(fz_raw > 10)[0]
start_active = active_idx[0] if len(active_idx) > 0 else 0
end_active = active_idx[-1] if len(active_idx) > 0 else len(fz_raw) - 1
fz_active = fz_raw[start_active : end_active + 1]

# ==============================================================================
# 3. 相互相関による最適な時間ズレ（ラグ）の自動算出
# ==============================================================================
# スケールを合わせて相関を計算
fz_norm = (fz_active - np.mean(fz_active)) / np.std(fz_active)
model_norm = (model_window - np.mean(model_window)) / np.std(model_window)

# fz_norm の上を model_norm がスライドして最も重なる位置を探す
corr = correlate(fz_norm, model_norm, mode='full')
lags = np.arange(-len(model_norm) + 1, len(fz_norm))
best_lag_in_fz = lags[np.argmax(corr)]

# オフセットの計算 (model_force[0] が fz_raw のどのインデックスに該当するか)
offset = start_active + best_lag_in_fz - model_start
print(f"最適なアライメント位置 (オフセット): センサー側のインデックスを {offset} ずらします")

# ==============================================================================
# 4. 同期データの作成（全フレームに対してオフセットを適用）
# ==============================================================================
fz_aligned_full = np.full(len(df_model), np.nan)
for i in range(len(df_model)):
    fz_idx = i + offset
    if 0 <= fz_idx < len(fz_raw):
        fz_aligned_full[i] = fz_raw[fz_idx]

df_output = pd.DataFrame({
    'Model_Original_Frame': np.arange(len(df_model)),
    'Floor Force(only y)_y': model_force,
    'Aligned_Sensor_Fz': fz_aligned_full
})

# ==============================================================================
# 5. 指定区間のみを抽出し、評価と出力
# ==============================================================================
# スクワット開始前10 ～ 終了後10 の区間のみを切り出す
df_output_cropped = df_output.iloc[model_start : model_end + 1].copy()

# 欠損値（NaN）がある行を削除し、完全に同期できた部分だけを残す
df_output_clean = df_output_cropped.dropna().reset_index(drop=True)

# 評価指標の計算
model_overlap = df_output_clean['Floor Force(only y)_y'].values
fz_overlap = df_output_clean['Aligned_Sensor_Fz'].values

correlation_coef = np.corrcoef(model_overlap, fz_overlap)[0, 1]
rmse = np.sqrt(np.mean((model_overlap - fz_overlap) ** 2))
mae = np.mean(np.abs(model_overlap - fz_overlap))

print("\n--- 指定区間での比較レポート ---")
print(f"相関係数 (Pearson R): {correlation_coef:.4f}")
print(f"二乗平均平方根誤差 (RMSE): {rmse:.2f} N")
print(f"平均絶対誤差 (MAE): {mae:.2f} N")

# CSV出力
output_csv_path = os.path.join(output_dir, 'squat02_aligned_comparison_cropped.csv')
df_output_clean.to_csv(output_csv_path, index=False)
print(f"\n[CSV] 抽出・同期済みのデータを保存しました（全 {len(df_output_clean)} 行）:\n--> {output_csv_path}")

# ==============================================================================
# 6. グラフ可視化と画像の保存
# ==============================================================================
# 画像1: 同期前のそのままの比較（全体のタイムライン）
plt.figure(figsize=(12, 5))
plt.plot(fz_raw, label='Fz (Sensor: Cleaned & Interpolated)', color='blue', alpha=0.7)
plt.plot(model_force, label='Floor Force(only y)_y (Model: Raw)', color='orange', alpha=0.7)
plt.title('Raw Comparison (Entire Timeline)')
plt.xlabel('Index / Frame')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_img_raw = os.path.join(output_dir, 'squat_force_comparison_raw.png')
plt.savefig(output_img_raw, dpi=150)
plt.close()

# 画像2: 時間軸を同期させた指定区間のみの比較
plt.figure(figsize=(12, 5))
plt.plot(df_output_clean['Floor Force(only y)_y'], label='Floor Force(only y)_y (Model)', color='orange', linewidth=2)
plt.plot(df_output_clean['Aligned_Sensor_Fz'], label='Fz (Sensor: Aligned)', color='blue', alpha=0.8, linewidth=1.5)
plt.title(f'Aligned Comparison (Cropped: Model Frames {model_start} to {model_end})')
plt.xlabel('Aligned Index (Cropped Region)')
plt.ylabel('Force [N]')
plt.legend()
plt.grid(True)
plt.tight_layout()
output_img_aligned = os.path.join(output_dir, 'squat_force_comparison_aligned_cropped.png')
plt.savefig(output_img_aligned, dpi=150)
plt.close()
print(f"[画像2] 指定区間の同期済みグラフを保存しました:\n--> {output_img_aligned}")
