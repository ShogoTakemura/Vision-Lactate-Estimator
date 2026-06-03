import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os

# 出力先フォルダの指定と自動作成
output_dir = r"C:\Users\ironm\squat_analyze\analyze_sensor\FP_comparison"
os.makedirs(output_dir, exist_ok=True)

# 1. データの読み込み（日本語エンコーディング対策付き）
df_sensor = pd.read_csv(
    r"C:\Users\ironm\squat_analyze\analyze_sensor\260428D\FP\resampled_30Hz\squat_1_30Hz.csv", 
    skiprows=5, 
    encoding='shift_jis'
)
df_model = pd.read_csv(
    r"C:\Users\ironm\squat_analyze\analyze_sensor\cop\squat01_correct_modelbase.csv"
)

# 2. 床反力(Fz)を用いて正確な時間同期のラグを算出
fz_raw = df_sensor['Fz'].values
model_force = df_model['FloorForce_y'].values

active_idx = np.where(fz_raw > 10)[0]
start_active = active_idx[0] if len(active_idx) > 0 else 0
end_active = active_idx[-1] if len(active_idx) > 0 else len(fz_raw)

fz_active = fz_raw[start_active:end_active+1]
fz_norm = (fz_active - np.mean(fz_active)) / np.std(fz_active)
model_norm = (model_force - np.mean(model_force)) / np.std(model_force)
corr = correlate(model_norm, fz_norm, mode='full')
lags = np.arange(-len(fz_norm) + 1, len(model_force))
best_lag = lags[np.argmax(corr)]

print(f"時間同期ラグ: {best_lag} フレーム")

# 3. 同期したアクティブ区間のCOPデータを抽出
model_cop_x = df_model['COP_x'].iloc[best_lag : best_lag + len(fz_active)].values
model_cop_z = df_model['COP_z'].iloc[best_lag : best_lag + len(fz_active)].values

sensor_cop_x = df_sensor['CopX'].iloc[start_active : end_active + 1].values
sensor_cop_y = df_sensor['CopY'].iloc[start_active : end_active + 1].values


# ==============================================================================
# 4. 原点ズレの補正 ＆ 【新規】X軸の向き（符号）反転処理
# ==============================================================================
# カメラ基準（右向き正）に統一するため、FPのX軸の差分に対してマイナス（-1）を掛けます
model_cop_x_rel = model_cop_x - model_cop_x[0]
model_cop_z_rel = model_cop_z - model_cop_z[0]

sensor_cop_x_rel = -(sensor_cop_x - sensor_cop_x[0])  # ← ここで符号を反転
sensor_cop_y_rel = sensor_cop_y - sensor_cop_y[0]


# 5. 完全に1対1で対応した補正済みCSVの作成・出力
df_cop_comparison = pd.DataFrame({
    'Model_COP_x_Relative': model_cop_x_rel,
    'Model_COP_z_Relative': model_cop_z_rel,
    'Sensor_CopX_Relative': sensor_cop_x_rel,
    'Sensor_CopY_Relative': sensor_cop_y_rel
})

output_csv_path = os.path.join(output_dir, 'squat01_cop_aligned_comparison.csv')
df_cop_comparison.to_csv(output_csv_path, index=False)
print(f"\n[CSV] 軸向き・オフセット補正済みの対応データを保存しました:\n--> {output_csv_path}")


# ==============================================================================
# 6. グラフ可視化と画像の保存（2枚に分割出力）
# ==============================================================================

# 画像1: 左右方向 (X軸) の移動量比較（符号反転により、波形の動きが一致します）
plt.figure(figsize=(12, 5))
plt.plot(df_cop_comparison['Model_COP_x_Relative'], label='Model COP_x (Relative: Right=+)', color='orange', linewidth=2)
plt.plot(df_cop_comparison['Sensor_CopX_Relative'], label='Sensor CopX (Relative: Flipped to Right=+)', color='blue', alpha=0.8, linewidth=1.5)
plt.title('COP X-Coordinate (Left-Right Displacement: Aligned & Flipped)')
plt.xlabel('Aligned Index / Frame')
plt.ylabel('Displacement from start [m] (Right is Positive)')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_img_x = os.path.join(output_dir, 'squat_cop_comparison_X.png')
plt.savefig(output_img_x, dpi=150)
plt.close()
print(f"[画像1] 左右方向（反転補正済）の比較グラフを保存しました:\n--> {output_img_x}")


# 画像2: 前後方向 (Model_Z vs Sensor_Y) の移動量比較
plt.figure(figsize=(12, 5))
plt.plot(df_cop_comparison['Model_COP_z_Relative'], label='Model COP_z (Relative)', color='red', linewidth=2)
plt.plot(df_cop_comparison['Sensor_CopY_Relative'], label='Sensor CopY (Relative)', color='green', alpha=0.8, linewidth=1.5)
plt.title('COP Anterior-Posterior (Displacement) Comparison')
plt.xlabel('Aligned Index / Frame')
plt.ylabel('Displacement from start [m]')
plt.legend()
plt.grid(True)
plt.tight_layout()

output_img_ap = os.path.join(output_dir, 'squat_cop_comparison_AP.png')
plt.savefig(output_img_ap, dpi=150)
plt.close()
print(f"[画像2] 前後方向の比較グラフを保存しました:\n--> {output_img_ap}")