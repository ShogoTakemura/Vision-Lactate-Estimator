import numpy as np
import pandas as pd
from pathlib import Path

from squat_core.paths import PROJECT_ROOT

_FP_DIR = PROJECT_ROOT / "analyze_sensor" / "260428D" / "FP" / "resampled_30Hz"


def calculate_total_weight(file_path, window_size=30, threshold_fz=300):
    """フォースプレートのデータから被験者＋負荷の総重量(kg)を自動算出する関数

    Parameters:
    -----------
    file_path : str
        CSVファイルのパス
    window_size : int
        静止区間を判定するためのウィンドウサイズ（30Hzデータの場合、30フレーム＝1秒間）
    threshold_fz : float
        人が乗っていると判断する垂直抗力(Fz)のしきい値（N）

    Returns:
    --------
    float
        算出された総重量 (kg)
    """
    # 1. メタデータ（最初の4行）をスキップしてデータを読み込み
    try:
        # まずは標準の UTF-8 で読み込みを試みる
        df = pd.read_csv(file_path, skiprows=4, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # 文字コードエラーが出た場合、Windows標準の CP932 (Shift_JIS) で読み込みを試みる
            df = pd.read_csv(file_path, skiprows=4, encoding="cp932")
        except Exception as e:
            print(f"ファイルの読み込みに失敗しました（文字コードエラー）: {e}")
            return None
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return None

    # カラム名の余分な空白を削除
    df.columns = df.columns.str.strip()

    if "Fz" not in df.columns:
        print("エラー: データに 'Fz' (垂直抗力) カラムが見つかりません。")
        return None

    # 2. 移動平均と移動標準偏差を計算して、最もブレの少ない「静止区間」を探す
    # window_size分（30フレーム=1秒）のデータのばらつき（標準偏差）を計算
    df["Fz_std"] = df["Fz"].rolling(window=window_size, center=True).std()
    df["Fz_mean"] = df["Fz"].rolling(window=window_size, center=True).mean()

    # 「人が乗っている状態（Fz > しきい値）」かつ「データが欠損していない行」を抽出
    valid_standing = df[(df["Fz"] > threshold_fz) & (df["Fz_std"].notna())]

    if valid_standing.empty:
        print(
            f"エラー: Fzが {threshold_fz} N を超える安定した立位区間が検出できませんでした。"
        )
        return None

    # 3. 標準偏差（ばらつき）が最も小さいインデックス（最もきれいに静止している瞬間）を特定
    best_index = valid_standing["Fz_std"].idxmin()

    # その瞬間の前後（window_size分）の平均垂直抗力を取得
    stable_fz_mean = valid_standing.loc[best_index, "Fz_mean"]

    # 4. 重力加速度を用いて ニュートン(N) から キログラム(kg) に変換
    g = 9.80665  # 標準重力加速度
    total_weight_kg = stable_fz_mean / g

    # 5. 検出結果の表示
    start_frame = max(0, best_index - window_size // 2)
    end_frame = min(len(df), best_index + window_size // 2)

    print("=" * 50)
    print(f"分析ファイル: {file_path}")
    print(f"検出された最も安定した静止区間: {start_frame} 〜 {end_frame} フレーム")
    print(f"静止区間の平均垂直抗力 (Fz): {stable_fz_mean:.2f} N")
    print(f"算出された総重量 (質量): {total_weight_kg:.2f} kg")
    print("=" * 50)

    return total_weight_kg


# --- スクリプトの実行例 ---
if __name__ == "__main__":
    # ファイル名を設定して実行
    file_name = str(_FP_DIR / "weight_bar_30Hz.csv")
    weight = calculate_total_weight(file_name)
