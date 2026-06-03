"""
estimate_blc.py
===============
血中乳酸濃度（BLC）推定プログラム

使用モデル（論文の設定に準拠）：
    - MR  : 重回帰分析（最小二乗法）
    - SVR : サポートベクター回帰（RBF カーネル）
    - NN  : ニューラルネットワーク（1 隠れ層 10 ノード, ReLU）

評価：5 分割交差検証 → RMSE / R²
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model  import LinearRegression
from sklearn.svm           import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics         import mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# ① 設定
# ═══════════════════════════════════════════════════════════════

# 入力 CSV のパス
INPUT_CSV = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\work_calculated\input_database_with_set.csv"

# 推定する目的変数
#   'after_lac' : 乳酸値の真値
#   'lac'       : 乳酸変化量
TARGET = 'after_lac'

# 入力特徴量
# ※ 論文では年齢(age)も使用しているが，本データセットには含まれないため除外
FEATURE_COLS = [
    'height',                       # 身長 [cm]
    'mass',                         # 体重 [kg]
    'gender',                       # 性別 (男性=1, 女性=0)
    'set',
    'age',                          # セット数
    'Set_Total_Work(J)',             # 位置エネルギー [J]
    'Set_Total_Kinetic_Energy(J)',   # 運動エネルギー [J]
]

# 交差検証の分割数
N_SPLITS = 5

# テスト被験者の Subject 列に含まれる文字列（前方一致）
# 空リストにすると全データを交差検証に使用する
# 例: ['squat01', 'squat02', 'squat03']
TEST_SUBJECTS: list[str] = ['squat01', 'squat02', 'squat03']

# 出力ディレクトリ（INPUT_CSV と同じ場所）
OUTPUT_DIR = os.path.dirname(INPUT_CSV)


# ═══════════════════════════════════════════════════════════════
# ② データ読み込み・前処理
# ═══════════════════════════════════════════════════════════════

def load_data(path: str):
    df = pd.read_csv(path)

    # テスト被験者 / 訓練被験者 を分離
    if TEST_SUBJECTS:
        test_mask = df['Subject'].astype(str).apply(
            lambda s: any(s.startswith(t) for t in TEST_SUBJECTS)
        )
    else:
        test_mask = pd.Series([False] * len(df), index=df.index)

    df_test_raw = df[test_mask].copy()

    # 訓練データ: TARGET と特徴量が揃っている行のみ
    df_train = df[~test_mask].dropna(subset=FEATURE_COLS + [TARGET]).copy()

    print(f"訓練データ: {len(df_train)} 件")
    print(f"テストデータ (lac 有): {df_test_raw.dropna(subset=[TARGET]).shape[0]} 件")
    print(f"テストデータ (lac 無, 予測のみ): {df_test_raw[df_test_raw[TARGET].isna()].shape[0]} 件")

    return df_train, df_test_raw


# ═══════════════════════════════════════════════════════════════
# ③ モデル定義（論文準拠）
# ═══════════════════════════════════════════════════════════════

def build_models():
    """
    MR  : 重回帰分析
          最小二乗法, ロバスト処理なし (fit_intercept=True)

    SVR : サポートベクター回帰
          C (Box Constraint) = 10.5
          epsilon             = 1.0
          kernel              = RBF（ガウスカーネル）
          kernel_scale        = 2.5
            → sklearn の gamma = 1 / (2 * scale²) = 1/12.5 ≒ 0.08

    NN  : ニューラルネットワーク
          構造: 入力層 → 隠れ層 (10 ノード, ReLU) → 出力層
          最適化: Adam
    """
    # SVR の gamma: MATLAB KernelScale=2.5 に相当
    kernel_scale = 2.5
    gamma_svr = 1.0 / (2.0 * kernel_scale ** 2)   # = 0.08

    models = {
        'MR': Pipeline([
            ('model', LinearRegression())
        ]),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  SVR(C=10.5, epsilon=1.0, kernel='rbf', gamma=gamma_svr))
        ]),
        'NN': Pipeline([
            ('scaler', StandardScaler()),
            ('model',  MLPRegressor(
                hidden_layer_sizes=(10,),
                activation='relu',
                solver='adam',
                max_iter=5000,
                random_state=42,
                n_iter_no_change=50,
            ))
        ]),
    }
    return models


# ═══════════════════════════════════════════════════════════════
# ④ 5 分割交差検証
# ═══════════════════════════════════════════════════════════════

def cross_validate(models: dict, X: np.ndarray, y: np.ndarray):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    cv_results = {}

    print(f"\n{'─'*55}")
    print(f"{'モデル':^8} {'RMSE [mmol/L]':^15} {'R²':^10}")
    print(f"{'─'*55}")

    for name, pipe in models.items():
        y_pred = cross_val_predict(pipe, X, y, cv=kf)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2   = r2_score(y, y_pred)
        cv_results[name] = {'y_pred': y_pred, 'RMSE': rmse, 'R2': r2}
        print(f"{'  '+name:<10} {rmse:^15.4f} {r2:^10.4f}")

    print(f"{'─'*55}\n")
    return cv_results


# ═══════════════════════════════════════════════════════════════
# ⑤ テストデータ評価（lac 値がある場合）
# ═══════════════════════════════════════════════════════════════

def evaluate_test(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray):
    test_results = {}

    print(f"{'─'*55}")
    print(f"{'モデル':^8} {'RMSE [mmol/L]':^15} {'R²':^10}  (テストデータ)")
    print(f"{'─'*55}")

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        test_results[name] = {'y_pred': y_pred, 'RMSE': rmse, 'R2': r2}
        print(f"{'  '+name:<10} {rmse:^15.4f} {r2:^10.4f}")

    print(f"{'─'*55}\n")
    return test_results


# ═══════════════════════════════════════════════════════════════
# ⑥ 可視化
# ═══════════════════════════════════════════════════════════════

COLORS = {'MR': '#4472C4', 'SVR': '#ED7D31', 'NN': '#FFC000'}

def plot_cv_metrics(cv_results: dict, save_dir: str):
    """交差検証 RMSE / R² の棒グラフ（Fig. 5-2 相当）"""
    names = list(cv_results.keys())
    rmses = [cv_results[n]['RMSE'] for n in names]
    r2s   = [cv_results[n]['R2']   for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = [COLORS[n] for n in names]

    axes[0].bar(names, rmses, color=colors, edgecolor='k', width=0.5)
    axes[0].set_ylabel('RMSE [mmol/L]', fontsize=12)
    axes[0].set_title('Cross-Validation RMSE', fontsize=13)
    axes[0].set_ylim(0, max(rmses) * 1.3)
    for i, v in enumerate(rmses):
        axes[0].text(i, v + max(rmses)*0.03, f'{v:.3f}', ha='center', fontsize=10)

    axes[1].bar(names, r2s, color=colors, edgecolor='k', width=0.5)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('Cross-Validation R²', fontsize=13)
    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='--')
    for i, v in enumerate(r2s):
        offset = max(r2s) * 0.03 if v >= 0 else -max(abs(r for r in r2s)) * 0.08
        axes[1].text(i, v + offset, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, 'cv_metrics.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"グラフ保存: {path}")


def plot_scatter(y_true, results: dict, title_prefix: str, save_name: str, save_dir: str):
    """実測値 vs 推定値の散布図（Fig. 5-3〜5-5 相当）"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]

    lim_max = max(np.max(y_true) * 1.1, 1)

    for ax, (name, res) in zip(axes, results.items()):
        y_pred = res['y_pred']
        ax.scatter(y_true, y_pred, color=COLORS[name], alpha=0.7,
                   edgecolors='k', linewidths=0.4, s=50)
        ax.plot([0, lim_max], [0, lim_max], 'k-', linewidth=1.2, label='y=x')
        ax.set_xlabel(f'Actual {TARGET} [mmol/L]', fontsize=11)
        ax.set_ylabel(f'Predict {TARGET} [mmol/L]', fontsize=11)
        ax.set_title(f'{title_prefix} result by {name}', fontsize=12)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.text(0.05, 0.92, f'RMSE={res["RMSE"]:.3f}\nR²={res["R2"]:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{save_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"グラフ保存: {path}")


# ─── 実測値 vs 予測値 CSV ──────────────────────────────────────

def save_predictions_csv(df_source: pd.DataFrame, y_true: np.ndarray,
                         results: dict, save_dir: str, prefix: str) -> pd.DataFrame:
    """
    実測値と全モデルの予測値を 1 行 1 サンプルで CSV に保存する。

    出力列: Subject, set, actual_{TARGET}, pred_{TARGET}_MR,
            pred_{TARGET}_SVR, pred_{TARGET}_NN
    """
    df_out = df_source[['Subject', 'set']].copy().reset_index(drop=True)
    df_out[f'actual_{TARGET}'] = np.round(y_true, 3)
    for name, res in results.items():
        df_out[f'pred_{TARGET}_{name}'] = np.round(res['y_pred'], 3)

    # 誤差列も付加
    for name in results:
        df_out[f'error_{name}'] = np.round(
            df_out[f'pred_{TARGET}_{name}'] - df_out[f'actual_{TARGET}'], 3
        )

    path = os.path.join(save_dir, f'{prefix}_actual_vs_predicted.csv')
    df_out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"CSV 保存: {path}")
    return df_out


# ─── 実測値 vs 予測値 折れ線グラフ ────────────────────────────

def plot_actual_vs_predicted(df_source: pd.DataFrame, y_true: np.ndarray,
                              results: dict, save_dir: str, prefix: str):
    """
    サンプルごとの実測値と推定値を並べた折れ線グラフ。

    上段: 実測値 vs 各モデル推定値（折れ線）
    下段: モデルごとの残差（誤差バー）
    """
    n = len(y_true)
    x = np.arange(n)

    # X 軸ラベル: Subject_setN
    src = df_source.reset_index(drop=True)
    labels = []
    for _, row in src.iterrows():
        subj = str(row['Subject'])
        s    = int(row['set']) if pd.notna(row.get('set')) else '?'
        labels.append(f"{subj}\ns{s}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(max(14, n * 0.35), 8),
        gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    # ── 上段: 実測値 + 推定値 ──────────────────────────────
    ax_top.plot(x, y_true, 'ko-', linewidth=1.8, markersize=5,
                label=f'Actual {TARGET}', zorder=6)

    line_styles = {'MR': '--', 'SVR': '-.', 'NN': ':'}
    markers     = {'MR': 's',  'SVR': '^',  'NN': 'D'}
    for name, res in results.items():
        ax_top.plot(x, res['y_pred'],
                    color=COLORS[name],
                    linestyle=line_styles.get(name, '-'),
                    marker=markers.get(name, 'o'),
                    markersize=4, linewidth=1.4,
                    label=f'{name}  RMSE={res["RMSE"]:.3f}  R²={res["R2"]:.3f}')

    ax_top.set_ylabel(f'{TARGET} [mmol/L]', fontsize=12)
    ax_top.set_title(f'{prefix}: Actual vs Predicted {TARGET}', fontsize=13)
    ax_top.legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax_top.grid(True, linestyle='--', alpha=0.4)

    # ── 下段: 残差（予測値 − 実測値） ─────────────────────
    bar_width = 0.25
    offsets   = {'MR': -bar_width, 'SVR': 0, 'NN': bar_width}
    for name, res in results.items():
        residuals = np.array(res['y_pred']) - y_true
        ax_bot.bar(x + offsets[name], residuals,
                   width=bar_width, color=COLORS[name],
                   alpha=0.75, edgecolor='k', linewidth=0.3,
                   label=name)

    ax_bot.axhline(0, color='k', linewidth=1.0)
    ax_bot.set_ylabel('Residual\n[mmol/L]', fontsize=10)
    ax_bot.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax_bot.grid(True, linestyle='--', alpha=0.4, axis='y')

    # X 軸ラベル（多すぎる場合は間引く）
    step = max(1, n // 40)
    ax_bot.set_xticks(x[::step])
    ax_bot.set_xticklabels(labels[::step], rotation=90, fontsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{prefix}_actual_vs_predicted.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"グラフ保存: {path}")


# ═══════════════════════════════════════════════════════════════
# ⑦ 未知データの予測（lac 値がない場合）
# ═══════════════════════════════════════════════════════════════

def predict_new(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                df_new: pd.DataFrame) -> pd.DataFrame:
    """
    lac 値のない行に対して全モデルで予測する。
    set 列が NaN の場合は File_Name 末尾の数字から推定する。
    （例: squat01 → set=1, squat02 → set=2, squat03 → set=3）
    """
    df_result = df_new.copy()

    # set が NaN の場合は Subject 名末尾の数字を使う
    set_missing = df_result['set'].isna()
    if set_missing.any():
        inferred = df_result.loc[set_missing, 'Subject'].astype(str).str.extract(r'(\d+)$')[0]
        df_result.loc[set_missing, 'set'] = pd.to_numeric(inferred, errors='coerce')
        print(f"  set 推定: {dict(zip(df_result.loc[set_missing,'Subject'], df_result.loc[set_missing,'set']))}")

    valid_mask = df_result[FEATURE_COLS].notna().all(axis=1)
    if valid_mask.sum() == 0:
        print("予測対象データに有効な特徴量がありません。")
        return df_result

    X_new = df_result.loc[valid_mask, FEATURE_COLS].values

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_new)
        col = f'pred_{TARGET}_{name}'
        df_result.loc[valid_mask, col] = np.round(pred, 3)

    return df_result


# ═══════════════════════════════════════════════════════════════
# ⑧ メイン処理
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # データ読み込み
    df_train, df_test_raw = load_data(INPUT_CSV)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET].values

    # モデル構築
    models = build_models()

    # ── 5 分割交差検証 ──────────────────────────────────────
    print("【 5 分割交差検証 】")
    cv_results = cross_validate(models, X_train, y_train)

    # ── テストデータ評価（lac 値がある場合）────────────────
    df_test_labeled   = df_test_raw.dropna(subset=FEATURE_COLS + [TARGET])
    df_test_unlabeled = df_test_raw[
        df_test_raw[TARGET].isna() & df_test_raw[FEATURE_COLS].notna().all(axis=1)
    ]

    if not df_test_labeled.empty:
        print("【 テストデータ評価 】")
        X_test = df_test_labeled[FEATURE_COLS].values
        y_test = df_test_labeled[TARGET].values
        test_results = evaluate_test(models, X_train, y_train, X_test, y_test)
    else:
        test_results = None

    # ── 可視化 ──────────────────────────────────────────────
    plot_cv_metrics(cv_results, OUTPUT_DIR)
    plot_scatter(y_train, cv_results,
                 title_prefix='Cross-Validation',
                 save_name='cv_scatter',
                 save_dir=OUTPUT_DIR)
    # 実測値 vs 予測値（交差検証）
    save_predictions_csv(df_train, y_train, cv_results, OUTPUT_DIR, 'cv')
    plot_actual_vs_predicted(df_train, y_train, cv_results, OUTPUT_DIR, 'cv')

    if test_results:
        plot_scatter(y_test, test_results,
                     title_prefix='Test',
                     save_name='test_scatter',
                     save_dir=OUTPUT_DIR)
        # 実測値 vs 予測値（テストデータ）
        save_predictions_csv(df_test_labeled, y_test, test_results, OUTPUT_DIR, 'test')
        plot_actual_vs_predicted(df_test_labeled, y_test, test_results, OUTPUT_DIR, 'test')

    # ── 未知データの予測 ────────────────────────────────────
    # set が NaN でも predict_new 内で推定するため、他の特徴量で判定
    non_set_features = [c for c in FEATURE_COLS if c != 'set']
    df_test_unlabeled = df_test_raw[
        df_test_raw[TARGET].isna() & df_test_raw[non_set_features].notna().all(axis=1)
    ]
    if not df_test_unlabeled.empty:
        print("【 未知データ予測 】")
        df_pred = predict_new(models, X_train, y_train, df_test_unlabeled)
        pred_cols = [c for c in df_pred.columns if c.startswith('pred_')]
        print(df_pred[['Subject', 'set'] + FEATURE_COLS + pred_cols].to_string(index=False))
        pred_path = os.path.join(OUTPUT_DIR, 'blc_predictions.csv')
        df_pred.to_csv(pred_path, index=False, encoding='utf-8-sig')
        print(f"\n予測結果保存: {pred_path}")

    # ── 交差検証結果を CSV 保存 ─────────────────────────────
    rows = []
    for name, res in cv_results.items():
        rows.append({'Model': name, 'Phase': 'CrossValidation',
                     'RMSE': round(res['RMSE'], 4), 'R2': round(res['R2'], 4)})
    if test_results:
        for name, res in test_results.items():
            rows.append({'Model': name, 'Phase': 'Test',
                         'RMSE': round(res['RMSE'], 4), 'R2': round(res['R2'], 4)})

    df_metrics = pd.DataFrame(rows)
    metrics_path = os.path.join(OUTPUT_DIR, 'blc_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"\n評価指標保存: {metrics_path}")
    print("\n✅ 全処理完了")


if __name__ == '__main__':
    main()
