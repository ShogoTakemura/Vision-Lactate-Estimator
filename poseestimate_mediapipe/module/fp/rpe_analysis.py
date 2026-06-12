"""
rpe_analysis.py
===============
RPE (Rate of Perceived Exertion) 推定のための特徴量分析スクリプト

対応データ:
    squat02_results.csv / squat03_results.csv  : per-rep RPE + フレーム情報
    squat02_correct_modelbase.csv / squat03_*  : 床反力 (FloorForce_y) [N]
    squat_2_30Hz.csv / squat_3_30Hz.csv        : フォースプレート実測値

出力:
    rpe_feature_analysis.png  : 各特徴量とRPEの相関分析
    rpe_prediction.png        : LOO-CV予測結果
    rpe_features.csv          : per-rep 特徴量テーブル
    rpe_correlation.csv       : 相関係数テーブル

依存ライブラリ:
    numpy, pandas, scipy, scikit-learn, matplotlib

使い方:
    python rpe_analysis.py
    (カレントディレクトリにデータファイルが存在することを前提)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from poseestimate_mediapipe.config.constants import GRAVITY
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 設定
# ============================================================
FPS          = 30.0      # フレームレート [Hz]
# GRAVITY is imported from poseestimate_mediapipe.config.constants
LP_CUTOFF    = 6.0       # ローパスフィルタ遮断周波数 [Hz]
RIDGE_ALPHA  = 1.0       # Ridge回帰の正則化係数

# squat02/03 の同期ラグ (FP→ビデオ frame offset = com_start - lag)
SYNC = {
    'squat02': {'lag': 84, 'com_start': 662},
    'squat03': {'lag': 82, 'com_start': 1137},
}

DATA_DIR = Path(r'C:\Users\ironm\squat_analyze\poseestimate_mediapipe\module\fp')   # データファイルのパス (必要に応じて変更)
OUT_DIR  = DATA_DIR   # 出力先

COLORS = {'squat02': '#378ADD', 'squat03': '#E24B4A'}

# ============================================================
# 1. FP ピーク検出
# ============================================================

def load_fp(path: Path, encoding: str = 'cp932') -> pd.DataFrame:
    """FP CSV を読み込む (先頭5行をスキップ)"""
    try:
        return pd.read_csv(path, skiprows=5, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(path, skiprows=5, encoding='utf-8')


def detect_reps_from_fp(fp_df: pd.DataFrame,
                         fp_to_video_offset: int,
                         fps: float = FPS,
                         lp_cutoff: float = LP_CUTOFF) -> tuple[pd.DataFrame, float]:
    """
    FP Fz からスクワット rep を検出する。

    手順:
        1. Fz を低域通過フィルタ (lp_cutoff Hz) でスムージング
        2. Fz > body_weight * 1.05 の concentric peak を検出
        3. 隣接ピーク間の谷 (squat bottom) を探索
        4. FP フレーム index → ビデオフレーム index に変換

    Returns:
        df_reps : per-rep 特徴量 DataFrame
        body_weight : 推定体重 [N]
    """
    fz_all = fp_df['Fz'].values
    on = np.where(np.abs(fz_all) > 200)[0]
    fp_start = int(on[0])
    fz = fz_all[fp_start: int(on[-1]) + 1]

    # 体重推定 (on-plate 区間の中央 80% 平均)
    n = len(fz)
    bw = float(np.mean(fz[int(n * 0.1): int(n * 0.9)]))

    # ローパスフィルタ
    b, a = butter(4, lp_cutoff / (fps / 2.0), btype='low')
    fz_filt = filtfilt(b, a, fz)

    # Concentric peak 検出
    peaks, _ = find_peaks(fz_filt,
                           height    = bw * 1.05,
                           distance  = int(fps * 0.8),
                           prominence= bw * 0.04)

    # 隣接ピーク間の谷
    valleys = [peaks[i] + int(np.argmin(fz_filt[peaks[i]: peaks[i + 1]]))
               for i in range(len(peaks) - 1)]
    valleys = np.array(valleys, dtype=int)

    # video frame 変換
    peaks_vid   = (peaks   + fp_start) + fp_to_video_offset
    valleys_vid = (valleys + fp_start) + fp_to_video_offset

    rows = []
    dt = 1.0 / fps
    for i, p in enumerate(peaks):
        v   = valleys[i - 1] if i > 0 and i - 1 < len(valleys) else None
        p_prev = peaks[i - 1] if i > 0 else max(0, p - int(fps * 0.8))
        dur = (p - p_prev) * dt

        if v is not None:
            impulse = float(np.sum(np.maximum(fz_filt[v: p + 1] - bw, 0)) * dt)
            rfd     = (fz_filt[p] - fz_filt[v]) / ((p - v) * dt) if p > v else np.nan
            v_fz    = float(fz_filt[v])
            v_vid   = int(valleys_vid[i - 1])
            unload  = v_fz / bw
        else:
            impulse = rfd = v_fz = unload = np.nan
            v_vid = np.nan

        rows.append({
            'rep_no'            : i + 1,
            'peak_video_frame'  : int(peaks_vid[i]),
            'peak_fz_N'         : float(fz_filt[p]),
            'valley_video_frame': v_vid,
            'valley_fz_N'       : v_fz,
            'rep_duration_fp_s' : dur,
            'impulse_Ns'        : impulse,
            'rfd_N_per_s'       : rfd,
            'unload_ratio'      : unload,
        })

    return pd.DataFrame(rows), bw


# ============================================================
# 2. per-rep 特徴量の構築
# ============================================================

def build_rep_features(label: str,
                        df_res: pd.DataFrame,
                        df_ff: pd.DataFrame,
                        df_fp_reps: pd.DataFrame,
                        bw: float) -> list[dict]:
    """
    results CSV の各 rep に対して特徴量を付与する。

    特徴量一覧:
        duration_sec    : results CSV のレップ所要時間 [s]
        relative_depth  : 最大CoM降下量に対する相対値 [%]  ← 主要特徴量
        peak_ff_N       : 床反力ピーク [N]  (FloorForce_y から)
        mean_ff_N       : 床反力平均 [N]
        std_ff_N        : 床反力標準偏差 [N]
        valley_ff_N     : 床反力谷値 [N]
        impulse_Ns      : 立ち上がり相の力積 [N·s]
        rfd_N_per_s     : Rate of Force Development [N/s]
        unload_ratio    : 荷重抜け率 = valley_fz / body_weight
    """
    ff_y = df_ff['Floor Force(only y)_y'].values
    rows = []

    for _, row in df_res.iterrows():
        sf  = int(row['start_frame'])
        tf  = int(row['target_frame'])
        ef  = int(row['end_frame'])

        # FloorForce_y の区間統計
        sf_c = max(0, sf); ef_c = min(len(ff_y) - 1, ef)
        ff_seg = ff_y[sf_c: ef_c + 1]
        valid  = ff_seg[(ff_seg > 0) & (ff_seg < 5000)]

        peak_ff = float(valid.max())  if len(valid) > 5 else np.nan
        mean_ff = float(valid.mean()) if len(valid) > 5 else np.nan
        std_ff  = float(valid.std())  if len(valid) > 5 else np.nan
        val_ff  = float(valid.min())  if len(valid) > 5 else np.nan
        impulse = float(np.sum(np.maximum(valid - bw, 0)) / FPS) if len(valid) > 5 else np.nan

        # FP ピーク rep との対応 (target_frame に最も近い peak を使用)
        match = df_fp_reps.iloc[
            (df_fp_reps['peak_video_frame'] - tf).abs().argsort()[:1]
        ]
        if len(match) > 0 and abs(match['peak_video_frame'].values[0] - tf) < 60:
            fp_r   = match.iloc[0]
            rfd    = fp_r['rfd_N_per_s']
            unload = fp_r['unload_ratio']
        else:
            rfd = unload = np.nan

        rows.append({
            'squat'         : label,
            'rep'           : int(row['レップ_number']),
            'RPE'           : float(row['RPE']),
            'duration_sec'  : float(row['duration_sec']),
            'relative_depth': float(row['relative_value']),
            'peak_ff_N'     : peak_ff,
            'mean_ff_N'     : mean_ff,
            'std_ff_N'      : std_ff,
            'valley_ff_N'   : val_ff,
            'impulse_Ns'    : impulse,
            'rfd_N_s'       : rfd,
            'unload_ratio'  : unload,
        })

    return rows


# ============================================================
# 3. 相関分析
# ============================================================

FEATURE_COLS = [
    'duration_sec', 'relative_depth',
    'peak_ff_N', 'mean_ff_N', 'std_ff_N', 'valley_ff_N',
    'impulse_Ns', 'rfd_N_s', 'unload_ratio',
]

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """全特徴量と RPE の Pearson r・Spearman ρ を算出する"""
    rows = []
    for feat in FEATURE_COLS:
        sub = df[['RPE', feat]].dropna()
        if len(sub) < 5:
            continue
        pr, pp = pearsonr(sub['RPE'], sub[feat])
        sr, sp = spearmanr(sub['RPE'], sub[feat])
        rows.append({
            'feature'   : feat,
            'n'         : len(sub),
            'pearson_r' : pr,
            'pearson_p' : pp,
            'spearman_r': sr,
            'spearman_p': sp,
        })
    return (pd.DataFrame(rows)
              .sort_values('pearson_r', key=abs, ascending=False)
              .reset_index(drop=True))


# ============================================================
# 4. RPE 推定モデル (Ridge + LOO-CV)
# ============================================================

def build_rpe_model(df: pd.DataFrame,
                     feat_cols: list[str]) -> dict:
    """
    Ridge 回帰で RPE を推定し、Leave-One-Out CV で汎化性能を評価する。

    Returns:
        dict with keys: model, scaler, y_true, y_pred, rmse, r2, feature_cols
    """
    sub = df[['RPE'] + feat_cols].dropna()
    X   = StandardScaler().fit_transform(sub[feat_cols].values)
    y   = sub['RPE'].values

    scaler = StandardScaler().fit(sub[feat_cols].values)
    Xs     = scaler.transform(sub[feat_cols].values)
    model  = Ridge(alpha=RIDGE_ALPHA)
    model.fit(Xs, y)

    y_pred = cross_val_predict(model, Xs, y, cv=LeaveOneOut())
    rmse   = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    r2     = float(r2_score(y, y_pred))

    return {
        'model'       : model,
        'scaler'      : scaler,
        'y_true'      : y,
        'y_pred'      : y_pred,
        'squat_labels': sub['squat'].values if 'squat' in sub.columns else None,
        'rmse'        : rmse,
        'r2'          : r2,
        'feature_cols': feat_cols,
        'n'           : len(sub),
    }


# ============================================================
# 5. 可視化
# ============================================================

def sig_stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def plot_feature_analysis(df: pd.DataFrame,
                           df_corr: pd.DataFrame,
                           out_path: Path) -> None:
    """特徴量相関バーチャート + 上位6特徴量の散布図"""
    fig = plt.figure(figsize=(15, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)

    # ── 上段: 相関バーチャート ──────────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    bar_colors = ['#E24B4A' if r > 0 else '#378ADD'
                  for r in df_corr['pearson_r']]
    bars = ax_bar.barh(df_corr['feature'], df_corr['pearson_r'],
                        color=bar_colors, alpha=0.8)
    for bar, row in zip(bars, df_corr.itertuples()):
        s = sig_stars(row.pearson_p)
        x_off = 0.02 if row.pearson_r > 0 else -0.02
        ax_bar.text(row.pearson_r + x_off,
                    bar.get_y() + bar.get_height() / 2,
                    f'{row.pearson_r:+.3f}{s}', va='center', fontsize=9)
    ax_bar.axvline(0, color='k', lw=0.8)
    ax_bar.set_xlabel('Pearson r with RPE')
    ax_bar.set_title('Feature correlation with RPE  (* p<.05  ** p<.01  *** p<.001)',
                     fontsize=11)
    ax_bar.grid(axis='x', alpha=0.3)

    # ── 中・下段: 上位6特徴量の散布図 ────────────────────────
    top6 = df_corr.head(6)['feature'].tolist()
    for idx, feat in enumerate(top6):
        row_idx = 1 + idx // 4
        col_idx = idx % 4
        ax = fig.add_subplot(gs[row_idx, col_idx])

        for sq, grp in df.groupby('squat'):
            sub = grp[['RPE', feat]].dropna()
            ax.scatter(sub['RPE'], sub[feat], s=35, alpha=0.75,
                       color=COLORS[sq], label=sq)
            if len(sub) > 3:
                z = np.polyfit(sub['RPE'], sub[feat], 1)
                xr = np.linspace(sub['RPE'].min(), sub['RPE'].max(), 50)
                ax.plot(xr, np.polyval(z, xr), color=COLORS[sq],
                        lw=1.2, alpha=0.7, ls='--')

        cr = df_corr[df_corr['feature'] == feat]['pearson_r'].values[0]
        cp = df_corr[df_corr['feature'] == feat]['pearson_p'].values[0]
        ax.set_title(f'{feat}\nr={cr:+.3f}{sig_stars(cp)}', fontsize=9)
        ax.set_xlabel('RPE')
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle(
        f'RPE Estimation Features Analysis  '
        f'(squat02: n={len(df[df.squat=="squat02"])}, '
        f'squat03: n={len(df[df.squat=="squat03"])})',
        fontsize=12, y=1.01
    )
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  saved: {out_path.name}')


def plot_rpe_prediction(df: pd.DataFrame,
                         result1: dict,
                         out_path: Path) -> None:
    """LOO-CV 予測プロット + 時系列 + 散布図"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sub1 = df[['RPE', 'relative_depth', 'squat']].dropna()
    y1, y_pred1 = result1['y_true'], result1['y_pred']

    # 左: relative_depth vs RPE
    for sq, grp in sub1.groupby('squat'):
        axes[0].scatter(grp['relative_depth'], grp['RPE'],
                        s=55, alpha=0.8, color=COLORS[sq], label=sq, zorder=5)
        z = np.polyfit(grp['relative_depth'], grp['RPE'], 1)
        xr = np.linspace(grp['relative_depth'].min(),
                          grp['relative_depth'].max(), 50)
        axes[0].plot(xr, np.polyval(z, xr), color=COLORS[sq],
                     lw=1.5, ls='--')
    z_all = np.polyfit(sub1['relative_depth'], y1, 1)
    xr_all = np.linspace(sub1['relative_depth'].min(),
                          sub1['relative_depth'].max(), 100)
    axes[0].plot(xr_all, np.polyval(z_all, xr_all), 'k-', lw=2, label='Overall fit')
    r_all, p_all = pearsonr(y1, sub1['relative_depth'].values)
    axes[0].set_xlabel('Relative depth [% of max CoM drop]')
    axes[0].set_ylabel('RPE')
    axes[0].set_title(f'Relative depth vs RPE\nr={r_all:.4f} (p<.001)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # 中: 時系列
    for sq, grp in df.groupby('squat'):
        grp = grp.reset_index(drop=True)
        axes[1].plot(grp['rep'], grp['RPE'], 'o-',
                     color=COLORS[sq], lw=1.5, ms=5, label=f'{sq} RPE')
        ax2 = axes[1].twinx()
        ax2.plot(grp['rep'], grp['relative_depth'], 's--',
                 color=COLORS[sq], lw=1.0, ms=4, alpha=0.5)
        ax2.set_ylabel('Relative depth [%]', fontsize=9)
    axes[1].set_xlabel('Rep #')
    axes[1].set_ylabel('RPE')
    axes[1].set_title('RPE (line) & relative depth (dashed) per rep')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 右: LOO-CV 予測 vs 実測
    for sq, grp in sub1.groupby('squat'):
        idx = list(range(*sub1.index.slice_locs(
            sub1[sub1['squat'] == sq].index[0],
            sub1[sub1['squat'] == sq].index[-1] + 1
        )))
        mask = sub1['squat'].values == sq
        axes[2].scatter(y1[mask], y_pred1[mask],
                        color=COLORS[sq], s=55, alpha=0.8, label=sq, zorder=5)
    lims = [0.5, 10.5]
    axes[2].plot(lims, lims, 'k--', lw=1.0, label='y=x')
    axes[2].set_xlabel('Actual RPE')
    axes[2].set_ylabel('Predicted RPE (LOO-CV)')
    axes[2].set_title(
        f'LOO-CV prediction  (relative_depth)\n'
        f'RMSE={result1["rmse"]:.3f}  R²={result1["r2"]:.3f}'
    )
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(lims)
    axes[2].set_ylim(lims)

    plt.suptitle(
        f'RPE Estimation from Relative COM Depth  (n={result1["n"]} reps)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'  saved: {out_path.name}')


# ============================================================
# 6. メイン処理
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for label in ['squat02', 'squat03']:
        fp_suffix = label[-1]   # '2' or '3'
        print(f'\n[{label}] 処理中...')

        # --- データ読み込み ---
        df_res  = pd.read_csv(DATA_DIR / f'{label}_results.csv')
        df_res.columns = (df_res.columns
                          .str.strip()
                          .str.replace('\ufeff', '', regex=False))

        df_ff   = pd.read_csv(DATA_DIR / f'{label}_correct_modelbase.csv')
        fp_df   = load_fp(DATA_DIR / f'squat_{fp_suffix}_30Hz.csv')

        # --- FP ピーク検出 ---
        offset    = SYNC[label]['com_start'] - SYNC[label]['lag']
        df_fp_reps, bw = detect_reps_from_fp(fp_df, fp_to_video_offset=offset)
        print(f'  body_weight = {bw:.1f} N,  reps detected = {len(df_fp_reps)}')

        # --- per-rep 特徴量構築 ---
        rows = build_rep_features(label, df_res, df_ff, df_fp_reps, bw)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # --- 相関分析 ---
    print('\n[相関分析]')
    df_corr = compute_correlations(df)
    print(df_corr[['feature', 'n', 'pearson_r', 'pearson_p',
                    'spearman_r', 'spearman_p']].to_string(index=False))

    # --- RPE 推定モデル ---
    print('\n[RPE 推定モデル]')
    # モデル1: relative_depth 単体
    result1 = build_rpe_model(df, ['relative_depth'])
    print(f'  single feature (relative_depth): '
          f'LOO-CV RMSE={result1["rmse"]:.3f}  R²={result1["r2"]:.3f}  n={result1["n"]}')

    # モデル2: 多特徴量
    result2 = build_rpe_model(df, ['relative_depth', 'duration_sec',
                                    'unload_ratio', 'rfd_N_s'])
    print(f'  multi-feature:                  '
          f'LOO-CV RMSE={result2["rmse"]:.3f}  R²={result2["r2"]:.3f}  n={result2["n"]}')

    # squat 別相関
    print('\n[squat 別 relative_depth → RPE 相関]')
    for sq, grp in df.groupby('squat'):
        sub = grp[['RPE', 'relative_depth']].dropna()
        r, p = pearsonr(sub['RPE'], sub['relative_depth'])
        sr, sp = spearmanr(sub['RPE'], sub['relative_depth'])
        print(f'  {sq}: n={len(sub)}  '
              f'Pearson r={r:.4f}(p={p:.4f})  '
              f'Spearman ρ={sr:.4f}(p={sp:.4f})')

    # --- 出力 ---
    print('\n[ファイル出力]')
    df.to_csv(OUT_DIR / 'rpe_features.csv', index=False)
    print(f'  saved: rpe_features.csv  (n={len(df)} reps)')

    df_corr.to_csv(OUT_DIR / 'rpe_correlation.csv', index=False)
    print(f'  saved: rpe_correlation.csv')

    plot_feature_analysis(df, df_corr,    OUT_DIR / 'rpe_feature_analysis.png')
    plot_rpe_prediction  (df, result1,    OUT_DIR / 'rpe_prediction.png')


if __name__ == '__main__':
    main()