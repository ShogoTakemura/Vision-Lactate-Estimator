"""
hr_rep_analysis.py
──────────────────────────────────────────────────────────────────────
セット全体で rPPG 信号を推定し、レップ時間帯でスライスして
レップ単位の心拍数（HR）および RRI を算出するスクリプト。

【処理フロー】
  1. セット全体の RGB CSV を読み込み → 欠損補間
  2. POS 法で rPPG 生信号を生成（セット全体）
  3. バターワース BPF (0.8–2.0 Hz) で整形（セット全体）
  4. セット全体でピーク検出（プロミネンス閾値処理）
  5. レップ CSV の start_frame/end_frame でピークをスライス
  6. スライス区間内の RRI から HR 算出（不足時は Welch で補完）
  7. 結果を CSV 保存 ＋ グラフ描画

【入力ファイル構成】
  <SET_DIR>/
    ├── <prefix>set1_rgb.csv   # mediapipe_roi_face.py の出力
    ├── <prefix>set1_rep.csv   # rep 区間 CSV (rep, start_frame, bottom_frame, end_frame)
    ├── <prefix>set2_rgb.csv
    ├── <prefix>set2_rep.csv
    └── ...

【出力ファイル】
  <OUT_DIR>/
    ├── rep_hr_result.csv      # レップ単位 HR / RRI テーブル
    └── rep_hr_analysis.png    # 分析グラフ（5パネル）

【使い方】
  # ① スクリプト末尾の INPUT_DIR / OUT_DIR / PREFIX / N_SETS を編集して実行
  python hr_rep_analysis.py

  # ② コマンドライン引数で指定
  python hr_rep_analysis.py \\
      --input_dir  C:/data/20250121_uchino \\
      --out_dir    C:/data/20250121_uchino/results \\
      --prefix     20250121_uchino-RIR1-nonmax- \\
      --n_sets     3 \\
      --no_plot        # グラフ非表示（CSV のみ出力）
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # 画面なし環境でも動作（表示したい場合は削除）
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, welch, find_peaks

# ══════════════════════════════════════════════════════════════════════
# ★ 実行設定（コマンドライン引数を使わない場合はここを直接編集）
# ══════════════════════════════════════════════════════════════════════
INPUT_DIR = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\roi_out\20241119"
OUT_DIR   = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\reps\processed"
PREFIX    = "20250121_uchino-RIR1-nonmax-"   # set番号の直前までのファイル名共通部分
N_SETS    = 3                                 # 処理するセット数
SHOW_PLOT = True                              # グラフをウィンドウ表示するか
# ══════════════════════════════════════════════════════════════════════

# ─── 論文準拠パラメータ（市川 2025, 3.2章）──────────────────────────
HR_LOW          = 0.8    # BPF 下限 [Hz]
HR_HIGH         = 2.0    # BPF 上限 [Hz]
BPF_ORDER       = 4      # バターワースフィルタ次数
POS_WIN_SEC     = 1.6    # POS 法の時間窓 [s]
PEAK_PROM_RATIO = 1/10   # プロミネンス閾値（最大値の 1/10）
RRI_TOL         = 0.3    # RRI 外れ値除去幅 [s]
WELCH_NPSEG     = 30 * 1024
WELCH_NOVERLAP  = 10 * 1024
# ─────────────────────────────────────────────────────────────────────

SET_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF5722",
              4: "#9C27B0", 5: "#FF9800"}   # 最大 5 セット分


# ══════════════════════════════════════════════════════════════════════
# rPPG 推定モジュール
# ══════════════════════════════════════════════════════════════════════

def pos_rppg(rgb: np.ndarray, fps: float) -> np.ndarray:
    """
    POS 法（Wang et al. 2017）で RGB 時系列から rPPG 信号を生成する。
    論文 式(3-7)〜(3-9) 準拠。

    Parameters
    ----------
    rgb : (N, 3) [R, G, B]
    fps : フレームレート [Hz]

    Returns
    -------
    h : (N,) rPPG 信号
    """
    N   = len(rgb)
    win = max(2, int(POS_WIN_SEC * fps))
    h   = np.zeros(N)
    cnt = np.zeros(N, dtype=int)

    for s in range(0, N - win + 1):
        seg  = rgb[s : s + win].copy()
        mean = seg.mean(axis=0)
        if np.any(mean == 0):
            continue
        xn = seg / mean                              # 正規化 RGB
        S1 = xn[:, 0] - xn[:, 1]                    # 式(3-7)
        S2 = -2 * xn[:, 0] + xn[:, 1] + xn[:, 2]   # 式(3-8)
        std2 = np.std(S2)
        if std2 == 0:
            continue
        alpha = np.std(S1) / std2                    # 式(3-9)
        h_seg = S1 + alpha * S2
        h[s : s + win] += h_seg
        cnt[s : s + win] += 1

    valid      = cnt > 0
    h[valid]  /= cnt[valid]
    return h


def bandpass_filter(sig: np.ndarray, fps: float) -> np.ndarray:
    """バターワース BPF (0.8–2.0 Hz)。論文 Table 3-4 準拠。"""
    nyq  = fps / 2.0
    low  = np.clip(HR_LOW  / nyq, 1e-4, 0.999)
    high = np.clip(HR_HIGH / nyq, 1e-4, 0.999)
    b, a = butter(BPF_ORDER, [low, high], btype="band")
    return filtfilt(b, a, sig)


def welch_peak_freq(sig: np.ndarray, fps: float) -> float:
    """Welch 法でピーク周波数を推定する。論文 p.24 準拠。"""
    nperseg  = min(len(sig), WELCH_NPSEG)
    noverlap = min(len(sig) - 1, min(nperseg - 1, WELCH_NOVERLAP))
    freqs, psd = welch(sig, fs=fps, window="hamming",
                       nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= HR_LOW) & (freqs <= HR_HIGH)
    if not np.any(mask):
        return 0.0
    psd_hr = psd.copy()
    psd_hr[~mask] = 0
    return float(freqs[np.argmax(psd_hr)])


def detect_peaks_global(sig: np.ndarray) -> np.ndarray:
    """
    プロミネンス閾値処理でピーク検出する（セット全体信号用）。
    論文 p.23 準拠。

    Returns
    -------
    peaks : ピークのサンプルインデックス配列
    """
    peaks, props = find_peaks(sig, prominence=0)
    proms = props["prominences"]
    if len(proms) == 0:
        return np.array([], dtype=int)
    thresh = proms.max() * PEAK_PROM_RATIO
    return peaks[proms >= thresh]


def filter_rri(rri: np.ndarray, ref: float) -> np.ndarray:
    """
    基準 RRI ± RRI_TOL の範囲外を除去する。
    除去数が半数超なら除去しない（論文 p.24）。
    """
    mask = np.abs(rri - ref) <= RRI_TOL
    return rri if mask.sum() < len(rri) / 2 else rri[mask]


def hr_from_rep_peaks(rep_peak_frames: np.ndarray,
                      fps: float,
                      pf_ref: float) -> dict:
    """
    レップ内ピーク列から HR / RRI を算出する。

    Parameters
    ----------
    rep_peak_frames : レップ内ピークのフレームインデックス（セット先頭基準）
    fps             : フレームレート
    pf_ref          : 参照ピーク周波数（外れ値除去の基準に使用）

    Returns
    -------
    dict : hr_bpm, hr_rri, hr_welch, rri_mean_ms, rri_count, method
    """
    result = dict(hr_bpm=np.nan, hr_rri=np.nan, hr_welch=np.nan,
                  rri_mean_ms=np.nan, rri_count=0, method="none")

    if len(rep_peak_frames) >= 2:
        pt  = rep_peak_frames / fps
        rri = np.diff(pt)

        # 外れ値除去（1回目: Welch 基準、2回目: RRI 平均基準）
        ref = 1.0 / pf_ref if pf_ref > 0 else np.median(rri)
        rri = filter_rri(rri, ref)
        if len(rri) > 0:
            rri = filter_rri(rri, rri.mean())

        if len(rri) >= 1:
            hr_rri = 60.0 / rri.mean()
            result.update(
                hr_rri      = round(hr_rri, 1),
                hr_bpm      = round(hr_rri, 1),
                rri_mean_ms = round(rri.mean() * 1000, 1),
                rri_count   = len(rri),
                method      = "RRI",
            )

    return result


# ══════════════════════════════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════════════════════════════

def process_set(set_num: int, input_dir: str, prefix: str) -> tuple[pd.DataFrame, dict]:
    """
    1 セット分の処理を行う。

    Returns
    -------
    result_df : レップ単位の結果 DataFrame
    ctx       : 可視化用の信号データ辞書
    """
    rgb_path = os.path.join(input_dir, f"{prefix}set{set_num}_rgb.csv")
    rep_path = os.path.join(input_dir, f"{prefix}set{set_num}_rep.csv")

    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB CSV が見つかりません: {rgb_path}")
    if not os.path.exists(rep_path):
        raise FileNotFoundError(f"Rep CSV が見つかりません: {rep_path}")

    rgb_df = pd.read_csv(rgb_path)
    rep_df = pd.read_csv(rep_path)

    # fps 推定
    fps = 1.0 / rgb_df["Time(s)"].diff().median()

    # 欠損フレーム補間（ランドマーク未検出行）
    rgb_df[["R_mean", "G_mean", "B_mean"]] = (
        rgb_df[["R_mean", "G_mean", "B_mean"]]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    rgb_all = rgb_df[["R_mean", "G_mean", "B_mean"]].values.astype(float)

    # ── セット全体で rPPG → BPF → ピーク検出 ──────────────────
    print(f"  [Set{set_num}] POS 法で rPPG 計算中... (fps={fps:.2f}, frames={len(rgb_all)})")
    h_raw        = pos_rppg(rgb_all, fps)
    h_filt       = bandpass_filter(h_raw, fps)
    pf_global    = welch_peak_freq(h_filt, fps)
    all_peaks    = detect_peaks_global(h_filt)   # フレーム行インデックス

    print(f"  [Set{set_num}] Welch pf={pf_global:.3f} Hz ({pf_global*60:.1f} bpm), "
          f"total peaks={len(all_peaks)}, reps={len(rep_df)}")

    frame_arr = rgb_df["Frame"].values
    rows      = []

    for _, row in rep_df.iterrows():
        rep_n = int(row["rep"])
        sf    = int(row["start_frame"])
        bf    = int(row["bottom_frame"])
        ef    = int(row["end_frame"])

        # フレーム番号 → 行インデックスに変換
        idx_s = int(np.searchsorted(frame_arr, sf))
        idx_e = int(np.searchsorted(frame_arr, ef, side="right"))

        # レップ内に含まれるピーク（行インデックス）
        rep_peaks = all_peaks[(all_peaks >= idx_s) & (all_peaks < idx_e)]

        # レップ区間の Welch（短すぎる場合はセット全体の値を流用）
        seg_filt = h_filt[idx_s:idx_e]
        pf_rep   = welch_peak_freq(seg_filt, fps) if len(seg_filt) > 60 else pf_global

        rep_time = (ef - sf) / fps
        hr_info  = hr_from_rep_peaks(rep_peaks, fps, pf_ref=pf_rep)

        # HR が RRI から取れなかった場合は Welch で補完
        hr_welch = pf_rep * 60.0 if pf_rep > 0 else np.nan
        if np.isnan(hr_info["hr_bpm"]) and not np.isnan(hr_welch):
            hr_info["hr_bpm"]  = round(hr_welch, 1)
            hr_info["method"]  = "Welch"
        hr_info["hr_welch"] = round(hr_welch, 1) if not np.isnan(hr_welch) else np.nan

        rows.append({
            "set":          set_num,
            "rep":          rep_n,
            "rep_time_s":   round(rep_time, 2),
            "n_frames":     idx_e - idx_s,
            "peak_count":   len(rep_peaks),
            "hr_bpm":       hr_info["hr_bpm"],
            "hr_rri":       hr_info["hr_rri"],
            "hr_welch":     hr_info["hr_welch"],
            "rri_mean_ms":  hr_info["rri_mean_ms"],
            "rri_count":    hr_info["rri_count"],
            "method":       hr_info["method"],
        })

    result_df = pd.DataFrame(rows)

    ctx = {
        "fps":        fps,
        "h_filt":     h_filt,
        "all_peaks":  all_peaks,
        "frame_arr":  frame_arr,
        "rep_df":     rep_df,
        "pf_global":  pf_global,
    }
    return result_df, ctx


def run(input_dir: str, out_dir: str, prefix: str,
        n_sets: int, show_plot: bool = True, no_plot: bool = False):
    """全セットを処理して CSV 保存 ＋ グラフ出力。"""
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    all_ctx     = {}

    for s in range(1, n_sets + 1):
        print(f"\n=== Set {s} 処理開始 ===")
        try:
            df_s, ctx = process_set(s, input_dir, prefix)
            all_results.append(df_s)
            all_ctx[s] = ctx
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

    if not all_results:
        print("処理できるセットがありませんでした。")
        return

    result_df = pd.concat(all_results, ignore_index=True)

    # ── CSV 保存 ──────────────────────────────────────────────
    csv_out = os.path.join(out_dir, "rep_hr_result.csv")
    result_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n[保存] {csv_out}")
    print(result_df.to_string(index=False))

    # ── グラフ ────────────────────────────────────────────────
    if no_plot:
        return

    sets = sorted(result_df["set"].unique())
    fig  = plt.figure(figsize=(16, 14))
    gs   = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.36)

    def color(s):
        return SET_COLORS.get(s, "#607D8B")

    # (1) HR per rep
    ax1 = fig.add_subplot(gs[0, :])
    for s in sets:
        d = result_df[result_df["set"] == s]
        n = len(d)
        ax1.plot(d["rep"], d["hr_bpm"], "o-", color=color(s),
                 label=f"Set {s}  ({n} reps)", linewidth=1.8, markersize=5)
        # Welch 補完レップを × で強調
        welch_only = d[d["method"] == "Welch"]
        ax1.scatter(welch_only["rep"], welch_only["hr_bpm"],
                    marker="x", color=color(s), s=90, zorder=5, linewidths=2)
    ax1.set_xlabel("Rep number", fontsize=11)
    ax1.set_ylabel("HR [bpm]", fontsize=11)
    ax1.set_title("(1) HR per Rep  [set-level rPPG -> rep slice]"
                  "   x = Welch fallback", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_xticks(range(1, result_df["rep"].max() + 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 135)

    # (2) RRI mean per rep
    ax2 = fig.add_subplot(gs[1, 0])
    for s in sets:
        d = result_df[result_df["set"] == s].dropna(subset=["rri_mean_ms"])
        ax2.plot(d["rep"], d["rri_mean_ms"], "o-", color=color(s),
                 label=f"Set {s}", linewidth=1.8, markersize=5)
    ax2.set_xlabel("Rep number")
    ax2.set_ylabel("RRI mean [ms]")
    ax2.set_title("(2) RRI mean per Rep")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (3) Rep duration
    ax3 = fig.add_subplot(gs[1, 1])
    for s in sets:
        d = result_df[result_df["set"] == s]
        ax3.plot(d["rep"], d["rep_time_s"], "o-", color=color(s),
                 label=f"Set {s}", linewidth=1.8, markersize=5)
    ax3.set_xlabel("Rep number")
    ax3.set_ylabel("Duration [s]")
    ax3.set_title("(3) Rep Duration")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (4) Peak count per rep（棒グラフ）
    ax4 = fig.add_subplot(gs[2, 0])
    x_all     = np.arange(len(result_df))
    bar_colors = [color(s) for s in result_df["set"]]
    ax4.bar(x_all, result_df["peak_count"], color=bar_colors, alpha=0.75, edgecolor="white")
    offset = 0
    for s in sets:
        n = len(result_df[result_df["set"] == s])
        if offset > 0:
            ax4.axvline(offset - 0.5, color="gray", linestyle=":", linewidth=1.2)
        ax4.text(offset + n / 2 - 0.5,
                 result_df["peak_count"].max() + 0.3,
                 f"Set{s}", ha="center", fontsize=8, color=color(s))
        offset += n
    ax4.axhline(2, color="red", linestyle="--", linewidth=1, label="min 2 peaks")
    ax4.set_ylabel("Detected peaks")
    ax4.set_xlabel("Rep (sequential)")
    ax4.set_title("(4) Peak Count per Rep")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    # (5) HR distribution boxplot per set
    ax5 = fig.add_subplot(gs[2, 1])
    data_box = [result_df[result_df["set"] == s]["hr_bpm"].dropna().values
                for s in sets]
    bp = ax5.boxplot(data_box, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, s in zip(bp["boxes"], sets):
        patch.set_facecolor(color(s))
        patch.set_alpha(0.7)
    ax5.set_xticklabels([f"Set {s}\n({len(result_df[result_df['set']==s])} reps)"
                         for s in sets])
    ax5.set_ylabel("HR [bpm]")
    ax5.set_title("(5) HR Distribution by Set")
    ax5.grid(True, alpha=0.3, axis="y")
    for i, (d, s) in enumerate(zip(data_box, sets), 1):
        if len(d) > 0:
            ax5.text(i, d.max() + 2,
                     f"mean={d.mean():.0f}\nsd={d.std():.0f}",
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle("rPPG Heart Rate Analysis — Set-level Signal -> Rep Slice\n"
                 f"prefix: {prefix}",
                 fontsize=13, fontweight="bold")

    png_out = os.path.join(out_dir, "rep_hr_analysis.png")
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    print(f"[保存] {png_out}")

    if show_plot:
        matplotlib.use("TkAgg")   # ウィンドウ表示（環境によって変更）
        plt.show()


# ══════════════════════════════════════════════════════════════════════
# エントリーポイント
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="rPPG レップ単位心拍推定（セット全体→スライス方式）"
    )
    parser.add_argument("--input_dir", default=None,
                        help="RGB/rep CSV が置かれたフォルダ")
    parser.add_argument("--out_dir",   default=None,
                        help="結果の出力先フォルダ")
    parser.add_argument("--prefix",    default=None,
                        help="ファイル名の共通プレフィックス（例: 20250121_uchino-RIR1-nonmax-）")
    parser.add_argument("--n_sets",    default=None, type=int,
                        help="処理するセット数")
    parser.add_argument("--no_plot",   action="store_true",
                        help="グラフを表示せず CSV のみ出力")
    args = parser.parse_args()

    # コマンドライン引数が指定された場合はそちらを優先
    input_dir = args.input_dir if args.input_dir else INPUT_DIR
    out_dir   = args.out_dir   if args.out_dir   else OUT_DIR
    prefix    = args.prefix    if args.prefix    else PREFIX
    n_sets    = args.n_sets    if args.n_sets    else N_SETS
    show_plot = SHOW_PLOT and not args.no_plot

    run(input_dir  = input_dir,
        out_dir    = out_dir,
        prefix     = prefix,
        n_sets     = n_sets,
        show_plot  = show_plot,
        no_plot    = args.no_plot)
