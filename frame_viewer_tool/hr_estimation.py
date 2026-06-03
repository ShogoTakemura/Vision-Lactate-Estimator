"""
心拍推定スクリプト（市川太陽 卒業論文 3.2章 準拠）

処理フロー:
  1. ROI から RGB 信号取得（mediapipe_roi.py で生成した CSV を使用）
  2. 肌色マスク（HSV + YCbCr 閾値）  ※ CSVには既に肌色ROIの平均値が入っているため省略可
  3. POS 法（Wang ら）で rPPG 信号を合成
  4. バターワースBPF（0.8–2.0 Hz）で整形
  5. Welch 法でピーク周波数→心拍数(bpm)算出
  6. プロミネンス閾値処理で RRI 検出
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os

# ─────────────────────────────────────────────
# 定数（論文 Table 3-4 / 3.2章 準拠）
# ─────────────────────────────────────────────
HR_BPF_LOW  = 0.8   # Hz
HR_BPF_HIGH = 2.0   # Hz
POS_WINDOW_SEC = 1.6  # 秒（論文 p.22）
WELCH_NPerseg_FACTOR = 1024  # FFT 長さ = 30×1024（論文 p.24）
WELCH_NOVERLAP_FACTOR = 1024 # オーバーラップ = 10×1024
PEAK_PROM_RATIO = 1 / 10     # プロミネンス閾値（最大値の 1/10）
RRI_TOLERANCE = 0.3           # RRI 外れ値除去幅 [s]（論文 p.24）


# ─────────────────────────────────────────────
# 1. CSV 読み込み
# ─────────────────────────────────────────────
def load_rgb_csv(csv_path: str) -> tuple[np.ndarray, float]:
    """
    mediapipe_roi.py が出力する CSV（Frame, Time(s), R_mean, G_mean, B_mean）を読み込む。
    None が含まれる行（ランドマーク未検出フレーム）は線形補間する。

    Returns
    -------
    rgb : np.ndarray  shape=(N, 3)  [R, G, B]
    fps : float       フレームレート
    """
    df = pd.read_csv(csv_path)
    required = {"Frame", "Time(s)", "R_mean", "G_mean", "B_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV に必要な列がありません。必要: {required}")

    # 欠損補間
    df[["R_mean", "G_mean", "B_mean"]] = (
        df[["R_mean", "G_mean", "B_mean"]]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )
    df = df.dropna(subset=["R_mean", "G_mean", "B_mean"])

    rgb = df[["R_mean", "G_mean", "B_mean"]].values.astype(float)

    # fps: Time 列から推定
    times = df["Time(s)"].values
    if len(times) > 1:
        fps = 1.0 / np.median(np.diff(times))
    else:
        fps = 30.0
        print(f"[警告] フレーム数が少ないため fps={fps} と仮定します。")

    print(f"[INFO] フレーム数: {len(rgb)}, fps: {fps:.2f} Hz, 計測時間: {times[-1]:.1f} s")
    return rgb, fps


# ─────────────────────────────────────────────
# 2. POS 法（Wang et al. 2017）
#    論文 式 3-7 〜 3-9 準拠
# ─────────────────────────────────────────────
def pos_rppg(rgb: np.ndarray, fps: float, window_sec: float = POS_WINDOW_SEC) -> np.ndarray:
    """
    POS 法で RGB 時系列信号から rPPG 信号を算出する。

    Parameters
    ----------
    rgb        : (N, 3)  [R, G, B] 各フレームの平均画素値
    fps        : フレームレート [Hz]
    window_sec : 時間窓長 [s]（論文では 1.6 s）

    Returns
    -------
    h : (N,) rPPG 信号（オーバーラップ加算済み）
    """
    N = len(rgb)
    win = max(2, int(window_sec * fps))   # 窓サイズ [frames]

    h = np.zeros(N)
    cnt = np.zeros(N, dtype=int)

    for start in range(0, N - win + 1):
        end = start + win
        segment = rgb[start:end].copy()   # (win, 3)

        # --- 正規化 RGB（平均で割る）---
        mean_rgb = segment.mean(axis=0)   # (3,)
        if np.any(mean_rgb == 0):
            continue
        xn = segment / mean_rgb           # 式 3-7 準拠（各成分を平均で割る）

        xR, xG, xB = xn[:, 0], xn[:, 1], xn[:, 2]

        # --- POS 平面への投影（式 3-7, 3-8）---
        S1 = xR - xG                      # 式 3-7
        S2 = -2 * xR + xG + xB           # 式 3-8

        # --- Alpha-tuning（式 3-9）---
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        if std_S2 == 0:
            continue
        alpha = std_S1 / std_S2           # 式 3-9

        # --- 1 次元脈波信号 h(t) ---
        h_seg = S1 + alpha * S2           # 式 3-9

        # オーバーラップ加算
        h[start:end] += h_seg
        cnt[start:end] += 1

    # 平均化（加算回数で割る）
    valid = cnt > 0
    h[valid] /= cnt[valid]

    return h


# ─────────────────────────────────────────────
# 3. バターワース BPF
#    論文 Table 3-4: 心拍 0.8–2.0 Hz
# ─────────────────────────────────────────────
def bandpass_filter(signal: np.ndarray, fps: float,
                    low: float = HR_BPF_LOW, high: float = HR_BPF_HIGH,
                    order: int = 4) -> np.ndarray:
    nyq = fps / 2.0
    low_n  = low  / nyq
    high_n = high / nyq
    # Nyquist を超えないようクリップ
    low_n  = np.clip(low_n,  1e-4, 0.999)
    high_n = np.clip(high_n, 1e-4, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────
# 4. Welch 法でピーク周波数算出
#    論文 p.24: nperseg=30×1024, noverlap=10×1024, 窓=ハミング
# ─────────────────────────────────────────────
def welch_peak_freq(signal: np.ndarray, fps: float,
                    f_low: float = HR_BPF_LOW,
                    f_high: float = HR_BPF_HIGH) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    peak_freq : ピーク周波数 [Hz]
    freqs     : 周波数軸
    psd       : パワースペクトル密度
    """
    nperseg  = min(len(signal), 30 * WELCH_NPerseg_FACTOR)
    noverlap = min(len(signal) - 1, 10 * WELCH_NOVERLAP_FACTOR)
    noverlap = min(noverlap, nperseg - 1)

    freqs, psd = welch(signal, fs=fps, window="hamming",
                       nperseg=nperseg, noverlap=noverlap)

    # 心拍帯域のみでピーク検索
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0, freqs, psd

    psd_hr = psd.copy()
    psd_hr[~mask] = 0
    peak_idx = np.argmax(psd_hr)
    peak_freq = freqs[peak_idx]
    return peak_freq, freqs, psd


# ─────────────────────────────────────────────
# 5. RRI 検出（プロミネンス閾値 + 外れ値除去）
#    論文 p.23–24 準拠
# ─────────────────────────────────────────────
def detect_rri(signal: np.ndarray, fps: float,
               peak_freq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    プロミネンス閾値処理でピーク検出し、外れ値 RRI を除去する。

    Returns
    -------
    rri_clean   : クリーニング後の RRI 配列 [s]
    peak_times  : ピーク時刻 [s]
    """
    # --- ピーク検出 ---
    peaks, props = find_peaks(signal, prominence=0)
    prominences = props["prominences"]

    if len(prominences) == 0:
        return np.array([]), np.array([])

    # 閾値 = 最大プロミネンスの 1/10（論文 p.23）
    prom_thresh = prominences.max() * PEAK_PROM_RATIO
    valid_mask  = prominences >= prom_thresh
    peaks = peaks[valid_mask]

    if len(peaks) < 2:
        return np.array([]), peaks / fps

    # --- RRI 算出 ---
    peak_times = peaks / fps
    rri = np.diff(peak_times)

    # --- 外れ値除去（論文 式: RRI_freq ± 0.3 s）---
    if peak_freq > 0:
        rri_ref = 1.0 / peak_freq
    else:
        rri_ref = np.median(rri)

    def _filter_rri(rri_arr, ref):
        mask = np.abs(rri_arr - ref) <= RRI_TOLERANCE
        # 除去数が半数超えなら除去しない（論文 p.24）
        if mask.sum() < len(rri_arr) / 2:
            return rri_arr
        return rri_arr[mask]

    rri_clean = _filter_rri(rri, rri_ref)

    # 2 回目: 平均値で再フィルタ（論文 p.24）
    if len(rri_clean) > 0:
        rri_mean2 = rri_clean.mean()
        rri_clean = _filter_rri(rri_clean, rri_mean2)

    return rri_clean, peak_times


# ─────────────────────────────────────────────
# 6. 心拍数算出
# ─────────────────────────────────────────────
def calc_heart_rate(rri_clean: np.ndarray, peak_freq: float) -> dict:
    """
    RRI から HR を算出する。
    RRI が少ない場合は周波数解析結果（Welch）を優先する。
    """
    hr_rri    = None
    hr_welch  = peak_freq * 60.0 if peak_freq > 0 else None

    if len(rri_clean) >= 2:
        hr_rri = 60.0 / rri_clean.mean()

    # 優先順位: RRI 平均 > Welch
    hr_final = hr_rri if hr_rri is not None else hr_welch

    return {
        "hr_bpm_rri"   : hr_rri,
        "hr_bpm_welch" : hr_welch,
        "hr_bpm"       : hr_final,
        "rri_mean_s"   : rri_clean.mean() if len(rri_clean) >= 2 else None,
        "rri_count"    : len(rri_clean),
    }


# ─────────────────────────────────────────────
# 7. 可視化
# ─────────────────────────────────────────────
def plot_results(rgb: np.ndarray, h_raw: np.ndarray, h_filt: np.ndarray,
                 peak_times: np.ndarray, freqs: np.ndarray, psd: np.ndarray,
                 fps: float, result: dict, save_path: str = None):

    times = np.arange(len(rgb)) / fps

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── (a) RGB 時系列 ──
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(times, rgb[:, 0], "r", alpha=0.7, label="R", linewidth=0.8)
    ax0.plot(times, rgb[:, 1], "g", alpha=0.7, label="G", linewidth=0.8)
    ax0.plot(times, rgb[:, 2], "b", alpha=0.7, label="B", linewidth=0.8)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Pixel intensity")
    ax0.set_title("(a) ROI RGB time series")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    # ── (b) rPPG 生信号 ──
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(times, h_raw, color="purple", linewidth=0.8)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("(b) rPPG signal (POS, raw)")
    ax1.grid(True, alpha=0.3)

    # ── (c) BPF 後 rPPG + ピーク ──
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(times, h_filt, color="darkcyan", linewidth=0.9, label="Filtered rPPG")
    # ピーク位置をプロット
    if len(peak_times) > 0:
        peak_frames = (peak_times * fps).astype(int)
        peak_frames = peak_frames[peak_frames < len(h_filt)]
        ax2.scatter(peak_times[:len(peak_frames)], h_filt[peak_frames],
                    color="red", zorder=5, s=25, label="Peaks")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("(c) Filtered rPPG + detected peaks")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── (d) Welch PSD ──
    ax3 = fig.add_subplot(gs[2, 0])
    mask = (freqs >= HR_BPF_LOW) & (freqs <= HR_BPF_HIGH)
    ax3.plot(freqs[mask], psd[mask], color="navy", linewidth=1.0)
    if result["hr_bpm_welch"]:
        pf = result["hr_bpm_welch"] / 60.0
        ax3.axvline(pf, color="red", linestyle="--", linewidth=1.2,
                    label=f"Peak: {pf:.3f} Hz\n({result['hr_bpm_welch']:.1f} bpm)")
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("PSD")
    ax3.set_title("(d) Welch PSD (HR band)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── (e) 結果サマリ ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    lines = [
        "── Estimation Results ──",
        f"HR (Mean RRI)  : {result['hr_bpm_rri']:.1f} bpm" if result['hr_bpm_rri'] else "HR (Mean RRI)  : N/A",
        f"HR (Welch)     : {result['hr_bpm_welch']:.1f} bpm" if result['hr_bpm_welch'] else "HR (Welch)     : N/A",
        f"HR (Final)     : {result['hr_bpm']:.1f} bpm" if result['hr_bpm'] else "HR (Final)     : N/A",
        "",
        f"Mean RRI       : {result['rri_mean_s']*1000:.1f} ms" if result['rri_mean_s'] else "Mean RRI       : N/A",
        f"Valid RRI Count: {result['rri_count']}",
    ]
    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
             fontsize=11, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax4.set_title("(e) Summary")

    fig.suptitle("rPPG Heart Rate Estimation  (POS Method, Ichikawa 2025)", fontsize=13, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Figure saved to: {save_path}")
    
    # hspace等で余白調整している場合は外しても構いませんが、元コード通り残しています
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────
def estimate_hr(csv_path: str, trim_start: float = 2.0, trim_end: float = 2.0,
                plot: bool = True, save_fig: str = None) -> dict:
    """
    Parameters
    ----------
    csv_path    : mediapipe_roi.py が出力した _rgb.csv のパス
    trim_start  : 開始トリミング秒数（論文 3.4.2②）
    trim_end    : 終了トリミング秒数
    plot        : 可視化するか
    save_fig    : 図の保存先パス（None なら保存しない）

    Returns
    -------
    result : dict（hr_bpm, hr_bpm_rri, hr_bpm_welch, rri_mean_s, rri_count）
    """
    # 1. CSV 読み込み
    rgb, fps = load_rgb_csv(csv_path)

    # 2. トリミング（論文 p.35 ②）
    trim_s = int(trim_start * fps)
    trim_e = int(trim_end   * fps)
    if trim_e > 0:
        rgb = rgb[trim_s: len(rgb) - trim_e]
    else:
        rgb = rgb[trim_s:]
    print(f"[INFO] トリミング後フレーム数: {len(rgb)} ({len(rgb)/fps:.1f} s)")

    if len(rgb) < int(POS_WINDOW_SEC * fps) + 1:
        raise ValueError("データが短すぎます。trim_start/trim_end を小さくしてください。")

    # 3. POS 法（rPPG 生信号）
    print("[INFO] POS 法で rPPG 信号を計算中...")
    h_raw = pos_rppg(rgb, fps)

    # 4. BPF
    h_filt = bandpass_filter(h_raw, fps)

    # 5. Welch PSD
    peak_freq, freqs, psd = welch_peak_freq(h_filt, fps)
    print(f"[INFO] Welch ピーク周波数: {peak_freq:.4f} Hz  ({peak_freq*60:.1f} bpm)")

    # 6. RRI 検出
    rri_clean, peak_times = detect_rri(h_filt, fps, peak_freq)
    print(f"[INFO] 有効 RRI 数: {len(rri_clean)}")

    # 7. 心拍数算出
    result = calc_heart_rate(rri_clean, peak_freq)

    print("\n" + "="*45)
    if result["hr_bpm_rri"]:
        print(f"  HR (RRI 平均)  : {result['hr_bpm_rri']:.1f} bpm")
    if result["hr_bpm_welch"]:
        print(f"  HR (Welch)     : {result['hr_bpm_welch']:.1f} bpm")
    print(f"  HR (最終推定)  : {result['hr_bpm']:.1f} bpm")
    if result["rri_mean_s"]:
        print(f"  RRI 平均       : {result['rri_mean_s']*1000:.1f} ms")
    print("="*45 + "\n")

    # 8. 可視化
    if plot:
        plot_results(rgb, h_raw, h_filt, peak_times, freqs, psd, fps, result,
                     save_path=save_fig)

    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="rPPG 心拍推定（POS法 / 市川 2025 論文準拠）"
    )
    parser.add_argument("csv_path",   help="mediapipe_roi.py が出力した _rgb.csv のパス")
    parser.add_argument("--trim_start", type=float, default=2.0, help="開始トリミング秒数（デフォルト: 2.0）")
    parser.add_argument("--trim_end",   type=float, default=2.0, help="終了トリミング秒数（デフォルト: 2.0）")
    parser.add_argument("--no_plot",  action="store_true",       help="グラフを表示しない")
    parser.add_argument("--save_fig", type=str, default=None,    help="グラフ保存先パス（例: result.png）")
    args = parser.parse_args()

    estimate_hr(
        csv_path    = args.csv_path,
        trim_start  = args.trim_start,
        trim_end    = args.trim_end,
        plot        = not args.no_plot,
        save_fig    = args.save_fig,
    )