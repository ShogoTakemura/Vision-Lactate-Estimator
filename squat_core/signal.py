"""
squat_core/signal.py
--------------------
rPPG 心拍推定用信号処理の正規実装 (POS / BPF / Welch / RRI)。
hr_estimation.py と hr_rep_analysis.py の重複関数を統合した単一実装。

参考: 市川太陽 卒業論文 3.2章 (2025) / Wang et al. 2017 (POS 法)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

# 論文 Table 3-4 / p.22–24 準拠デフォルト値
_HR_BPF_LOW: float = 0.8    # Hz
_HR_BPF_HIGH: float = 2.0   # Hz
_HR_POS_WINDOW_S: float = 1.6  # s

_WELCH_NPERSEG_FACTOR: int = 1024   # FFT 長さ = 30 × 1024
_WELCH_NOVERLAP_FACTOR: int = 1024  # オーバーラップ = 10 × 1024
_PEAK_PROM_RATIO: float = 1 / 10   # 最大プロミネンスの 1/10 (論文 p.23)
_RRI_TOLERANCE: float = 0.3         # RRI 外れ値除去幅 [s] (論文 p.24)


def pos_rppg(
    rgb: np.ndarray,
    fps: float,
    window_sec: float = _HR_POS_WINDOW_S,
) -> np.ndarray:
    """POS 法（Wang et al. 2017）で RGB 時系列から rPPG 信号を生成する。
    論文 式(3-7)〜(3-9) 準拠。

    Parameters
    ----------
    rgb        : (N, 3)  [R, G, B]
    fps        : フレームレート [Hz]
    window_sec : 時間窓 [s]  (デフォルト 1.6 s)

    Returns
    -------
    h : (N,) rPPG 信号（オーバーラップ加算・平均化済み）
    """
    N = len(rgb)
    win = max(2, int(window_sec * fps))

    h = np.zeros(N)
    cnt = np.zeros(N, dtype=int)

    for s in range(0, N - win + 1):
        seg = rgb[s : s + win].copy()
        mean_rgb = seg.mean(axis=0)
        if np.any(mean_rgb == 0):
            continue
        xn = seg / mean_rgb
        S1 = xn[:, 0] - xn[:, 1]
        S2 = -2 * xn[:, 0] + xn[:, 1] + xn[:, 2]
        std_S2 = np.std(S2)
        if std_S2 == 0:
            continue
        alpha = np.std(S1) / std_S2
        h_seg = S1 + alpha * S2
        h[s : s + win] += h_seg
        cnt[s : s + win] += 1

    valid = cnt > 0
    h[valid] /= cnt[valid]
    return h


def bandpass_filter(
    signal: np.ndarray,
    fps: float,
    low: float = _HR_BPF_LOW,
    high: float = _HR_BPF_HIGH,
    order: int = 4,
) -> np.ndarray:
    """バターワース BPF。論文 Table 3-4: 0.8–2.0 Hz。"""
    nyq = fps / 2.0
    low_n = np.clip(low / nyq, 1e-4, 0.999)
    high_n = np.clip(high / nyq, 1e-4, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def welch_peak_freq(
    signal: np.ndarray,
    fps: float,
    f_low: float = _HR_BPF_LOW,
    f_high: float = _HR_BPF_HIGH,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Welch 法でピーク周波数を推定する。論文 p.24 準拠。

    Returns
    -------
    peak_freq : ピーク周波数 [Hz]
    freqs     : 周波数軸
    psd       : パワースペクトル密度
    """
    nperseg = min(len(signal), 30 * _WELCH_NPERSEG_FACTOR)
    noverlap = min(len(signal) - 1, min(nperseg - 1, 10 * _WELCH_NOVERLAP_FACTOR))
    freqs, psd = welch(signal, fs=fps, window="hamming",
                       nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0, freqs, psd
    psd_hr = psd.copy()
    psd_hr[~mask] = 0
    return float(freqs[np.argmax(psd_hr)]), freqs, psd


def detect_peaks_prominence(
    signal: np.ndarray,
    prom_ratio: float = _PEAK_PROM_RATIO,
) -> np.ndarray:
    """プロミネンス閾値処理でピーク検出する（セット全体信号用）。論文 p.23 準拠。

    Returns
    -------
    peaks : ピークのサンプルインデックス配列
    """
    peaks, props = find_peaks(signal, prominence=0)
    proms = props["prominences"]
    if len(proms) == 0:
        return np.array([], dtype=int)
    return peaks[proms >= proms.max() * prom_ratio]


def filter_rri(
    rri: np.ndarray,
    ref: float,
    tolerance: float = _RRI_TOLERANCE,
) -> np.ndarray:
    """基準 RRI ± tolerance の範囲外を除去する。除去数が半数超なら除去しない。論文 p.24 準拠。"""
    mask = np.abs(rri - ref) <= tolerance
    return rri if mask.sum() < len(rri) / 2 else rri[mask]


def detect_rri(
    signal: np.ndarray,
    fps: float,
    peak_freq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """プロミネンス閾値処理でピーク検出し、外れ値 RRI を除去する。

    Returns
    -------
    rri_clean  : クリーニング後の RRI 配列 [s]
    peak_times : ピーク時刻 [s]
    """
    peaks = detect_peaks_prominence(signal)
    if len(peaks) < 2:
        return np.array([]), peaks / fps

    peak_times = peaks / fps
    rri = np.diff(peak_times)

    rri_ref = 1.0 / peak_freq if peak_freq > 0 else np.median(rri)
    rri = filter_rri(rri, rri_ref)
    if len(rri) > 0:
        rri = filter_rri(rri, rri.mean())

    return rri, peak_times
