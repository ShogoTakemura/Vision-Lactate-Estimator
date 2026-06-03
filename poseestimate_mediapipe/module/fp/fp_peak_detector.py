"""
FP (フォースプレート) Fz 信号に基づくスクワット rep 検出モジュール

poseestimate_mediapipe/module/fp/fp_peak_detector.py として配置する。

使い方:
    detector = FPPeakDetector(
        fp_df     = fp_dataframe,       # FPの読み込み済みDataFrame
        fps       = 30.0,
        body_weight_n = 929.1,          # M*g (kg * 9.79)
        fp_to_video_offset = com_start - lag,
    )
    detector.run()

    # 全 rep の情報
    df_reps = detector.rep_table()      # per-rep DataFrame
    features = detector.set_features()  # セット単位の特徴量 dict
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


class FPPeakDetector:
    """
    FP Fz 信号からスクワット rep を検出し、per-rep 特徴量を算出するクラス。

    検出手順:
        1. Fz を 6Hz ローパスフィルタでスムージング
        2. Fz > body_weight * height_ratio のピーク (concentric peak) を検出
        3. 隣接ピーク間の谷 (squat bottom) を探索
        4. FP フレームインデックス → ビデオフレームインデックスに変換

    per-rep 特徴量:
        peak_fz     : concentric ピーク力 [N]
        valley_fz   : スクワット底部の最小力 [N]
        rep_duration: レップ所要時間 [s]
        impulse     : (Fz - body_weight) の正積分 [N·s]  (concentric 相)
        rfd         : Rate of Force Development [N/s]  (谷 → ピーク間)
        unload_ratio: valley_fz / body_weight  (1.0 なら荷重抜けなし)
    """

    def __init__(
        self,
        fp_df: pd.DataFrame,
        fps: float = 30.0,
        body_weight_n: float | None = None,
        fp_to_video_offset: int = 0,
        lp_cutoff_hz: float = 6.0,
        peak_height_ratio: float = 1.05,
        peak_min_duration_s: float = 0.8,
        peak_prominence_ratio: float = 0.04,
        fz_col: str = 'Fz',
        on_plate_threshold_n: float = 200.0,
    ) -> None:
        """
        Args:
            fp_df               : FP DataFrame (skiprows=5 済み)
            fps                 : フレームレート
            body_weight_n       : M*g [N]。None なら on-plate 区間の中央 80% mean から推定
            fp_to_video_offset  : fp_global_idx + this = video_frame_idx
                                  = com_start - lag
            lp_cutoff_hz        : ローパスフィルタ遮断周波数 [Hz]
            peak_height_ratio   : concentric peak の閾値 (body_weight の何倍以上)
            peak_min_duration_s : レップ間最小間隔 [s]
            peak_prominence_ratio: ピーク顕著さの閾値 (body_weight の何倍以上)
            fz_col              : Fz の列名
            on_plate_threshold_n: 乗板判定閾値 [N]
        """
        self._fz_all = fp_df[fz_col].values.copy()
        self._fps = fps
        self._offset = fp_to_video_offset
        self._lp_cut = lp_cutoff_hz
        self._height_r = peak_height_ratio
        self._min_dist = int(fps * peak_min_duration_s)
        self._prom_r = peak_prominence_ratio
        self._on_thresh = on_plate_threshold_n

        # on-plate 区間を特定
        on = np.where(np.abs(self._fz_all) > on_plate_threshold_n)[0]
        if len(on) == 0:
            raise ValueError("Fz が閾値を超えるフレームが見つかりません")
        self._fp_motion_start = int(on[0])
        self._fp_motion_end   = int(on[-1])
        self._fz_motion = self._fz_all[self._fp_motion_start : self._fp_motion_end + 1]

        # 体重推定
        if body_weight_n is None:
            n = len(self._fz_motion)
            i1, i2 = int(n * 0.1), int(n * 0.9)
            self._bw = float(np.mean(self._fz_motion[i1:i2]))
        else:
            self._bw = float(body_weight_n)

        # 結果
        self._peaks_local:   np.ndarray = np.array([], dtype=int)
        self._valleys_local: np.ndarray = np.array([], dtype=int)
        self._fz_filt: np.ndarray = np.array([])

    # ----------------------------------------------------------------
    # 実行
    # ----------------------------------------------------------------

    def run(self) -> None:
        """ピーク検出を実行"""
        # 1. ローパスフィルタ
        b, a = butter(4, self._lp_cut / (self._fps / 2.0), btype='low')
        self._fz_filt = filtfilt(b, a, self._fz_motion)

        # 2. Concentric peak 検出
        self._peaks_local, _ = find_peaks(
            self._fz_filt,
            height     = self._bw * self._height_r,
            distance   = self._min_dist,
            prominence = self._bw * self._prom_r,
        )

        # 3. 隣接ピーク間の谷を検出
        valleys = []
        for i in range(len(self._peaks_local) - 1):
            seg = self._fz_filt[self._peaks_local[i] : self._peaks_local[i + 1]]
            valleys.append(self._peaks_local[i] + int(np.argmin(seg)))
        self._valleys_local = np.array(valleys, dtype=int)

    # ----------------------------------------------------------------
    # インデックス変換
    # ----------------------------------------------------------------

    def _to_fp_global(self, local_idx: np.ndarray) -> np.ndarray:
        return local_idx + self._fp_motion_start

    def _to_video_frame(self, local_idx: np.ndarray) -> np.ndarray:
        return self._to_fp_global(local_idx) + self._offset

    # ----------------------------------------------------------------
    # per-rep 特徴量
    # ----------------------------------------------------------------

    def rep_table(self) -> pd.DataFrame:
        """全 rep の特徴量 DataFrame を返す"""
        if len(self._peaks_local) == 0:
            return pd.DataFrame()

        rows = []
        peaks = self._peaks_local
        valleys = self._valleys_local
        fz = self._fz_filt
        dt = 1.0 / self._fps

        for i, p in enumerate(peaks):
            # valley は i 番目の peak の前にある (peak_i-1 < valley_i < peak_i)
            v = valleys[i - 1] if i > 0 and i - 1 < len(valleys) else None

            # rep duration (前ピークから現ピーク)
            p_prev = peaks[i - 1] if i > 0 else max(0, p - self._min_dist)
            dur_s = (p - p_prev) * dt

            # 力積 (concentric 相: valley → peak)
            if v is not None:
                seg = fz[v:p + 1]
                impulse = float(np.sum(np.maximum(seg - self._bw, 0)) * dt)
                # RFD: (peak_fz - valley_fz) / time
                rfd = (fz[p] - fz[v]) / ((p - v) * dt) if p > v else np.nan
                valley_fz = float(fz[v])
                valley_video = int(self._to_video_frame(np.array([v]))[0])
                unload_ratio = valley_fz / self._bw
            else:
                impulse = np.nan
                rfd = np.nan
                valley_fz = np.nan
                valley_video = np.nan
                unload_ratio = np.nan

            rows.append({
                'rep_no':            i + 1,
                'peak_fp_idx':       int(self._to_fp_global(np.array([p]))[0]),
                'peak_video_frame':  int(self._to_video_frame(np.array([p]))[0]),
                'peak_fz_N':         float(fz[p]),
                'valley_fp_idx':     int(self._to_fp_global(np.array([v]))[0]) if v is not None else np.nan,
                'valley_video_frame': valley_video,
                'valley_fz_N':       valley_fz,
                'rep_duration_s':    dur_s,
                'impulse_Ns':        impulse,
                'rfd_N_per_s':       rfd,
                'unload_ratio':      unload_ratio,
            })

        return pd.DataFrame(rows)

    def set_features(self) -> dict:
        """セット単位の特徴量 dict を返す (乳酸推定の入力として使用)"""
        df = self.rep_table()
        if df.empty:
            return {}

        return {
            'n_reps':              len(df),
            'mean_peak_fz_N':      float(df['peak_fz_N'].mean()),
            'std_peak_fz_N':       float(df['peak_fz_N'].std()),
            'mean_valley_fz_N':    float(df['valley_fz_N'].mean()),
            'mean_rep_duration_s': float(df['rep_duration_s'].mean()),
            'std_rep_duration_s':  float(df['rep_duration_s'].std()),
            'mean_impulse_Ns':     float(df['impulse_Ns'].mean()),
            'mean_rfd_N_per_s':    float(df['rfd_N_per_s'].mean()),
            'mean_unload_ratio':   float(df['unload_ratio'].mean()),
            # 疲労指標: 後半 / 前半 の比 (疲れると遅くなる・力が落ちる)
            'fatigue_dur_ratio':   float(df['rep_duration_s'].iloc[len(df)//2:].mean()
                                         / df['rep_duration_s'].iloc[:len(df)//2].mean()),
            'fatigue_peak_ratio':  float(df['peak_fz_N'].iloc[len(df)//2:].mean()
                                         / df['peak_fz_N'].iloc[:len(df)//2].mean()),
            'body_weight_N':       self._bw,
        }

    @property
    def n_reps(self) -> int:
        return len(self._peaks_local)

    @property
    def body_weight(self) -> float:
        return self._bw


# ────────────────────────────────────────────────────────────────────
# ヘルパ: FP DataFrame の読み込み
# ────────────────────────────────────────────────────────────────────

def load_fp_csv(path: str, encoding: str = 'cp932') -> pd.DataFrame:
    """FP CSV を読み込む (サンプリング周波数行をスキップ)"""
    import chardet
    with open(path, 'rb') as f:
        enc = chardet.detect(f.read(4096))['encoding'] or encoding
    return pd.read_csv(path, skiprows=5, encoding=enc)
