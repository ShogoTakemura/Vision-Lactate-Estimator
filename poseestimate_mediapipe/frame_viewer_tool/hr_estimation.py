"""
心拍推定スクリプト（市川太陽 卒業論文 3.2章 準拠）

処理フロー:
  1. ROI から RGB 信号取得（mediapipe_roi_face.py で生成した CSV を使用）
  2. Skin_pixels=0 のフレームを無効化し線形補間
  3. POS 法（Wang ら）で rPPG 信号を合成
  4. バターワースBPF（0.8–2.0 Hz）で整形
  5. Welch 法でピーク周波数→心拍数(bpm)算出
  6. プロミネンス閾値処理で RRI 検出

対応 CSV 形式（mediapipe_roi_face.py 出力）:
  Frame, Time(s), R_mean, G_mean, B_mean [, Skin_pixels, Nose_x, Nose_y]
  Skin_pixels 列がある場合、0 のフレームを補間対象として扱う。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import os

from squat_core.signal import (
    pos_rppg,
    bandpass_filter,
    welch_peak_freq,
    detect_rri,
)

# ─────────────────────────────────────────────
# 定数（論文 Table 3-4 / 3.2章 準拠）
# ─────────────────────────────────────────────
HR_BPF_LOW     = 0.8   # Hz — Welch PSD プロット用 (論文 Table 3-4)
HR_BPF_HIGH    = 2.0   # Hz
POS_WINDOW_SEC = 1.6   # 秒 — 最短フレーム数チェック用 (論文 p.22)


# ─────────────────────────────────────────────
# 1. CSV 読み込み
# ─────────────────────────────────────────────
def load_rgb_csv(csv_path: str) -> tuple[np.ndarray, float]:
    """
    mediapipe_roi_face.py が出力する CSV を読み込む。
    対応列: Frame, Time(s), R_mean, G_mean, B_mean [, Skin_pixels, Nose_x, Nose_y]

    - Skin_pixels 列が存在する場合: Skin_pixels==0 のフレームを NaN に置換してから補間
    - Skin_pixels 列がない場合: NaN 行をそのまま補間（旧 mediapipe_roi.py 互換）

    Returns
    -------
    rgb : np.ndarray  shape=(N, 3)  [R, G, B]
    fps : float       フレームレート
    """
    df = pd.read_csv(csv_path)
    required = {"Frame", "Time(s)", "R_mean", "G_mean", "B_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns missing: {required - set(df.columns)}")

    df[["R_mean", "G_mean", "B_mean"]] = df[["R_mean", "G_mean", "B_mean"]].apply(
        pd.to_numeric, errors="coerce"
    )

    # Skin_pixels==0 のフレームを無効化（mediapipe_roi_face.py 出力対応）
    if "Skin_pixels" in df.columns:
        df["Skin_pixels"] = pd.to_numeric(df["Skin_pixels"], errors="coerce").fillna(0)
        no_skin = df["Skin_pixels"] == 0
        df.loc[no_skin, ["R_mean", "G_mean", "B_mean"]] = np.nan
        n_masked = int(no_skin.sum())
        if n_masked > 0:
            print(f"[INFO] Skin_pixels=0 frames masked: {n_masked} "
                  f"({n_masked / len(df) * 100:.1f}%)")

    # 線形補間 → 端点補完
    df[["R_mean", "G_mean", "B_mean"]] = (
        df[["R_mean", "G_mean", "B_mean"]]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )
    df = df.dropna(subset=["R_mean", "G_mean", "B_mean"])

    rgb = df[["R_mean", "G_mean", "B_mean"]].values.astype(float)

    times = df["Time(s)"].values
    if len(times) > 1:
        fps = 1.0 / np.median(np.diff(times))
    else:
        fps = 30.0
        print(f"[WARNING] Too few frames; assuming fps={fps}.")

    print(f"[INFO] Frames: {len(rgb)}, fps: {fps:.2f} Hz, duration: {times[-1]:.1f} s")
    return rgb, fps




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
    print(f"[INFO] Frames after trim: {len(rgb)} ({len(rgb)/fps:.1f} s)")

    if len(rgb) < int(POS_WINDOW_SEC * fps) + 1:
        raise ValueError("Too few frames. Reduce trim_start/trim_end.")

    # 3. POS 法（rPPG 生信号）
    print("[INFO] Computing rPPG signal (POS)...")
    h_raw = pos_rppg(rgb, fps)

    # 4. BPF
    h_filt = bandpass_filter(h_raw, fps)

    # 5. Welch PSD
    peak_freq, freqs, psd = welch_peak_freq(h_filt, fps)
    print(f"[INFO] Welch peak: {peak_freq:.4f} Hz  ({peak_freq*60:.1f} bpm)")

    # 6. RRI 検出
    rri_clean, peak_times = detect_rri(h_filt, fps, peak_freq)
    print(f"[INFO] Valid RRI count: {len(rri_clean)}")

    # 7. 心拍数算出
    result = calc_heart_rate(rri_clean, peak_freq)

    print("\n" + "="*45)
    if result["hr_bpm_rri"]:
        print(f"  HR (Mean RRI)  : {result['hr_bpm_rri']:.1f} bpm")
    if result["hr_bpm_welch"]:
        print(f"  HR (Welch)     : {result['hr_bpm_welch']:.1f} bpm")
    print(f"  HR (Final)     : {result['hr_bpm']:.1f} bpm")
    if result["rri_mean_s"]:
        print(f"  Mean RRI       : {result['rri_mean_s']*1000:.1f} ms")
    print("="*45 + "\n")

    # 8. 可視化
    if plot:
        plot_results(rgb, h_raw, h_filt, peak_times, freqs, psd, fps, result,
                     save_path=save_fig)

    return result


# ─────────────────────────────────────────────
# レップ別 HR 推定
# ─────────────────────────────────────────────
def estimate_hr_per_rep(rgb_csv: str, rep_csv: str,
                        save_dir: str = None) -> pd.DataFrame:
    """
    rep CSV で定義された各レップ区間ごとに HR を推定する。

    Parameters
    ----------
    rgb_csv  : mediapipe_roi_face.py が出力した _rgb.csv のパス
    rep_csv  : reps/processed/*_rep.csv のパス
               (列: rep, start_frame, end_frame [, bottom_frame])
    save_dir : PNG / CSV の出力先ディレクトリ（None なら保存しない）

    Returns
    -------
    pd.DataFrame  列: rep, start_frame, end_frame, HR_rri, HR_welch, HR_final,
                      rri_mean_ms, rri_count, n_frames
    """
    import matplotlib.font_manager as fm
    _JP = ["Yu Gothic", "Meiryo", "MS Gothic"]
    _av = {f.name for f in fm.fontManager.ttflist}
    for _f in _JP:
        if _f in _av:
            plt.rcParams["font.family"] = _f
            break

    df_rgb = pd.read_csv(rgb_csv)
    required = {"Frame", "Time(s)", "R_mean", "G_mean", "B_mean"}
    if not required.issubset(df_rgb.columns):
        raise ValueError(f"RGB CSV missing columns: {required - set(df_rgb.columns)}")
    df_rgb[["R_mean", "G_mean", "B_mean"]] = df_rgb[["R_mean", "G_mean", "B_mean"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if "Skin_pixels" in df_rgb.columns:
        df_rgb["Skin_pixels"] = pd.to_numeric(df_rgb["Skin_pixels"], errors="coerce").fillna(0)
        df_rgb.loc[df_rgb["Skin_pixels"] == 0, ["R_mean", "G_mean", "B_mean"]] = np.nan

    df_rep = pd.read_csv(rep_csv)
    for col in ["rep", "start_frame", "end_frame"]:
        if col not in df_rep.columns:
            raise ValueError(f"Rep CSV missing column: {col}")
    df_rep["rep"]         = pd.to_numeric(df_rep["rep"], errors="coerce").astype("Int64")
    df_rep["start_frame"] = pd.to_numeric(df_rep["start_frame"], errors="coerce").astype("Int64")
    df_rep["end_frame"]   = pd.to_numeric(df_rep["end_frame"],   errors="coerce").astype("Int64")

    rows = []
    for _, row in df_rep.dropna(subset=["rep", "start_frame", "end_frame"]).iterrows():
        rep_num = int(row["rep"])
        s, e    = int(row["start_frame"]), int(row["end_frame"])
        seg     = df_rgb[(df_rgb["Frame"] >= s) & (df_rgb["Frame"] <= e)].copy()

        if len(seg) < 10:
            print(f"  [Rep {rep_num}] SKIP: only {len(seg)} frames in range {s}-{e}")
            rows.append({"rep": rep_num, "start_frame": s, "end_frame": e,
                         "HR_rri": None, "HR_welch": None, "HR_final": None,
                         "rri_mean_ms": None, "rri_count": 0, "n_frames": len(seg)})
            continue

        # 補間
        seg[["R_mean", "G_mean", "B_mean"]] = (
            seg[["R_mean", "G_mean", "B_mean"]]
            .interpolate(method="linear").bfill().ffill()
        )
        seg = seg.dropna(subset=["R_mean", "G_mean", "B_mean"])

        times = seg["Time(s)"].values
        fps   = 1.0 / np.median(np.diff(times)) if len(times) > 1 else 30.0
        rgb   = seg[["R_mean", "G_mean", "B_mean"]].values.astype(float)

        if len(rgb) < int(POS_WINDOW_SEC * fps) + 1:
            print(f"  [Rep {rep_num}] SKIP: too short for POS ({len(rgb)} frames)")
            rows.append({"rep": rep_num, "start_frame": s, "end_frame": e,
                         "HR_rri": None, "HR_welch": None, "HR_final": None,
                         "rri_mean_ms": None, "rri_count": 0, "n_frames": len(rgb)})
            continue

        h_raw  = pos_rppg(rgb, fps)
        h_filt = bandpass_filter(h_raw, fps)
        peak_freq, _, _ = welch_peak_freq(h_filt, fps)
        rri_clean, _    = detect_rri(h_filt, fps, peak_freq)
        res = calc_heart_rate(rri_clean, peak_freq)

        print(f"  [Rep {rep_num}] HR={res['hr_bpm']:.1f} bpm  "
              f"(RRI={res['hr_bpm_rri']:.1f} bpm, Welch={res['hr_bpm_welch']:.1f} bpm, "
              f"n_frames={len(rgb)})" if res["hr_bpm"] else
              f"  [Rep {rep_num}] HR=N/A")

        rows.append({
            "rep"        : rep_num,
            "start_frame": s,
            "end_frame"  : e,
            "HR_rri"     : res["hr_bpm_rri"],
            "HR_welch"   : res["hr_bpm_welch"],
            "HR_final"   : res["hr_bpm"],
            "rri_mean_ms": res["rri_mean_s"] * 1000 if res["rri_mean_s"] else None,
            "rri_count"  : res["rri_count"],
            "n_frames"   : len(rgb),
        })

    result_df = pd.DataFrame(rows)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(rgb_csv))[0]

        # CSV 保存
        csv_out = os.path.join(save_dir, f"{base}_hr_per_rep.csv")
        result_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"  CSV saved: {csv_out}")

        # グラフ保存
        valid = result_df.dropna(subset=["HR_final"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(valid["rep"], valid["HR_rri"],   "r--o", label="HR (RRI)",   markersize=5)
            ax.plot(valid["rep"], valid["HR_welch"], "b--s", label="HR (Welch)", markersize=5)
            ax.plot(valid["rep"], valid["HR_final"], "k-o",  label="HR (Final)", linewidth=2, markersize=6)
            ax.set_xlabel("Rep")
            ax.set_ylabel("Heart Rate [bpm]")
            ax.set_xticks(result_df["rep"])
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.suptitle(f"Per-Rep Heart Rate -- {base}", fontsize=12)
            plt.tight_layout()
            fig_out = os.path.join(save_dir, f"{base}_hr_per_rep.png")
            fig.savefig(fig_out, dpi=150, bbox_inches="tight")
            print(f"  Figure saved: {fig_out}")
            plt.close(fig)

    return result_df


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="rPPG heart rate estimation (POS method, Ichikawa 2025)"
    )
    parser.add_argument("csv_path",
                        help="Path to _rgb.csv from mediapipe_roi_face.py")
    parser.add_argument("--rep_csv",    type=str,   default=None,
                        help="Path to *_rep.csv for per-rep estimation")
    parser.add_argument("--save_dir",   type=str,   default=None,
                        help="Output directory for per-rep results (used with --rep_csv)")
    parser.add_argument("--trim_start", type=float, default=2.0,
                        help="Trim seconds from start (default: 2.0)")
    parser.add_argument("--trim_end",   type=float, default=2.0,
                        help="Trim seconds from end (default: 2.0)")
    parser.add_argument("--no_plot",    action="store_true",
                        help="Suppress plot display")
    parser.add_argument("--save_fig",   type=str,   default=None,
                        help="Save figure to this path (e.g. result.png)")
    args = parser.parse_args()

    if args.rep_csv:
        estimate_hr_per_rep(
            rgb_csv  = args.csv_path,
            rep_csv  = args.rep_csv,
            save_dir = args.save_dir,
        )
    else:
        estimate_hr(
            csv_path   = args.csv_path,
            trim_start = args.trim_start,
            trim_end   = args.trim_end,
            plot       = not args.no_plot,
            save_fig   = args.save_fig,
        )