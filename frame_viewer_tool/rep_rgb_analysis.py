"""
rep_rgb_analysis.py
────────────────────────────────────────────────────────────────────
RGB時系列データ（roi_out/*_rgb.csv）と
レップアノテーション（reps/processed/*_rep.csv）を組み合わせ、
レップごとのRGB変化を可視化する。

【出力グラフ】
  1. RGB時系列 + レップ区間の背景色分け（縦線でstart/bottom/end表示）
  2. レップ番号 vs RGB平均値の折れ線グラフ

【使い方】
  RGB_CSV, REP_CSV のパスを下部の「設定」セクションで指定して実行。
  複数セットを比較するには COMPARE_SETS にリストとして追加する。
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォント設定（優先順で最初に見つかったものを使用）
_JP_FONTS = ["Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic", "IPAGothic"]
_available = {f.name for f in fm.fontManager.ttflist}
for _f in _JP_FONTS:
    if _f in _available:
        plt.rcParams["font.family"] = _f
        break


# ═══════════════════════════════════════════════════════
#  設定
# ═══════════════════════════════════════════════════════

# ── 単一セット解析 ──────────────────────────────────────
RGB_CSV = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\roi_out\20250114\kikuchi_sayaka_1set_rgb.csv"
REP_CSV = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\reps\processed\20221102_kikuchi_sayaka-80per-10rep-set1_rep.csv"

# ── 複数セット比較（リストに追加すると重ねてプロット） ─
#    各要素: (ラベル, rgb_csv_path, rep_csv_path)
COMPARE_SETS: list[tuple[str, str, str]] = [
    # ("set1", RGB_CSV, REP_CSV),
    # ("set2", r"..._2set_rgb.csv", r"..._set2_rep.csv"),
]

OUTPUT_DIR   = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\rep_analysis_out"
SAVE_FIGURES = True    # True: PNG保存, False: plt.show()
EXPORT_CSV   = True    # True: レップ別平均CSVを出力

# ═══════════════════════════════════════════════════════


# ── データ読み込み ──────────────────────────────────────

def load_rgb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["Frame", "R_mean", "G_mean", "B_mean"]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が見つかりません: {path}")
    df[["R_mean", "G_mean", "B_mean"]] = df[["R_mean", "G_mean", "B_mean"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return df


def load_rep(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["rep", "start_frame", "end_frame"]:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が見つかりません: {path}")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce").astype("Int64")
    df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce").astype("Int64")
    df["end_frame"]   = pd.to_numeric(df["end_frame"],   errors="coerce").astype("Int64")
    if "bottom_frame" in df.columns:
        df["bottom_frame"] = pd.to_numeric(df["bottom_frame"], errors="coerce").astype("Int64")
    return df.dropna(subset=["rep", "start_frame", "end_frame"])


# ── レップごとのセグメント抽出 ──────────────────────────

def extract_segments(rgb_df: pd.DataFrame, rep_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    rgb_min, rgb_max = int(rgb_df["Frame"].min()), int(rgb_df["Frame"].max())
    segments = {}
    empty_reps = []
    for _, row in rep_df.iterrows():
        rep_num = int(row["rep"])
        start   = int(row["start_frame"])
        end     = int(row["end_frame"])
        mask    = (rgb_df["Frame"] >= start) & (rgb_df["Frame"] <= end)
        seg = rgb_df[mask].copy()
        segments[rep_num] = seg
        if seg.empty:
            empty_reps.append(rep_num)
    if empty_reps:
        print(f"  [WARNING] Rep {empty_reps}: no RGB data (0 frames).")
        print(f"         RGB range: {rgb_min}-{rgb_max}, Rep range: "
              f"{int(rep_df['start_frame'].min())}-{int(rep_df['end_frame'].max())}")
        print("         Check that the frame numbers in RGB CSV and Rep CSV match.")
    return segments


def compute_stats(segments: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for rep_num, seg in sorted(segments.items()):
        valid = seg.dropna(subset=["R_mean", "G_mean", "B_mean"])
        if valid.empty:
            rows.append({"rep": rep_num, "R_mean": np.nan, "G_mean": np.nan, "B_mean": np.nan,
                         "R_std": np.nan, "G_std": np.nan, "B_std": np.nan, "n_frames": 0})
        else:
            rows.append({
                "rep"     : rep_num,
                "R_mean"  : valid["R_mean"].mean(),
                "G_mean"  : valid["G_mean"].mean(),
                "B_mean"  : valid["B_mean"].mean(),
                "R_std"   : valid["R_mean"].std(),
                "G_std"   : valid["G_mean"].std(),
                "B_std"   : valid["B_mean"].std(),
                "n_frames": len(valid),
            })
    return pd.DataFrame(rows)


# ── プロット：RGB時系列 ────────────────────────────────

def plot_timeseries(rgb_df: pd.DataFrame, rep_df: pd.DataFrame,
                    title: str, save_path: str | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    ch_info = [("R_mean", "red"), ("G_mean", "green"), ("B_mean", "blue")]

    bg_colors = ["#f5f5f5", "#e8f0ff"]
    has_bottom = "bottom_frame" in rep_df.columns

    for _, row in rep_df.iterrows():
        s = int(row["start_frame"])
        e = int(row["end_frame"])
        rep_num = int(row["rep"])
        for ax in axes:
            ax.axvspan(s, e, color=bg_colors[rep_num % 2], alpha=0.7, zorder=0)
            ax.axvline(s, color="gray",  linewidth=0.6, linestyle="--", zorder=1)
            ax.axvline(e, color="gray",  linewidth=0.6, linestyle=":",  zorder=1)
            if has_bottom and pd.notna(row.get("bottom_frame")):
                b = int(row["bottom_frame"])
                ax.axvline(b, color="orange", linewidth=0.8, linestyle="-.", zorder=1)

    for ax, (col, color) in zip(axes, ch_info):
        valid = rgb_df.dropna(subset=[col])
        ax.plot(valid["Frame"], valid[col], color=color, linewidth=0.7)
        ax.set_ylabel(col.replace("_mean", ""), color=color, fontsize=10)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, alpha=0.2)

    # レップ番号をグラフ上部に表示
    ylim_top = axes[0].get_ylim()[1]
    for _, row in rep_df.iterrows():
        mid = (int(row["start_frame"]) + int(row["end_frame"])) / 2
        axes[0].text(mid, ylim_top * 0.98, f"R{int(row['rep'])}",
                     ha="center", va="top", fontsize=7, color="dimgray")

    axes[-1].set_xlabel("Frame", fontsize=10)
    fig.suptitle(title, fontsize=12, y=1.01)

    # 凡例（最初のサブプロットに）
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color="gray",   linestyle="--", label="Start"),
        Line2D([0], [0], color="gray",   linestyle=":",  label="End"),
    ]
    if has_bottom:
        legend_items.append(Line2D([0], [0], color="orange", linestyle="-.", label="Bottom"))
    axes[0].legend(handles=legend_items, loc="upper right", fontsize=7)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── プロット：レップ別RGB平均 ──────────────────────────

def plot_rep_means(stats_df: pd.DataFrame, title: str, save_path: str | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    reps = stats_df["rep"]
    for col, color, label in [("R_mean", "red", "R"), ("G_mean", "green", "G"), ("B_mean", "blue", "B")]:
        y = stats_df[col]
        err = stats_df[col.replace("mean", "std")]
        ax.plot(reps, y, f"{color[0]}-o", label=label, linewidth=1.5, markersize=5)
        ax.fill_between(reps, y - err, y + err, alpha=0.15, color=color)

    ax.set_xlabel("Rep", fontsize=11)
    ax.set_ylabel("Mean pixel value", fontsize=11)
    ax.set_xticks(reps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── 比較プロット：複数セットのレップ別RGB平均 ────────────

def plot_compare_sets(label_stats_pairs: list[tuple[str, pd.DataFrame]],
                      save_path: str | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ch_pairs = [("R_mean", "red", "R"), ("G_mean", "green", "G"), ("B_mean", "blue", "B")]
    linestyles = ["-", "--", "-.", ":"]

    for ax, (col, color, label) in zip(axes, ch_pairs):
        for i, (set_label, stats_df) in enumerate(label_stats_pairs):
            ls = linestyles[i % len(linestyles)]
            ax.plot(stats_df["rep"], stats_df[col],
                    color=color, linestyle=ls, marker="o", markersize=5,
                    label=f"{label} – {set_label}")
        ax.set_ylabel(label, color=color, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Rep", fontsize=11)
    fig.suptitle("セット間比較 – レップ別RGB平均", fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ── ユーティリティ ─────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ── メイン ──────────────────────────────────────────────

def analyze_single(rgb_path: str, rep_path: str, label: str | None = None) -> pd.DataFrame:
    if label is None:
        label = _basename(rgb_path)

    rgb_df = load_rgb(rgb_path)
    rep_df = load_rep(rep_path)

    segments = extract_segments(rgb_df, rep_df)
    stats    = compute_stats(segments)

    print(f"\n=== {label} ===")
    print(stats[["rep", "R_mean", "G_mean", "B_mean", "n_frames"]].to_string(index=False))

    ts_path   = os.path.join(OUTPUT_DIR, f"{label}_timeseries.png") if SAVE_FIGURES else None
    mean_path = os.path.join(OUTPUT_DIR, f"{label}_rep_means.png")  if SAVE_FIGURES else None

    plot_timeseries(rgb_df, rep_df, f"RGB 時系列 – {label}", ts_path)
    plot_rep_means(stats, f"レップ別 RGB 平均 – {label}", mean_path)

    if EXPORT_CSV:
        csv_out = os.path.join(OUTPUT_DIR, f"{label}_rep_stats.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stats.to_csv(csv_out, index=False, encoding="utf-8-sig")
        print(f"  CSV saved: {csv_out}")

    return stats


def main(rgb_csv: str = RGB_CSV, rep_csv: str = REP_CSV,
         output_dir: str = OUTPUT_DIR, save_figures: bool = SAVE_FIGURES,
         export_csv: bool = EXPORT_CSV) -> None:
    # モジュールレベル変数を引数でオーバーライドできるようにする
    global OUTPUT_DIR, SAVE_FIGURES, EXPORT_CSV
    OUTPUT_DIR   = output_dir
    SAVE_FIGURES = save_figures
    EXPORT_CSV   = export_csv

    # 単一セット解析
    analyze_single(rgb_csv, rep_csv)

    # 複数セット比較（COMPARE_SETS はスクリプト先頭の設定セクションで指定）
    if COMPARE_SETS:
        label_stats_pairs = []
        for lbl, r, p in COMPARE_SETS:
            stats = analyze_single(r, p, label=lbl)
            label_stats_pairs.append((lbl, stats))

        cmp_path = os.path.join(OUTPUT_DIR, "compare_sets_rep_means.png") if SAVE_FIGURES else None
        plot_compare_sets(label_stats_pairs, cmp_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-rep RGB analysis from mediapipe_roi_face output")
    parser.add_argument("--rgb_csv",    default=RGB_CSV,    help="Path to _rgb.csv")
    parser.add_argument("--rep_csv",    default=REP_CSV,    help="Path to _rep.csv")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for PNGs and CSV")
    parser.add_argument("--no_save",    action="store_true", help="Display plots instead of saving")
    parser.add_argument("--no_csv",     action="store_true", help="Skip CSV export")
    args = parser.parse_args()

    main(
        rgb_csv      = args.rgb_csv,
        rep_csv      = args.rep_csv,
        output_dir   = args.output_dir,
        save_figures = not args.no_save,
        export_csv   = not args.no_csv,
    )
