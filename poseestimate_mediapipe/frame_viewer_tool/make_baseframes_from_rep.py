"""
make_baseframes_from_rep.py
----------------------------
frame_viewer_tool/reps/processed/ 内の *_rep.csv を読み込み、
BASEFRAMES.csv を生成する。

【生成される列】
  filename     : *_rep.csv のファイル名（元ファイル名そのまま）
  start        : rep1 の start_frame（セット開始フレーム）
  end          : 最終rep の end_frame（セット終了フレーム）
  base1        : rep2 の start_frame
  base2        : rep3 の start_frame
  baseframe id : 連番ID（1始まり）

【failure データの扱い】
  failure setは最終repの end_frame が「持ち上がらなかったフレームまで」
  含まれているが、end にはそのまま使用する（セット全体を網羅するため）。
  base1/base2 は2rep・3rep が存在する場合のみ設定し、
  存在しない場合は空欄とする。

【使い方】
  1. PROCESSED_DIR を実際のパスに変更して実行
  2. OUTPUT_PATH に BASEFRAMES.csv が生成される

     python make_baseframes_from_rep.py
"""

import os
import glob
import pandas as pd

# ============================================================
# パス設定（環境に合わせて変更してください）
# ============================================================
PROCESSED_DIR = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\frame_viewer_tool\reps\processed"
OUTPUT_PATH   = os.path.join(PROCESSED_DIR, "BASEFRAMES.csv")


def main():
    # *_rep.csv を検索
    search_pattern = os.path.join(PROCESSED_DIR, "*_rep.csv")
    rep_files = sorted(glob.glob(search_pattern))

    if not rep_files:
        print(f"[ERROR] *_rep.csv が見つかりません: {PROCESSED_DIR}")
        return

    print(f"{len(rep_files)} 件の *_rep.csv を検出しました。\n")

    output_rows = []
    baseframe_id = 1

    for file_path in rep_files:
        filename = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path, encoding="utf-8")

            # 必須列チェック
            required_cols = {"rep", "start_frame", "end_frame"}
            if not required_cols.issubset(df.columns):
                print(f"[SKIP] 必須列が不足しています ({required_cols - set(df.columns)}): {filename}")
                continue

            # rep 列を整数に統一
            df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
            df = df.dropna(subset=["rep"])
            df["rep"] = df["rep"].astype(int)

            # セット全体の start / end
            start = int(df.loc[df["rep"] == 1, "start_frame"].iloc[0]) \
                if not df[df["rep"] == 1].empty else int(df["start_frame"].min())
            end   = int(df["end_frame"].max())

            # base1 : rep2 の start_frame
            rep2_rows = df[df["rep"] == 2]
            base1 = int(rep2_rows["start_frame"].iloc[0]) if not rep2_rows.empty else ""

            # base2 : rep3 の start_frame
            rep3_rows = df[df["rep"] == 3]
            base2 = int(rep3_rows["start_frame"].iloc[0]) if not rep3_rows.empty else ""

            output_rows.append({
                "filename"    : filename,
                "start"       : start,
                "end"         : end,
                "base1"       : base1,
                "base2"       : base2,
                "baseframe id": baseframe_id,
            })

            rep_count = len(df)
            b1_str = str(base1) if base1 != "" else "N/A"
            b2_str = str(base2) if base2 != "" else "N/A"
            print(f"  [{baseframe_id:>3}] {filename}")
            print(f"         reps={rep_count}  start={start}  end={end}"
                  f"  base1={b1_str}  base2={b2_str}")

            baseframe_id += 1

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    if not output_rows:
        print("\n[ERROR] 有効なデータが1件もありませんでした。出力を中止します。")
        return

    # DataFrame 作成・保存
    out_df = pd.DataFrame(
        output_rows,
        columns=["filename", "start", "end", "base1", "base2", "baseframe id"]
    )

    # base1/base2 が空欄の行は空文字のまま（Int64 変換しない）
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ 処理完了: {len(output_rows)} 件")
    print(f"   出力先: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
