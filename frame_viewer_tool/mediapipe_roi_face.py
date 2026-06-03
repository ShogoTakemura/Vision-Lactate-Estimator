"""
mediapipe_roi_face.py
─────────────────────────────────────────────────────────────
論文「自律型非接触バイタルサイン計測と機械学習に基づく信頼度判定」
（市川太陽, 東京理科大学, 2025）3.2.2 章 準拠

【処理フロー】
  MediaPipe Pose → Landmark 0（鼻先）中心に ±100 px 顔画像クロップ（論文 p.17, Fig.3-8）
  → HSV + YCbCr 二重閾値で肌色マスク生成（論文 式3-1〜3-6, Table 3-2, 3-3）
  → 肌色領域画素の RGB 平均を CSV に記録

【旧 mediapipe_roi.py との変更点】
  旧: Landmark 0/9/10 から鼻周辺矩形を計算 → ROI全画素のBGR平均（肌色マスクなし）
  新: Landmark 0（鼻先）中心 ±100 px → HSV+YCbCr肌色マスク → 肌色領域のみRGB平均

【肌色マスク閾値（論文 Table 3-2, 3-3）】
  HSV  : H ∈ [0,17],  S ∈ [15,170], V ∈ [0,255]
  YCbCr: Y ∈ [0,255], Cb ∈ [85,135], Cr ∈ [135,180]
  ※ 論文 Table 3-3 の Cb/Cr 列は式(3-5)(3-6)の定義と表記が入れ替わっている。
     実際の肌色サンプルで式(3-5)(3-6)の計算値を検証し、上記範囲を採用。

【使用ランドマーク（MediaPipe Pose）】
  Landmark 0: NOSE（鼻先）→ クロップ基準点
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import glob

# ─── 論文準拠定数 ─────────────────────────────────────────────
CROP_SIZE = 100   # 鼻先中心から ±100 px（論文 p.17）

# HSV 肌色閾値（論文 Table 3-2）
HSV_H_MIN, HSV_H_MAX =   0,  17
HSV_S_MIN, HSV_S_MAX =  15, 170
HSV_V_MIN, HSV_V_MAX =   0, 255

# YCbCr 肌色閾値（論文 Table 3-3 / 式(3-5)(3-6)で動作確認済み）
# OpenCV の COLOR_BGR2YCrCb は (Y, Cr, Cb) 順で格納される点に注意
YCBCR_Y_MIN,  YCBCR_Y_MAX  =   0, 255
YCBCR_CB_MIN, YCBCR_CB_MAX =  85, 135   # 論文式(3-5) の Cb
YCBCR_CR_MIN, YCBCR_CR_MAX = 135, 180   # 論文式(3-6) の Cr
# ──────────────────────────────────────────────────────────────


def skin_mask_hsv_ycbcr(bgr_img: np.ndarray) -> np.ndarray:
    """
    HSV と YCbCr の二重閾値で肌色マスクを生成する。
    論文 式(3-1)〜(3-6) および Table 3-2, 3-3 準拠。

    Parameters
    ----------
    bgr_img : (H, W, 3) uint8  BGR 画像

    Returns
    -------
    mask : (H, W) uint8  肌色領域=255, 非肌色=0
    """
    # ── HSV マスク（論文 式3-1〜3-3, Table 3-2）──────────────
    hsv      = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(
        hsv,
        (HSV_H_MIN, HSV_S_MIN, HSV_V_MIN),
        (HSV_H_MAX, HSV_S_MAX, HSV_V_MAX),
    )

    # ── YCbCr マスク（論文 式3-4〜3-6, Table 3-3）────────────
    # OpenCV の COLOR_BGR2YCrCb は channel 順が (Y, Cr, Cb)
    ycrcb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    Y_ch   = ycrcb[:, :, 0]   # Y  （輝度）
    Cr_ch  = ycrcb[:, :, 1]   # Cr （論文式3-6）
    Cb_ch  = ycrcb[:, :, 2]   # Cb （論文式3-5）

    mask_ycbcr = (
        (Y_ch  >= YCBCR_Y_MIN)  & (Y_ch  <= YCBCR_Y_MAX)  &
        (Cb_ch >= YCBCR_CB_MIN) & (Cb_ch <= YCBCR_CB_MAX) &
        (Cr_ch >= YCBCR_CR_MIN) & (Cr_ch <= YCBCR_CR_MAX)
    ).astype(np.uint8) * 255

    # AND（両条件を同時に満たす画素のみを肌色と判定）
    return cv2.bitwise_and(mask_hsv, mask_ycbcr)


def extract_face_rgb_with_pose(input_path: str,
                                output_path: str,
                                csv_path: str) -> bool:
    """
    MediaPipe Pose で顔ランドマークを取得し、
    鼻先(Landmark 0)中心 ±CROP_SIZE px の顔画像から
    肌色マスク後の RGB 平均を CSV に記録する。

    CSV 列: Frame, Time(s), R_mean, G_mean, B_mean, Skin_pixels, Nose_x, Nose_y

    Parameters
    ----------
    input_path  : 入力動画パス
    output_path : ROI 可視化済み動画の出力パス
    csv_path    : RGB 時系列 CSV の出力パス

    Returns
    -------
    bool : True=正常完了, False=ユーザー中断
    """
    # ── MediaPipe Pose 初期化 ──────────────────────────────────
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした ({input_path})")
        return False

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    user_aborted = False

    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "Time(s)",
                             "R_mean", "G_mean", "B_mean",
                             "Skin_pixels", "Nose_x", "Nose_y"])

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = pose.process(rgb_frame)
                time_sec  = frame_idx / fps

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    # ── Landmark 0 = NOSE（論文 Fig. 3-8 基準点）──
                    nose_x = int(lm[0].x * width)
                    nose_y = int(lm[0].y * height)

                    # ── ±CROP_SIZE px で顔画像をクロップ（論文 p.17）──
                    x_min = max(0, nose_x - CROP_SIZE)
                    x_max = min(width,  nose_x + CROP_SIZE)
                    y_min = max(0, nose_y - CROP_SIZE)
                    y_max = min(height, nose_y + CROP_SIZE)

                    face_crop = frame[y_min:y_max, x_min:x_max]

                    if face_crop.size == 0:
                        writer.writerow([frame_idx, round(time_sec, 3),
                                         None, None, None, 0, nose_x, nose_y])
                        out.write(frame)
                        frame_idx += 1
                        continue

                    # ── HSV + YCbCr 肌色マスク（論文 Table 3-2, 3-3）──
                    mask        = skin_mask_hsv_ycbcr(face_crop)
                    skin_pixels = int(np.count_nonzero(mask))

                    if skin_pixels > 0:
                        # 肌色領域のみの BGR 平均 → RGB として保存
                        b_mean = float(cv2.mean(face_crop[:, :, 0], mask=mask)[0])
                        g_mean = float(cv2.mean(face_crop[:, :, 1], mask=mask)[0])
                        r_mean = float(cv2.mean(face_crop[:, :, 2], mask=mask)[0])
                        writer.writerow([frame_idx, round(time_sec, 3),
                                         round(r_mean, 4),
                                         round(g_mean, 4),
                                         round(b_mean, 4),
                                         skin_pixels, nose_x, nose_y])
                    else:
                        writer.writerow([frame_idx, round(time_sec, 3),
                                         None, None, None, 0, nose_x, nose_y])

                    # ── 可視化描画 ─────────────────────────────────
                    # 肌色マスクを水色で半透明オーバーレイ
                    overlay = np.zeros_like(face_crop)
                    overlay[mask > 0] = (255, 200, 0)   # BGR 水色
                    blended = cv2.addWeighted(face_crop, 0.7, overlay, 0.3, 0)
                    frame[y_min:y_max, x_min:x_max] = blended

                    # 顔クロップ枠（緑）
                    cv2.rectangle(frame,
                                  (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 1)

                    # 鼻先ランドマーク（赤）
                    cv2.circle(frame, (nose_x, nose_y), 3, (0, 0, 255), -1)

                    # 肌色画素数を表示
                    cv2.putText(frame,
                                f"Skin:{skin_pixels}px",
                                (x_min, max(0, y_min - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    # ランドマーク未検出フレーム
                    writer.writerow([frame_idx, round(time_sec, 3),
                                     None, None, None, 0, None, None])

                out.write(frame)
                cv2.imshow("Pose ROI Preview (Press 'q' to exit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[中断] ユーザー操作により処理を中止します。")
                    user_aborted = True
                    break

                frame_idx += 1

    finally:
        cap.release()
        out.release()
        pose.close()
        cv2.destroyAllWindows()

    return not user_aborted


def process_directory(input_dir: str, output_dir: str) -> None:
    """input_dir 内の全 .mp4 を一括処理する。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力フォルダを作成しました: {output_dir}")

    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    if not video_files:
        print(f"指定フォルダに .mp4 ファイルが見つかりません: {input_dir}")
        return

    print(f"合計 {len(video_files)} 件の動画ファイルを処理します。")
    print("プレビュー画面で 'q' キーを押すと一括処理を途中終了できます。\n")

    for idx, video_path in enumerate(video_files, 1):
        basename              = os.path.basename(video_path)
        name_without_ext, ext = os.path.splitext(basename)

        out_video = os.path.join(output_dir, f"{name_without_ext}_roi{ext}")
        out_csv   = os.path.join(output_dir, f"{name_without_ext}_rgb.csv")

        print(f"[{idx}/{len(video_files)}] 処理中: {basename}")
        success = extract_face_rgb_with_pose(video_path, out_video, out_csv)

        if not success:
            print("一括処理を中断しました。")
            break

        print(f"  完了 → 動画: {os.path.basename(out_video)}, "
              f"CSV: {os.path.basename(out_csv)}\n")

    print("すべての処理が完了（または終了）しました。")


# ── 実行部分 ────────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_DIR  = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\20241119"
    OUTPUT_DIR = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\roi_out\20241119"

    process_directory(INPUT_DIR, OUTPUT_DIR)
