import cv2
import mediapipe as mp
import numpy as np
import csv

def extract_nostril_rgb_with_pose(input_path, output_path, csv_path):
    # MediaPipe Poseの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0, 1, 2から選択（1が速度と精度のバランスが良い）
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 動画の読み込み
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。パスを確認してください。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 動画保存の設定 (mp4形式)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("処理を開始します。プレビュー画面を選択した状態で 'q' キーを押すと途中終了できます。")

    try:
        # CSVファイルの初期化
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time(s)', 'R_mean', 'G_mean', 'B_mean'])

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # BGRからRGBに変換してPose処理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 鼻と口の両端のランドマークを取得
                    # 0: 鼻, 9: 口(左), 10: 口(右)
                    nose = landmarks[0]
                    mouth_left = landmarks[9]
                    mouth_right = landmarks[10]

                    # 画像上のピクセル座標に変換
                    x_nose, y_nose = int(nose.x * width), int(nose.y * height)
                    x_ml, y_ml = int(mouth_left.x * width), int(mouth_left.y * height)
                    x_mr, y_mr = int(mouth_right.x * width), int(mouth_right.y * height)

                    # 口のY座標の平均と、口の幅を計算
                    y_mouth_avg = (y_ml + y_mr) / 2
                    mouth_width = abs(x_ml - x_mr)

                    # ----------------------------------------------------
                    # ROI領域の推定計算
                    # 鼻の頭から口に向かっての領域を鼻孔周辺(ROI)とする
                    # ----------------------------------------------------
                    # 上端：鼻の頭のY座標
                    y_min = y_nose
                    # 下端：鼻から口までの距離の70%くらいまで（唇を含まないように）
                    y_max = int(y_nose + (y_mouth_avg - y_nose) * 0.7)
                    # 横幅：鼻の中心から、口の幅の40%ずつ左右に広げる
                    x_min = int(x_nose - (mouth_width * 0.4))
                    x_max = int(x_nose + (mouth_width * 0.4))

                    # 画面外に出ないようにクリッピング（丸め込み）
                    x_min = max(0, min(width, x_min))
                    x_max = max(0, min(width, x_max))
                    y_min = max(0, min(height, y_min))
                    y_max = max(0, min(height, y_max))

                    if x_max > x_min and y_max > y_min:
                        # ROIの切り出しとRGB平均値の計算
                        roi = frame[y_min:y_max, x_min:x_max]
                        b_mean, g_mean, r_mean = cv2.mean(roi)[:3]
                        
                        # CSV書き込み
                        time_sec = frame_idx / fps
                        writer.writerow([frame_idx, round(time_sec, 3), r_mean, g_mean, b_mean])

                        # ROIの緑枠を描画
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        
                        # 基準とした3つの点（鼻、口の左右）を赤色で描画して確認しやすくする
                        cv2.circle(frame, (x_nose, y_nose), 1, (0, 0, 255), -1)
                        cv2.circle(frame, (x_ml, y_ml), 1, (0, 0, 255), -1)
                        cv2.circle(frame, (x_mr, y_mr), 1, (0, 0, 255), -1)
                else:
                    # 人物が検出されなかった場合
                    time_sec = frame_idx / fps
                    writer.writerow([frame_idx, round(time_sec, 3), None, None, None])

                out.write(frame)

                # リアルタイムポップアップ表示
                cv2.imshow('Pose ROI Preview (Press "q" to exit)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nユーザー操作により処理を中断しました。")
                    break

                frame_idx += 1

    finally:
        cap.release()
        out.release()
        pose.close()
        cv2.destroyAllWindows()
        print("-----------------------------------")
        print(f"処理が完了（または終了）しました。")
        print(f"・出力動画: {output_path}")
        print(f"・CSVデータ: {csv_path}")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
    INPUT_VIDEO = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\20241119\honma_1set_70%RM.mp4"   # 入力動画のパス
    OUTPUT_VIDEO = 'output_with_pose_roi.mp4' # 出力動画のパス
    CSV_FILE = 'pose_nostril_rgb_data.csv'    # 保存するCSVのパス

    extract_nostril_rgb_with_pose(INPUT_VIDEO, OUTPUT_VIDEO, CSV_FILE)