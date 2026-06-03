import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import glob

def extract_nostril_rgb_with_pose(input_path, output_path, csv_path):
    # MediaPipe Poseの初期化
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイルを開けませんでした ({input_path})")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    user_aborted = False

    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time(s)', 'R_mean', 'G_mean', 'B_mean'])

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    nose = landmarks[0]
                    mouth_left = landmarks[9]
                    mouth_right = landmarks[10]

                    x_nose, y_nose = int(nose.x * width), int(nose.y * height)
                    x_ml, y_ml = int(mouth_left.x * width), int(mouth_left.y * height)
                    x_mr, y_mr = int(mouth_right.x * width), int(mouth_right.y * height)

                    y_mouth_avg = (y_ml + y_mr) / 2
                    mouth_width = abs(x_ml - x_mr)

                    y_min = y_nose
                    y_max = int(y_nose + (y_mouth_avg - y_nose) * 0.7)
                    x_min = int(x_nose - (mouth_width * 0.4))
                    x_max = int(x_nose + (mouth_width * 0.4))

                    x_min = max(0, min(width, x_min))
                    x_max = max(0, min(width, x_max))
                    y_min = max(0, min(height, y_min))
                    y_max = max(0, min(height, y_max))

                    if x_max > x_min and y_max > y_min:
                        roi = frame[y_min:y_max, x_min:x_max]
                        b_mean, g_mean, r_mean = cv2.mean(roi)[:3]
                        
                        time_sec = frame_idx / fps
                        writer.writerow([frame_idx, round(time_sec, 3), r_mean, g_mean, b_mean])

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        
                        cv2.circle(frame, (x_nose, y_nose), 1, (0, 0, 255), -1)
                        cv2.circle(frame, (x_ml, y_ml), 1, (0, 0, 255), -1)
                        cv2.circle(frame, (x_mr, y_mr), 1, (0, 0, 255), -1)
                else:
                    time_sec = frame_idx / fps
                    writer.writerow([frame_idx, round(time_sec, 3), None, None, None])

                out.write(frame)

                cv2.imshow('Pose ROI Preview (Press "q" to exit)', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
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


def process_directory(input_dir, output_dir):
    # 出力先フォルダが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"出力フォルダを作成しました: {output_dir}")

    # 入力フォルダ内のmp4ファイルをすべて取得 (必要に応じて拡張子を追加・変更)
    search_pattern = os.path.join(input_dir, "*.mp4")
    video_files = glob.glob(search_pattern)

    if not video_files:
        print(f"指定されたフォルダに .mp4 ファイルが見つかりません: {input_dir}")
        return

    print(f"合計 {len(video_files)} 件の動画ファイルを処理します。")
    print("プレビュー画面を選択した状態で 'q' キーを押すと、一括処理を途中終了できます。\n")

    for idx, video_path in enumerate(video_files, 1):
        # ファイル名と拡張子を分離して、出力ファイル名を作成
        basename = os.path.basename(video_path)
        name_without_ext, ext = os.path.splitext(basename)

        out_video = os.path.join(output_dir, f"{name_without_ext}_roi{ext}")
        out_csv = os.path.join(output_dir, f"{name_without_ext}_rgb.csv")

        print(f"[{idx}/{len(video_files)}] 処理中: {basename}")
        
        # 抽出関数の実行（戻り値で中断を判定）
        success = extract_nostril_rgb_with_pose(video_path, out_video, out_csv)
        
        if not success:
            print("一括処理を中断しました。")
            break
            
        print(f"  完了 -> 動画: {os.path.basename(out_video)}, CSV: {os.path.basename(out_csv)}\n")

    print("すべての処理が完了（または終了）しました。")

# ==========================================
# 実行部分
# ==========================================
if __name__ == "__main__":
   
    INPUT_DIR = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\20250114"
    
    OUTPUT_DIR = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\roi_out\20250114"

    process_directory(INPUT_DIR, OUTPUT_DIR)