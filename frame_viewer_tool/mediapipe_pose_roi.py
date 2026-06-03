import cv2
import mediapipe as mp

# ==========================================
# パスの指定
# ==========================================
input_path = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\20241119\honma_2set_70%RM.mp4"   # 読み込む動画ファイルのパス
output_path = r"C:\Users\ironm\squat_analyze\frame_viewer_tool\mediapipe_check\mediapipe_roi_out\honma_2set_70%RM_roi.mp4" # 保存する動画ファイルのパス

# MediaPipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 動画ファイルの読み込み
cap = cv2.VideoCapture(input_path)

# 動画のプロパティを取得（動画保存用）
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 動画保存用の設定 (MP4フォーマット)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ROIのサイズ設定（例: 400x400ピクセル）
roi_w, roi_h = 400, 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("動画の再生が終了したか、読み込めませんでした。")
        break

    # 1. 画面中央のROIの座標を計算
    x1 = (frame_width - roi_w) // 2
    y1 = (frame_height - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    # 2. ROI領域を切り出す
    roi_frame = frame[y1:y2, x1:x2]
    
    # MediaPipe用の色変換
    rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

    # 3. ROI画像に対してのみ骨格検出を実行
    results = pose.process(rgb_roi)

    # 4. 検出結果を描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            roi_frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    # 確認用に元の画像にROIの枠線を描画
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 動画ファイルにフレームを書き込む
    out.write(frame)

    # 画面にも表示（処理が重い場合はコメントアウト推奨）
    cv2.imshow("MediaPipe Pose - Center ROI (Video)", frame)
    if cv2.waitKey(1) & 0xFF == 27: # ESCキーで中断
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"処理が完了しました。結果は {output_path} に保存されています。")