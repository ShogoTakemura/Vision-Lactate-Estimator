import cv2
import mediapipe as mp
import os

# --- 設定項目 ---
# 1. 解析したい元動画のパスを指定してください
video_path = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\squat02.mp4"  # 例: 実際の動画パスに変更

# 2. 確認用動画の出力先フォルダとファイル名
output_dir = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\frame_viewer_tool\mediapipe_check\out"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "squat_02_mediapipe_skeleton_check.mp4")
# ----------------

# MediaPipe Pose の初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# 骨格全体の描画スタイル（関節点と線の色や太さ）
mp_drawing_styles = mp.solutions.drawing_styles

# 動画ファイルの読み込み
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"エラー: 動画ファイルを開けませんでした。パスを確認してください: {video_path}")
    exit()

# 元動画のプロパティ（幅、高さ、FPS）を取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 保存用ビデオライターの設定 (MP4形式で保存するための主要なコーデック 'mp4v' を使用)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"動画の解析を開始します（解像度: {frame_width}x{frame_height}, FPS: {fps}）...")
print("※ 画面表示中に『 q 』キーを押すと途中で終了できます。")

# MediaPipeのコンテキストを開く
with mp_pose.Pose(
    static_image_mode=False,       # 動画（時系列）のトラッキングモードを有効化
    model_complexity=1,            # モデルの複雑さ（0:高速, 1:標準, 2:高精度。スクワット解析なら1か2を推奨）
    enable_segmentation=False,     # 背景透過（セグメンテーション）は不要なのでFalse
    min_detection_confidence=0.5,  # 人体検出の最低信頼度
    min_tracking_confidence=0.5    # 骨格追跡の最低信頼度
) as pose:

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("動画の終端に達したか、フレームの読み込みに失敗しました。")
            break

        frame_count += 1

        # MediaPipeはRGB画像を処理するため、OpenCVのBGRから変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 骨格検出を実行
        results = pose.process(frame_rgb)

        # 検出された関節点がある場合、元のフレーム（BGR）に骨格を上書き描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,  # 関節同士を繋ぐ線の定義
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        else:
            # 骨格が隠れたりして見失った場合に画面上で把握できるよう警告テキストを入れる
            cv2.putText(frame, "Pose Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # フレーム数カウントを表示（同期のデバッグ時に便利です）
        cv2.putText(frame, f"Frame: {frame_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 処理されたフレームを確認用動画として書き込み
        out.write(frame)

        # リアルタイムで画面に表示（不要な場合は以下の3行をコメントアウトしてください）
        cv2.imshow('MediaPipe Skeleton Check', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("ユーザーによって処理が中断されました。")
            break

# 後片付け
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n[完了] 確認用動画を保存しました:\n--> {output_path}")