# ==============================================================================
# 動画からレップの抽出を行うスクリプト：Todo レップ数と動画フレームの未一致を修正する
# ==============================================================================
import cv2
import csv
import os

# ==============================================================================
# 設定
# ==============================================================================
video_path = r"C:\Users\ironm\squat_program\python\githubrepos\frame_viewer_tool\out\yano\20231024_tanigawa failure set1.avi"  # 解析したい動画のパスを指定
csv_path = r"C:\Users\ironm\Desktop\tanigawaset1.csv"   # 保存するCSVのパス

# ==============================================================================
# 初期化
# ==============================================================================
cap = cv2.getBuildInformation() # ダミー呼び出し（OpenCVの確認用）
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print(f"エラー: 動画ファイル {video_path} を開けませんでした。")
    exit()

current_frame = 0
rep_list = []
current_rep = {"rep": 1, "S": None, "e": None} # bを削除

print("==================================================")
print(" 操作方法:")
print("  [→] または [d] : 1フレーム進む")
print("  [←] または [a] : 1フレーム戻る")
print("  [↑]           : 10フレーム進む (スキップ用)")
print("  [↓]           : 10フレーム戻る (スキップ用)")
print("--------------------------------------------------")
print("  [s] キー       : 現在のフレームを Start (S) として記録")
print("  [e] キー       : 現在のフレームを End (e) として記録")
print("--------------------------------------------------")
print("  [Enter]       : 現在のレップ(S, e)を確定して次のレップへ")
print("  [r] キー       : 現在のレップの記録をリセット")
print("  [q] キー       : 終了してCSVに保存")
print("==================================================")

window_name = "Rep Annotation Tool (S/E Only)"
cv2.namedWindow(window_name)

while True:
    # 指定フレームに移動して読み込み
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    
    if not ret:
        current_frame = max(0, current_frame - 1)
        continue

    # 画面表示用のイメージを作成（情報オーバーレイ用）
    display_frame = frame.copy()
    h, w, _ = display_frame.shape
    
    # 画面に現在のステータスを描画 (b を除外)
    info_text = f"Frame: {current_frame}/{total_frames} | Current Rep: {current_rep['rep']}"
    status_text = f"S: {current_rep['S']} | e: {current_rep['e']}"
    
    cv2.putText(display_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, "Enter: Confirm Rep | q: Save & Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(window_name, display_frame)
    
    # 拡張キーコードを取得 (矢印キー対応)
    key = cv2.waitKeyEx(0)
    
    # --- フレーム移動操作 ---
    if key in [2555904, 65363, ord('d')]:  # 右矢印 / d
        if current_frame < total_frames - 1:
            current_frame += 1
    elif key in [2424832, 65361, ord('a')]:  # 左矢印 / a
        if current_frame > 0:
            current_frame -= 1
    elif key in [2490368, 65362]:  # 上矢印 (10歩進む)
        current_frame = min(total_frames - 1, current_frame + 10)
    elif key in [2621440, 65364]:  # 下矢印 (10戻る)
        current_frame = max(0, current_frame - 10)
        
    # --- イベント記録操作 ---
    elif key in [ord('s'), ord('S')]:
        current_rep["S"] = current_frame
        print(f"Rep {current_rep['rep']} - S: {current_frame}")
    elif key in [ord('e'), ord('E')]:
        current_rep["e"] = current_frame
        print(f"Rep {current_rep['rep']} - e: {current_frame}")
        
    # --- レップ確定 ---
    elif key in [13, 10]:  # Enter キー
        if current_rep["S"] is not None and current_rep["e"] is not None:
            rep_list.append(current_rep.copy())
            print(f"【確定】Rep {current_rep['rep']} -> S:{current_rep['S']}, e:{current_rep['e']}\n")
            # 次のレップの準備
            current_rep["rep"] += 1
            current_rep["S"], current_rep["e"] = None, None
        else:
            print("警告: S と e の両方のポイントを打ってからEnterを押してください。")
            
    # --- リセット ---
    elif key in [ord('r'), ord('R')]:
        current_rep["S"], current_rep["e"] = None, None
        print("現在のレップの入力をクリアしました。")
        
    # --- 終了 ---
    elif key in [ord('q'), ord('Q')]:
        break

cap.release()
cv2.destroyAllWindows()

# ==============================================================================
# CSV出力処理
# ==============================================================================
if rep_list:
    # パターンA: 1行に1レップをまとめる形式（bottom_frameを空白にする）
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rep', 'start_frame', 'bottom_frame', 'end_frame'])
        for r in rep_list:
            # 3番目の要素（bottom_frame）を空文字 "" にしています
            writer.writerow([r['rep'], r['S'], "", r['e']])
    print(f"\nデータを確認しました。保存完了: {csv_path}")
else:
    print("\n保存されたデータはありません。")