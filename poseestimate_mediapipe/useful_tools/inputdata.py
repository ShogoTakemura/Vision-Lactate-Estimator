import pandas as pd
import re

# ==========================================
# 1. ファイルパスの設定（環境に合わせて修正してください）
# ==========================================
# 読み込む入力ファイル
input_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\work_calculated\input_database.csv"

# 出力するファイル名（同じフォルダに別名で保存します）
output_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\work_calculated\input_database_with_set.csv"

# ==========================================
# 2. データの読み込み
# ==========================================
df = pd.read_csv(input_csv)

# ==========================================
# 3. set番号の抽出処理
# ==========================================
def extract_set(filename):
    # ファイル名から "set" に続く数字を抽出（例: "set1" -> "1"）
    match = re.search(r'set(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return None

# 'File_Name' 列に対して関数を適用し、'set' 列を更新/追加
df['set'] = df['File_Name'].apply(extract_set)

# ==========================================
# 4. データの保存
# ==========================================
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"処理が完了しました。")
print(f"セット番号を抽出して追加しました。\n出力先: {output_csv}")