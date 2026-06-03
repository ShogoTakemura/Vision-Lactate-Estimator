import pandas as pd

# ==========================================
# 1. ファイルパスの設定（環境に合わせて修正してください）
# ==========================================
# ワーク量データ
total_work_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\work_calculated\TOTAL_WORK_DATABASE.csv"

# ID紐付け用データ
modelbased_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\MODELBASED_WORKSET.csv" 

# 被験者データベース (★ここを SUBJECTS_DATA.csv のパスに変更します★)
subject_data_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\SUBJECTS_DATA.csv" 

# 出力先
output_csv = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\out\work_calculated\TOTAL_WORK_DATABASE_with_subjects.csv"

# ==========================================
# 2. データの読み込み
# ==========================================
df_work = pd.read_csv(total_work_csv)
df_model = pd.read_csv(modelbased_csv)
df_sub = pd.read_csv(subject_data_csv) # ここで SUBJECTS_DATA.csv を読み込む

# ==========================================
# 3. リレーション（ID紐付け）による結合
# ==========================================
# ① TOTAL_WORK_DATABASE と MODELBASED_WORKSET をファイル名で結合して subject_id を取得
df_merged1 = pd.merge(df_work, 
                      df_model[['filename', 'subject_id']], 
                      left_on='File_Name', 
                      right_on='filename', 
                      how='left')

# ② SUBJECTS_DATA の性別(sex)を 男性(M)=1, 女性(F)=0 に変換して gender 列を作成
# (ここでエラーが起きていました)
df_sub['gender'] = df_sub['sex'].map({'M': 1, 'F': 0})

# ③ 取得した subject_id と SUBJECTS_DATA の segment_id を使って、身長・体重・性別を結合
df_merged2 = pd.merge(df_merged1, 
                      df_sub[['segment_id', 'height', 'mass', 'gender', 'age']], 
                      left_on='subject_id', 
                      right_on='segment_id', 
                      how='left')

# ==========================================
# 4. データの整理と保存
# ==========================================
# 作業用に使った結合キー（filename, subject_id, segment_id）を削除
df_final = df_merged2.drop(columns=['filename', 'subject_id', 'segment_id'])

# CSVとして出力
df_final.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f"処理が完了しました。\n出力先: {output_csv}")