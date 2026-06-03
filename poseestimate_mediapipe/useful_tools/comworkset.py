import pandas as pd
import os

def update_workset_with_subject_data():
    # configディレクトリのパスを設定
    base_dir = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config"
    com_workset_path = os.path.join(base_dir, "COM_WORKSET.csv")
    subjects_data_path = os.path.join(base_dir, "SUBJECTS_DATA.csv")

    # CSVの読み込み
    try:
        com_df = pd.read_csv(com_workset_path)
        sub_df = pd.read_csv(subjects_data_path)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return
    except PermissionError:
        print("エラー: CSVファイルが開かれています。Excel等を閉じてから再実行してください。")
        return

    # SUBJECTS_DATA から必要な列（segment_id, load, mass）を抽出
    # マージしやすいように 'segment_id' を 'subject_id' に名前変更
    sub_subset = sub_df[['segment_id', 'load', 'mass']].rename(columns={'segment_id': 'subject_id'})

    # COM_WORKSET にすでに古い load, mass 列がある場合は一度削除（更新のため）
    if 'load' in com_df.columns:
        com_df = com_df.drop(columns=['load'])
    if 'mass' in com_df.columns:
        com_df = com_df.drop(columns=['mass'])

    # subject_id をキーにして left join（左外部結合）
    com_merged = pd.merge(com_df, sub_subset, on='subject_id', how='left')

    # 列の順番を元の COM_WORKSET.csv に合わせる
    desired_cols = ['id', 'filename', 'load', 'mass', 'subject_id', 'picklepath']
    final_cols = [col for col in desired_cols if col in com_merged.columns]
    com_merged = com_merged[final_cols]

    # 上書き保存 (BOM付きUTF-8)
    try:
        com_merged.to_csv(com_workset_path, index=False, encoding='utf-8-sig')
        print(f"更新完了: {com_workset_path} にSUBJECTS_DATAの load と mass を反映しました。")
    except PermissionError:
        print(f"エラー: 保存に失敗しました。{com_workset_path} をExcel等で開いていないか確認してください。")

if __name__ == "__main__":
    update_workset_with_subject_data()