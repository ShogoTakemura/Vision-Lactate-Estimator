import pandas as pd
import os

# ==========================================
# パス指定の箇所（ご自身の環境に合わせて変更してください）
# ==========================================
BASEFRAMES_PATH = r"C:\Users\ironm\Desktop\peak\results\BASEFRAMES.csv"
MODELBASED_PATH = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\MODELBASED_WORKSET.csv" # 実際のパスに変更してください
OUTPUT_PATH = r"C:\Users\ironm\squat_analyze\poseestimate_mediapipe\config\MODELBASED_WORKSET.csv" # 出力先のパス

def main():
    print("CSVファイルを読み込んでいます...")
    try:
        # CSVの読み込み
        df_base = pd.read_csv(BASEFRAMES_PATH, encoding='utf-8')
        df_model = pd.read_csv(MODELBASED_PATH, encoding='utf-8')
    except Exception as e:
        print(f"ファイルの読み込みに失敗しました: {e}")
        return

    # 1. マッチング用の共通キー（ファイル名のベース部分）を作成する
    # BASEFRAMES側: 末尾の "_results.csv" を取り除く
    df_base['match_key'] = df_base['filename'].str.replace('_results.csv', '', regex=False)

    # MODELBASED_WORKSET側: 末尾の "_correct" を取り除く
    df_model['match_key'] = df_model['filename'].str.replace('_correct', '', regex=False)

    # 2. BASEFRAMESから「マッチングキー」と「ID」の対応辞書を作成
    # { '20221102_kikuchi_sayaka-80per-10rep-set1' : 1, ... } のようなデータを作成
    id_mapping = df_base.set_index('match_key')['baseframe id'].to_dict()

    # 3. 辞書をもとにIDを自動記入
    # match_keyに対応するIDを探し、basedframe_id列に上書きする
    df_model['basedframe_id'] = df_model['match_key'].map(id_mapping)

    # （オプション）IDがNaN（見つからなかった場合）の確認
    missing_ids = df_model['basedframe_id'].isnull().sum()
    if missing_ids > 0:
        print(f"警告: マッチするIDが見つからなかった行が {missing_ids} 件あります。")

    # 4. 処理用に作ったmatch_keyカラムを削除して綺麗にする
    df_model = df_model.drop(columns=['match_key'])

    # 5. 更新したデータを新しいCSVとして保存
    try:
        df_model.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"処理が完了しました！\n出力先: {OUTPUT_PATH}")
    except Exception as e:
        print(f"ファイルの保存に失敗しました: {e}")

if __name__ == "__main__":
    main()