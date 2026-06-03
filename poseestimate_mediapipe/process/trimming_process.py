import os
import pandas as pd
from configparser import ConfigParser
from pathlib import Path

def process(config: ConfigParser):
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)
    
    # --- 設定：処理対象のフォルダリスト ---
    # 解析フェーズごとのフォルダをすべて指定します
    target_folders = [
        'out/correctpose',
        'out/modelbased',
        'out/surface',
        'out/bodycom',
        'out/partscom',
        'out/com_features'
    ]

    # BASEFRAMES.csv の読み込み
    baseframe_path = os.path.join(packagepath, 'config', config.get('modelbasecorrect', 'baseframe'))
    if not os.path.exists(baseframe_path):
        print(f"Error: {baseframe_path} が見つかりません。")
        return
    base_df = pd.read_csv(baseframe_path)

    for folder_rel_path in target_folders:
        target_dir = os.path.join(packagepath, folder_rel_path)
        if not os.path.exists(target_dir):
            continue

        # 出力先：元のフォルダ名に _trimmed を付与
        output_dir = os.path.join(packagepath, f"{folder_rel_path}_trimmed")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Processing Folder: {folder_rel_path} ---")

        csv_files = list(Path(target_dir).glob('*.csv'))
        for csv_path in csv_files:
            filename_full = csv_path.stem
            
            # BASEFRAMESのfilename列と部分一致するかチェック（修正箇所）
            match = base_df[base_df['filename'].apply(lambda x: str(x).replace('_results.csv', '') in filename_full)]
            
            if match.empty:
                # デバッグ用に、マッチしなかったファイル名を表示したい場合は以下をコメントアウト解除
                # print(f"Skipped (No match in BASEFRAMES): {filename_full}")
                continue

            start_f = int(match.iloc[0]['start'])
            end_f = int(match.iloc[0]['end'])

            try:
                # 1. 元のファイルのヘッダー（最初の3行）をそのまま取得
                with open(csv_path, 'r', encoding='utf-8') as f:
                    header_lines = [f.readline() for _ in range(3)]

                # 2. データ本体を読み込み、指定区間を抽出
                # header=0は3行読み飛ばした後の「座標名(x,y,z...)」の行を指す
                df = pd.read_csv(csv_path, skiprows=3)
                trimmed_df = df.iloc[start_f : end_f + 1]

                # 3. 書き出し（ヘッダー3行 + 抽出データ）
                out_path = os.path.join(output_dir, f"{filename_full}_trimmed.csv")
                with open(out_path, 'w', encoding='utf-8', newline='') as f:
                    f.writelines(header_lines) # 元のメタデータを書き込む
                    trimmed_df.to_csv(f, index=False) # 続けてデータを書き込む
                
                print(f"Saved: {filename_full}_trimmed.csv")

            except Exception as e:
                print(f"Failed {filename_full}: {e}")

    print("\n✅ すべてのフォルダのトリミングが完了しました。")