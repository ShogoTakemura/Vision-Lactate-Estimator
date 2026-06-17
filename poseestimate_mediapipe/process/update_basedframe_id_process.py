"""update_basedframe_id_process.py
-----------------------------------
BASEFRAMES.csv の baseframe id を MODELBASED_WORKSET.csv の basedframe_id 列へ
自動反映する処理。

Workflowにおける位置づけ:
    Step 1: bagfile -> 動画書き出し
    Step 2: 手動でレップ区間を切り出し (frame_viewer_tool, 対話メニュー対象外)
    Step 3: 3D姿勢推定
    Step 4: basedframe_id の自動更新 (このモジュール)
    Step 5: model based correct pose

以前は同じロジックが useful_tools/modelbasedworkset.py (パスがハードコードされ
壊れていた) と process/auto_pipeline.py の Step1 に重複していたため、こちらに
一本化した。auto_pipeline.py からも本モジュールの update_basedframe_id() を
呼び出す。
"""

import os
from configparser import ConfigParser

import pandas as pd


def process(config: ConfigParser) -> None:
    processpath = os.path.dirname(os.path.abspath(__file__))
    packagepath = os.path.dirname(processpath)

    baseframe_path = os.path.join(
        packagepath, 'config', config.get('modelbasecorrect', 'baseframe'))
    modelbased_ws_path = os.path.join(
        packagepath, 'config', config.get('modelbasecorrect', 'correctworkset'))

    update_basedframe_id(baseframe_path, modelbased_ws_path)


def update_basedframe_id(baseframe_path: str, modelbased_ws_path: str) -> None:
    """BASEFRAMES.csv の baseframe id を MODELBASED_WORKSET.csv へ反映する。

    Args:
        baseframe_path: BASEFRAMES.csv のパス。
        modelbased_ws_path: MODELBASED_WORKSET.csv のパス(更新後、同じ場所に上書き保存する)。
    """
    if not os.path.exists(baseframe_path):
        print(f"  [ERROR] BASEFRAMES.csv が見つかりません: {baseframe_path}")
        return
    if not os.path.exists(modelbased_ws_path):
        print(f"  [ERROR] MODELBASED_WORKSET.csv が見つかりません: {modelbased_ws_path}")
        return

    df_base = pd.read_csv(baseframe_path, encoding='utf-8-sig')
    df_model = pd.read_csv(modelbased_ws_path, encoding='utf-8-sig')

    # BASEFRAMES: filename から "_rep.csv" を除去してキー生成
    df_base['match_key'] = df_base['filename'].str.replace('_rep.csv', '', regex=False)

    # MODELBASED_WORKSET: filename から "_correct" を除去してキー生成
    df_model['match_key'] = df_model['filename'].str.replace('_correct', '', regex=False)

    id_map = df_base.set_index('match_key')['baseframe id'].to_dict()
    df_model['basedframe_id'] = df_model['match_key'].map(id_map)

    miss = df_model['basedframe_id'].isna().sum()
    if miss > 0:
        print(f"  [WARNING] basedframe_id が見つからない行: {miss} 件")
        print(df_model[df_model['basedframe_id'].isna()][['filename']].to_string())

    df_model = df_model.drop(columns=['match_key'])
    df_model.to_csv(modelbased_ws_path, index=False, encoding='utf-8-sig')
    print(f"  -> {len(df_model)}件更新完了: {modelbased_ws_path}")
