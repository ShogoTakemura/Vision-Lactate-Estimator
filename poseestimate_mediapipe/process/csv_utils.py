"""
csv_utils.py
------------
プロジェクト共通のCSV読み込みユーティリティ。

このプロジェクトのCSVフォーマット:
    行1: メタデータ (例: subject名, FPS, 作成日時 ...)
    行2: 空行
    行3: ヘッダー (列名)
    行4以降: データ

→ pd.read_csv(..., skiprows=2) で読み込む。
"""

import pandas as pd
from pathlib import Path


def read_pose_csv(csv_path: str | Path) -> tuple[list[str], pd.DataFrame]:
    """
    姿勢推定CSVを読み込む。

    Parameters
    ----------
    csv_path : str | Path
        読み込むCSVファイルのパス

    Returns
    -------
    header_lines : list[str]
        先頭2行のメタデータ文字列リスト（保存時に再付与するために使う）
    df : pd.DataFrame
        3行目をヘッダーとして読み込んだDataFrame（列名の空白はstrip済み）
    """
    csv_path = Path(csv_path)

    with open(csv_path, 'r', encoding='utf-8') as f:
        header_lines = [f.readline() for _ in range(2)]

    df = pd.read_csv(csv_path, skiprows=2)
    df.columns = df.columns.str.strip()

    return header_lines, df


def write_pose_csv(out_path: str | Path, header_lines: list[str], df: pd.DataFrame) -> None:
    """
    姿勢推定CSVを保存する（メタデータ行を先頭に再付与）。

    Parameters
    ----------
    out_path : str | Path
        保存先パス
    header_lines : list[str]
        read_pose_csv で取得したメタデータ行リスト
    df : pd.DataFrame
        保存するDataFrame
    """
    out_path = Path(out_path)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.writelines(header_lines)
        df.to_csv(f, index=False)