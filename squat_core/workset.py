"""squat_core/workset.py — WORKSET CSV の検証ユーティリティ。

calccomprocess.py / modelbasecorrect.py の check_worksetcol 重複を統合した単一実装。
"""

from __future__ import annotations


def check_worksetcol(col_index: int, row: tuple[str, ...], label: str = "") -> None:
    """WORKSET の指定列が空でないことを確認する。空なら即終了する。

    Parameters
    ----------
    col_index : チェックする列番号（0始まり）
    row       : WORKSET の1行データ
    label     : エラーメッセージに付加するファイル名等（省略可）
    """
    if not row[col_index]:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}{col_index + 1} 列目が記載されていません")
        raise SystemExit(0)
