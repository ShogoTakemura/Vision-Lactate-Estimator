"""squat_core/subjects.py — 被験者情報ローダ。

functest.py / modelbasecorrect.py の generate_subjectinfo 重複を統合した単一実装。
エンコーディングは utf-8-sig（BOM 付き UTF-8）に統一。
"""

from __future__ import annotations

import csv


def load_subjects(csv_path: str) -> list[dict[str, str]]:
    """被験者情報 CSV を読み込み、列名をキーとする辞書のリストを返す。

    Parameters
    ----------
    csv_path : 被験者情報 CSV のパス（utf-8-sig / utf-8 どちらも可）

    Returns
    -------
    list[dict[str, str]] : 被験者ごとの {列名: 値} 辞書のリスト（ヘッダ行は除く）
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]
    return [
        {name: value for name, value in zip(header, row, strict=False)}
        for row in rows[1:]
    ]
