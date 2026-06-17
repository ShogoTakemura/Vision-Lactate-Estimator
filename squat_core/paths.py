"""squat_core/paths.py — プロジェクトルートの単一定義。

このファイルの場所から親ディレクトリを逆算するため、
リポジトリをどのパスに置いても正しく動く。

Usage:
    from squat_core.paths import PROJECT_ROOT, PKG_ROOT, FRAME_VIEWER_ROOT

    data_path = FRAME_VIEWER_ROOT / "roi_out" / "20241119"
    config_path = PKG_ROOT / "config" / "config.ini"
"""

from __future__ import annotations

from pathlib import Path

# squat_core/ の2階層上 = リポジトリルート
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# poseestimate_mediapipe パッケージルート
PKG_ROOT: Path = PROJECT_ROOT / "poseestimate_mediapipe"

# frame_viewer_tool (旧: リポジトリ直下、現在は poseestimate_mediapipe 配下に統合)
FRAME_VIEWER_ROOT: Path = PKG_ROOT / "frame_viewer_tool"
