"""
poseestimate_mediapipe/config/constants.py
------------------------------------------
settings.toml をリポジトリルートから読み込み、プロジェクト全体で使う定数を提供する。

インポート例:
    from poseestimate_mediapipe.config.constants import GRAVITY, FPS, HR_BPF_LOW
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TOML_PATH = _REPO_ROOT / "settings.toml"


def _load_toml() -> dict:
    if _TOML_PATH.exists():
        with open(_TOML_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


_cfg   = _load_toml()
_phys  = _cfg.get("physics", {})
_sig   = _cfg.get("signal",  {})
_paths = _cfg.get("paths",   {})

# ── 物理定数 ────────────────────────────────────────────────
GRAVITY: float = _phys.get("gravity", 9.80665)  # [m/s²]
FPS:     float = _phys.get("fps",     30.0)      # [Hz]

# ── 心拍 (rPPG) 信号パラメータ ──────────────────────────────
HR_BPF_LOW:      float = _sig.get("hr_bpf_low",      0.8)
HR_BPF_HIGH:     float = _sig.get("hr_bpf_high",     2.0)
HR_POS_WINDOW_S: float = _sig.get("hr_pos_window_s", 1.6)

# ── パス ─────────────────────────────────────────────────────
DATA_ROOT:   str = os.environ.get("SQUAT_DATA_ROOT",   _paths.get("data_root",   ""))
OUTPUT_ROOT: str = os.environ.get("SQUAT_OUTPUT_ROOT", _paths.get("output_root", ""))
