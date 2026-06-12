"""Characterization tests for HR / rPPG signal processing (hr_estimation.py)."""

from __future__ import annotations

import numpy as np
import pytest


def _synthetic_rgb(fps: float = 30.0, seconds: float = 4.0, hr_hz: float = 1.2) -> np.ndarray:
    t = np.arange(int(fps * seconds)) / fps
    phase = 2.0 * np.pi * hr_hz * t
    return np.column_stack(
        [
            128.0 + 10.0 * np.sin(phase),
            128.0 + 8.0 * np.sin(phase + 0.1),
            128.0 + 6.0 * np.sin(phase + 0.2),
        ]
    )


def test_pos_rppg_characteristic_values(hr_estimation) -> None:
    rgb = _synthetic_rgb()
    h = hr_estimation.pos_rppg(rgb, 30.0)

    assert h.shape == (len(rgb),)
    assert h[:5] == pytest.approx(
        [
            -0.0010227936460131233,
            -0.001071264213370596,
            -0.0010565013820125443,
            -0.0009809779865947482,
            -0.0008495984920276693,
        ],
        rel=0,
        abs=1e-12,
    )
    assert float(np.round(h, 6).sum()) == pytest.approx(0.002105, rel=0, abs=1e-9)


def test_bandpass_filter_preserves_length(hr_estimation) -> None:
    rgb = _synthetic_rgb(seconds=6.0)
    raw = hr_estimation.pos_rppg(rgb, 30.0)
    filtered = hr_estimation.bandpass_filter(raw, 30.0)

    assert filtered.shape == raw.shape
    assert np.isfinite(filtered).all()
    assert float(np.std(filtered)) > 0.0


def test_welch_peak_freq_near_expected_hr(hr_estimation) -> None:
    fps = 30.0
    hr_hz = 1.2
    rgb = _synthetic_rgb(fps=fps, seconds=20.0, hr_hz=hr_hz)
    raw = hr_estimation.pos_rppg(rgb, fps)
    filtered = hr_estimation.bandpass_filter(raw, fps)

    peak_freq, freqs, psd = hr_estimation.welch_peak_freq(filtered, fps)
    assert peak_freq == pytest.approx(hr_hz, abs=0.05)
    assert len(freqs) == len(psd)
    assert psd.max() > 0.0


def test_detect_rri_from_synthetic_pulse(hr_estimation) -> None:
    fps = 30.0
    hr_hz = 1.2
    rgb = _synthetic_rgb(fps=fps, seconds=30.0, hr_hz=hr_hz)
    raw = hr_estimation.pos_rppg(rgb, fps)
    filtered = hr_estimation.bandpass_filter(raw, fps)
    peak_freq, _, _ = hr_estimation.welch_peak_freq(filtered, fps)

    rri_clean, peak_times = hr_estimation.detect_rri(filtered, fps, peak_freq)
    assert len(peak_times) >= 2
    if len(rri_clean) >= 1:
        assert np.all(rri_clean > 0.0)
        assert float(rri_clean.mean()) == pytest.approx(1.0 / hr_hz, rel=0.15)


def test_hr_rep_analysis_pos_rppg_matches_hr_estimation(hr_estimation) -> None:
    """Both pos_rppg implementations use window=1.6 s (48 frames at 30 Hz) and are identical.
    Phase 3 will merge them into a single function in squat_core/signal/.
    """
    from tests.conftest import ROOT, load_module_from_path, scipy_signal_available

    if not scipy_signal_available():
        pytest.skip("scipy.signal unavailable")

    hr_rep = load_module_from_path(
        "hr_rep_analysis",
        ROOT / "frame_viewer_tool" / "hr_rep_analysis.py",
    )
    rgb = _synthetic_rgb(seconds=3.0)
    h_main = hr_estimation.pos_rppg(rgb, 30.0)
    h_rep = hr_rep.pos_rppg(rgb, 30.0)

    # Both implementations use POS_WIN_SEC=1.6 → identical output.
    # Safe to unify in Phase 3.
    assert np.allclose(h_main, h_rep)
