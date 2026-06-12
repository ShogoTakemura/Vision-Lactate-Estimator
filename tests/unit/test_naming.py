"""Characterization tests for filename stem normalization (_stem_to_base)."""

from __future__ import annotations

import pytest

from poseestimate_mediapipe.process import calculate_work_process, rep_posture_analyze_process

CASES = [
    ("20221109_deguchi-80per-10rep-set1_correct", "20221109_deguchi-80per-10rep-set1"),
    ("20221109_deguchi-80per-10rep-set1_correct_modelbased", "20221109_deguchi-80per-10rep-set1"),
    ("20221109_deguchi-80per-10rep-set1_modelbased_correct", "20221109_deguchi-80per-10rep-set1"),
    ("foo_modelbase_analyzed_correct", "foo"),
    ("plain_name", "plain_name"),
    ("name_analyzed", "name"),
]


@pytest.mark.parametrize("stem,expected", CASES)
def test_stem_to_base_calculate_work(stem, expected) -> None:
    assert calculate_work_process._stem_to_base(stem) == expected


@pytest.mark.parametrize("stem,expected", CASES)
def test_stem_to_base_rep_posture(stem, expected) -> None:
    assert rep_posture_analyze_process._stem_to_base(stem) == expected


@pytest.mark.parametrize("stem,expected", CASES)
def test_stem_to_base_implementations_match(stem, expected) -> None:
    a = calculate_work_process._stem_to_base(stem)
    b = rep_posture_analyze_process._stem_to_base(stem)
    assert a == b == expected
