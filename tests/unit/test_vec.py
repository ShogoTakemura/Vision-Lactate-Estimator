"""Characterization tests for vec2 / vec3 helpers."""

from __future__ import annotations

import pytest

from poseestimate_mediapipe.module.vec2 import vec2 as v2
from poseestimate_mediapipe.module.vec3 import vec3 as v3


def test_vec2_distance_and_direction() -> None:
    a = (0.0, 0.0)
    b = (3.0, 4.0)
    assert v2.distance(a, b) == pytest.approx(5.0)
    assert v2.direc(a, b) == (3.0, 4.0)


def test_vec2_unit_and_normal() -> None:
    vec = (3.0, 4.0)
    assert v2.magni(vec) == pytest.approx(5.0)
    unit = v2.unit(vec)
    assert unit == pytest.approx((0.6, 0.8))
    assert v2.normal(unit) == pytest.approx((-0.8, 0.6))


def test_vec2_add_average_inner_divide() -> None:
    first = (0.0, 2.0)
    second = (2.0, 0.0)
    assert v2.add(first, second) == (2.0, 2.0)
    assert v2.average(first, second) == (1.0, 1.0)
    assert v2.inner_divine(first, second, 1.0, 1.0) == (1.0, 1.0)
    assert v2.inner_divine((0.0, 0.0), (10.0, 0.0), 1.0, 3.0) == (2.5, 0.0)


def test_vec3_operations() -> None:
    a = (1.0, 0.0, 0.0)
    b = (0.0, 1.0, 0.0)
    assert v3.direc_vec3(a, b) == (-1.0, 1.0, 0.0)
    assert v3.add_vec3(a, b) == (1.0, 1.0, 0.0)
    assert v3.ave_vec3(a, b) == (0.5, 0.5, 0.0)
    assert v3.magni_vec3((3.0, 0.0, 4.0)) == pytest.approx(5.0)
    assert v3.unit_vec3((3.0, 0.0, 4.0)) == pytest.approx((0.6, 0.0, 0.8))
    assert v3.multi_vec3(a, 2.0) == (2.0, 0.0, 0.0)
