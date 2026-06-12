"""Ensure duplicated kinematics implementations stay identical."""

from __future__ import annotations

import pytest

from poseestimate_mediapipe.process import calccomprocess, functest, modelbasecorrect
from poseestimate_mediapipe.config.constants import GRAVITY

COM_Y = [0.0, 1.0, 4.0, 9.0, 16.0]
FPS = 30.0
INERTIAL = [0.0, 10.0, -5.0]
MASS = 70.0

IMPLS = [
    pytest.param(calccomprocess, id="calccomprocess"),
    pytest.param(modelbasecorrect, id="modelbasecorrect"),
    pytest.param(functest, id="functest"),
]


@pytest.mark.parametrize("mod", IMPLS)
def test_calc_velocities_match(mod) -> None:
    ref = calccomprocess.calc_velocities(COM_Y, FPS)
    assert mod.calc_velocities(COM_Y, FPS) == ref


@pytest.mark.parametrize("mod", IMPLS)
def test_calc_accels_match(mod) -> None:
    ref = calccomprocess.calc_accels(COM_Y, FPS)
    assert mod.calc_accels(COM_Y, FPS) == ref


@pytest.mark.parametrize("mod", IMPLS)
def test_calc_floorforces_match(mod) -> None:
    ref = calccomprocess.calc_floorforces(INERTIAL, MASS, GRAVITY)
    assert mod.calc_floorforces(INERTIAL, MASS, GRAVITY) == ref


@pytest.mark.parametrize("mod", IMPLS)
def test_distribute_floorforce_match(mod) -> None:
    args = ((0.0, 1.0), (0.0, 0.0), (0.25, 0.75), (100.0, 200.0))
    ref = calccomprocess.distribute_floorforce(*args)
    assert mod.distribute_floorforce(*args) == ref
