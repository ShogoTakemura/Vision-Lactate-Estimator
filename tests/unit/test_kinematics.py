"""Characterization tests for kinematics helpers in calccomprocess."""

from __future__ import annotations

import pytest

from poseestimate_mediapipe.process import calccomprocess as kin

# Fixed inputs chosen so central-difference results are exact rationals.
COM_Y = [0.0, 1.0, 4.0, 9.0, 16.0]
FPS = 30.0


@pytest.mark.parametrize(
    ("comlist", "fps", "expected"),
    [
        (COM_Y, FPS, (0.0, 60.0, 120.0, 180.0, 0.0)),
        ([5.0], 30.0, (0.0,)),
        ([1.0, 3.0], 10.0, (0.0, 0.0)),
    ],
)
def test_calc_velocities(comlist, fps, expected) -> None:
    assert kin.calc_velocities(comlist, fps) == expected


@pytest.mark.parametrize(
    ("comlist", "fps", "expected"),
    [
        (COM_Y, FPS, (0.0, 1800.0, 1800.0, 1800.0, 0.0)),
        ([5.0], 30.0, (0.0,)),
    ],
)
def test_calc_accels(comlist, fps, expected) -> None:
    assert kin.calc_accels(comlist, fps) == expected


def test_calc_velocity_centered_difference() -> None:
    comlist = [0.0, 0.0, 3.0]
    timespan = 1.0 / FPS
    assert kin.calc_velocity(comlist, 1, timespan) == pytest.approx(45.0)


def test_calc_accel_centered_second_difference() -> None:
    timespan = 1.0 / FPS
    assert kin.calc_accel(4.0, 1.0, 0.0, timespan) == pytest.approx(1800.0)


def test_calc_floorforces_and_inertial_chain() -> None:
    inertial = [0.0, 10.0, -5.0]
    mass = 70.0
    gravity = 9.79

    floor = kin.calc_floorforces(inertial, mass, gravity)
    assert floor == (685.3, 675.3, 690.3)

    accels = kin.calc_accels(COM_Y, FPS)
    inertial_from_accel = kin.calc_inertialforces(accels, mass)
    recomputed_floor = kin.calc_floorforces(inertial_from_accel, mass, gravity)
    assert len(recomputed_floor) == len(COM_Y)
    assert recomputed_floor[1] == pytest.approx(mass * gravity - mass * accels[1])


@pytest.mark.parametrize(
    ("l_x", "r_x", "com_x", "force", "expected"),
    [
        (0.0, 1.0, 0.25, 100.0, (75.0, 25.0)),
        (0.0, 1.0, 0.75, 100.0, (25.0, 75.0)),
    ],
)
def test_calc_dist_force(l_x, r_x, com_x, force, expected) -> None:
    assert kin.calc_dist_force(l_x, r_x, com_x, force) == expected


def test_calc_dist_force_zero_width_raises() -> None:
    with pytest.raises(ZeroDivisionError):
        kin.calc_dist_force(0.0, 0.0, 0.0, 100.0)


def test_distribute_floorforce_batch() -> None:
    r_com = (0.0, 1.0)
    l_com = (0.0, 0.0)
    com = (0.25, 0.75)
    forces = (100.0, 200.0)

    out = kin.distribute_floorforce(r_com, l_com, com, forces)
    assert out == [(50.0, 50.0), (50.0, 150.0)]


def test_parse_axis_from_comvector() -> None:
    comlist = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
    xs, ys, zs = kin.parse_axis_from_comvector(comlist)
    assert xs == (1.0, 4.0)
    assert ys == (2.0, 5.0)
    assert zs == (3.0, 6.0)


def test_copcalculator_calc_accels_matches_calccomprocess() -> None:
    from poseestimate_mediapipe.module.com.copcalculator import calc_accels as cop_accels

    assert cop_accels(COM_Y, FPS) == kin.calc_accels(COM_Y, FPS)
