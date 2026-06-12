"""Verify that process-module re-exports resolve to the canonical squat_core implementations.

After Phase 3, calccomprocess / modelbasecorrect / functest no longer define their own
kinematics functions — they re-export from squat_core.kinematics.  These tests guard
against accidental re-introduction of local overrides.
"""

from __future__ import annotations

from poseestimate_mediapipe.process import calccomprocess, functest, modelbasecorrect
from squat_core import kinematics as _canonical

MODULES = [calccomprocess, modelbasecorrect, functest]
MODULE_IDS = ["calccomprocess", "modelbasecorrect", "functest"]

EXPORTS = [
    "calc_velocities",
    "calc_velocity",
    "calc_accels",
    "calc_accel",
    "calc_floorforces",
    "_calc_floorforce",
    "distribute_floorforce",
    "calc_dist_force",
]


def test_re_exports_are_canonical() -> None:
    """Every process module must re-export the identical function objects from squat_core."""
    for mod, mod_id in zip(MODULES, MODULE_IDS):
        for name in EXPORTS:
            assert getattr(mod, name) is getattr(_canonical, name), (
                f"{mod_id}.{name} is not squat_core.kinematics.{name} — "
                "a local override may have been re-introduced"
            )
