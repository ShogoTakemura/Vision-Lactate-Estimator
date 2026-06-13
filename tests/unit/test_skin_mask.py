"""Characterization tests for HSV+YCbCr skin masking."""

from __future__ import annotations

import numpy as np
from frame_viewer_tool.mediapipe_roi_face import skin_mask_hsv_ycbcr


def test_skin_mask_detects_skin_colored_pixel() -> None:
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[2, 2] = (120, 150, 200)  # BGR skin-like tone used during development

    mask = skin_mask_hsv_ycbcr(img)
    assert mask.shape == (5, 5)
    assert mask.dtype == np.uint8
    assert mask[2, 2] == 255
    assert mask.sum() == 255


def test_skin_mask_rejects_saturated_blue() -> None:
    img = np.full((3, 3, 3), (255, 0, 0), dtype=np.uint8)  # pure blue
    mask = skin_mask_hsv_ycbcr(img)
    assert mask.max() == 0


def test_skin_mask_requires_both_hsv_and_ycbcr() -> None:
    # Mid-gray passes Y but should fail HSV S/V constraints for skin rules
    img = np.full((3, 3, 3), (128, 128, 128), dtype=np.uint8)
    mask = skin_mask_hsv_ycbcr(img)
    assert mask.max() == 0
