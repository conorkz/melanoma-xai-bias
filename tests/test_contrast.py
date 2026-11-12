"""Tests for lesion-skin contrast features."""

import numpy as np
import pytest

from melanoma.fairness.contrast import lesion_contrast


def synthetic_pair():
    image = np.full((64, 64, 3), (205, 175, 150), dtype=np.uint8)
    image[22:42, 22:42] = (80, 50, 40)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[22:42, 22:42] = 1
    return image, mask


def test_dark_lesion_has_positive_contrast():
    image, mask = synthetic_pair()
    result = lesion_contrast(image, mask)
    assert result["delta_L"] > 20
    assert result["delta_E"] > 20
    assert result["lesion_area"] == 400
    assert result["skin_pixels"] > 0
    assert result["lesion_L"] < result["skin_L"]


def test_mask_is_resized_to_image_shape():
    image, mask = synthetic_pair()
    small_mask = mask[::2, ::2]
    result = lesion_contrast(image, small_mask)
    assert result["lesion_area"] == pytest.approx(400, rel=0.15)


def test_empty_mask_returns_nan():
    image, _ = synthetic_pair()
    empty = np.zeros((64, 64), dtype=np.uint8)
    result = lesion_contrast(image, empty)
    assert np.isnan(result["delta_L"])
    assert result["lesion_area"] == 0


def test_ring_excludes_dark_artifacts():
    image, mask = synthetic_pair()
    image[:5, :] = (0, 0, 0)
    result = lesion_contrast(image, mask)
    assert result["skin_L"] > 40
    assert result["skin_pixels"] > 0
