"""Tests for melanoma.fairness.tone."""

import numpy as np
import pytest

from melanoma.fairness.tone import compute_ita, ita_group, lightness_group


def make_image(rgb, size=96):
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_light_skin_has_higher_ita_than_dark():
    light = compute_ita(make_image((230, 190, 170)))
    dark = compute_ita(make_image((90, 60, 45)))
    assert np.isfinite(light["ita"])
    assert np.isfinite(dark["ita"])
    assert light["ita"] > dark["ita"]


def test_ita_ignores_lesion_region():
    image = make_image((220, 180, 160))
    image[40:56, 40:56] = (20, 10, 10)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[40:56, 40:56] = 255
    result = compute_ita(image, mask)
    skin_only = compute_ita(make_image((220, 180, 160)))
    assert abs(result["ita"] - skin_only["ita"]) < 2.0
    assert result["n_pixels"] > 1000


def test_no_skin_pixels_returns_nan():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    result = compute_ita(image, mask, min_pixels=100)
    assert result["n_pixels"] == 0
    assert np.isnan(result["ita"])


def test_ita_group_thresholds():
    assert ita_group(60) == "light"
    assert ita_group(30) == "medium"
    assert ita_group(0) == "dark"
    assert ita_group(-20) == "dark"
    assert ita_group(float("nan")) == "unknown"


def test_lightness_group_thresholds():
    assert lightness_group(80) == "light"
    assert lightness_group(60) == "medium"
    assert lightness_group(50) == "dark"
    assert lightness_group(float("nan")) == "unknown"


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        compute_ita(np.zeros((10, 10), dtype=np.uint8))
