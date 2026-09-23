"""Tests for melanoma.data.transforms."""

import numpy as np
import pytest
import torch

from melanoma.data.transforms import build_transforms


def test_train_transform_output_shape_and_dtype():
    transform = build_transforms(64, train=True)
    output = transform(image=np.zeros((100, 80, 3), dtype=np.uint8))["image"]
    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 64, 64)
    assert output.dtype == torch.float32


def test_eval_transform_is_deterministic():
    transform = build_transforms(48, train=False)
    image = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    first = transform(image=image)["image"]
    second = transform(image=image)["image"]
    assert torch.equal(first, second)


def test_strong_train_transform_works():
    transform = build_transforms(32, train=True, strong=True)
    output = transform(image=np.zeros((50, 50, 3), dtype=np.uint8))["image"]
    assert output.shape == (3, 32, 32)


def test_square_and_non_square_inputs():
    transform = build_transforms(32, train=False)
    for shape in [(10, 10, 3), (128, 64, 3), (33, 77, 3)]:
        output = transform(image=np.zeros(shape, dtype=np.uint8))["image"]
        assert output.shape == (3, 32, 32)


@pytest.mark.parametrize("bad_size", [0, -5, "64", 10.5])
def test_invalid_img_size_raises(bad_size):
    with pytest.raises(ValueError):
        build_transforms(bad_size, train=True)
