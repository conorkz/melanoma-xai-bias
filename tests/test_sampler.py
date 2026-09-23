"""Tests for melanoma.training.sampler."""

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from melanoma.training.sampler import (
    class_balanced_weights,
    compute_class_weights,
    make_weighted_sampler,
)


def test_weights_are_inverse_frequency():
    weights = class_balanced_weights([0, 0, 1])
    assert weights.tolist() == pytest.approx([0.5, 0.5, 1.0])


def test_single_class_labels_are_valid():
    weights = class_balanced_weights([0, 0, 0])
    assert weights.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_empty_labels_raise():
    with pytest.raises(ValueError):
        class_balanced_weights([])


def test_missing_middle_class_raises():
    with pytest.raises(ValueError, match="missing"):
        class_balanced_weights([0, 0, 2])


def test_make_sampler_returns_sampler():
    sampler = make_weighted_sampler([0, 1, 1])
    assert isinstance(sampler, WeightedRandomSampler)
    drawn = list(iter(sampler))
    assert len(drawn) == 3
    assert all(0 <= index < 3 for index in drawn)


def test_make_sampler_num_samples_override():
    sampler = make_weighted_sampler([0, 1] * 10, num_samples=5)
    assert len(list(iter(sampler))) == 5


@pytest.mark.parametrize("bad_num", [0, -1])
def test_invalid_num_samples_raises(bad_num):
    with pytest.raises(ValueError):
        make_weighted_sampler([0, 1], num_samples=bad_num)


def test_compute_class_weights_normalized():
    weights = compute_class_weights([0, 0, 0, 1])
    assert weights.tolist() == pytest.approx([0.5, 1.5])


def test_compute_class_weights_bad_mode_raises():
    with pytest.raises(ValueError):
        compute_class_weights([0, 1], mode="sqrt")


def test_compute_class_weights_missing_class_raises():
    with pytest.raises(ValueError):
        compute_class_weights([0, 0, 2])


def test_compute_class_weights_empty_raises():
    with pytest.raises(ValueError):
        compute_class_weights([])
