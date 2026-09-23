"""Tests for melanoma.utils."""

import random

import numpy as np
import pytest
import torch

from melanoma.utils import ensure_dir, get_device, seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(123)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    seed_everything(123)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_seed_everything_different_seeds_differ():
    seed_everything(1)
    first = random.random()
    seed_everything(2)
    second = random.random()
    assert first != second


@pytest.mark.parametrize("bad_seed", [-1, "seed", None])
def test_seed_everything_invalid_raises(bad_seed):
    with pytest.raises(ValueError):
        seed_everything(bad_seed)


def test_get_device_cpu():
    assert get_device("cpu").type == "cpu"


def test_get_device_auto_returns_known_type():
    assert get_device("auto").type in {"cpu", "mps", "cuda"}


def test_get_device_mps_raises_when_unavailable():
    if not torch.backends.mps.is_available():
        with pytest.raises(ValueError):
            get_device("mps")


def test_get_device_cuda_raises_when_unavailable():
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            get_device("cuda")


def test_get_device_mps_available():
    if torch.backends.mps.is_available():
        assert get_device("mps").type == "mps"


def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result == target
    assert target.is_dir()
    assert ensure_dir(target) == target
