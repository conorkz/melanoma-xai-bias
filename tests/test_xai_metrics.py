"""Tests for melanoma.xai.metrics."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from melanoma.xai.metrics import (
    attribution_iou,
    attribution_mass_in_mask,
    deletion_insertion_auc,
    pointing_game,
)


def test_attribution_mass_all_inside():
    heatmap = np.array([[0.0, 0.0], [0.0, 1.0]])
    mask = np.array([[0, 0], [0, 1]])
    assert attribution_mass_in_mask(heatmap, mask) == 1.0


def test_attribution_mass_all_outside():
    heatmap = np.array([[1.0, 0.0], [0.0, 0.0]])
    mask = np.array([[0, 0], [0, 1]])
    assert attribution_mass_in_mask(heatmap, mask) == 0.0


def test_attribution_mass_zero_heatmap_is_nan():
    heatmap = np.zeros((4, 4))
    mask = np.ones((4, 4))
    assert np.isnan(attribution_mass_in_mask(heatmap, mask))


def test_pointing_game_inside_and_outside():
    heatmap = np.zeros((4, 4))
    heatmap[3, 3] = 1.0
    inside = np.zeros((4, 4), dtype=np.uint8)
    inside[3, 3] = 1
    outside = np.zeros((4, 4), dtype=np.uint8)
    outside[0, 0] = 1
    assert pointing_game(heatmap, inside) is True
    assert pointing_game(heatmap, outside) is False


def test_pointing_game_zero_heatmap_is_nan():
    assert np.isnan(pointing_game(np.zeros((4, 4)), np.ones((4, 4))))


def test_attribution_iou_perfect_and_zero():
    heatmap = np.arange(16, dtype=float).reshape(4, 4)
    perfect = (heatmap >= 12).astype(np.uint8)
    zero = (heatmap < 4).astype(np.uint8)
    assert attribution_iou(heatmap, perfect, fraction=0.25) == 1.0
    assert attribution_iou(heatmap, zero, fraction=0.25) == 0.0


def test_attribution_iou_invalid_fraction():
    with pytest.raises(ValueError):
        attribution_iou(np.zeros((4, 4)), np.zeros((4, 4)), fraction=1.5)


class DummyModel(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        return torch.cat([torch.zeros_like(mean), mean], dim=1)


def test_deletion_insertion_returns_finite_aucs():
    model = DummyModel()
    image = torch.randn(1, 3, 8, 8)
    heatmap = np.random.default_rng(0).random((8, 8))
    result = deletion_insertion_auc(model, image, heatmap, target_class=1, steps=4)
    assert len(result["fractions"]) == 5
    assert len(result["deletion"]) == 5
    assert len(result["insertion"]) == 5
    assert np.isfinite(result["deletion_auc"])
    assert np.isfinite(result["insertion_auc"])


def test_deletion_insertion_rejects_bad_shapes():
    model = DummyModel()
    with pytest.raises(ValueError):
        deletion_insertion_auc(model, torch.randn(2, 3, 8, 8), np.zeros((8, 8)), 1)
    with pytest.raises(ValueError):
        deletion_insertion_auc(model, torch.randn(1, 3, 8, 8), np.zeros((4, 4)), 1)
