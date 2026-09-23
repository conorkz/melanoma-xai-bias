"""Tests for melanoma.training.losses."""

import math

import pytest
import torch
import torch.nn.functional as F

from melanoma.training.losses import FocalLoss


def test_gamma_zero_equals_cross_entropy():
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss = FocalLoss(gamma=0.0)(logits, targets)
    assert torch.allclose(loss, F.cross_entropy(logits, targets), atol=1e-6)


def test_hand_computed_value():
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([0])
    loss = FocalLoss(gamma=1.0)(logits, targets)
    assert loss.item() == pytest.approx(0.5 * math.log(2), rel=1e-5)


def test_perfect_predictions_give_near_zero_loss():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.item() < 1e-3


def test_scalar_alpha_scales_positive_class():
    logits = torch.randn(16, 2)
    targets = torch.ones(16, dtype=torch.long)
    base = FocalLoss(gamma=2.0)(logits, targets)
    weighted = FocalLoss(gamma=2.0, alpha=0.25)(logits, targets)
    assert weighted.item() == pytest.approx(0.25 * base.item(), rel=1e-5)


def test_tensor_alpha_applies_per_class():
    logits = torch.randn(16, 2)
    alpha = [0.9, 0.1]

    zeros = torch.zeros(16, dtype=torch.long)
    base_zero = FocalLoss(gamma=2.0)(logits, zeros)
    weighted_zero = FocalLoss(gamma=2.0, alpha=alpha)(logits, zeros)
    assert weighted_zero.item() == pytest.approx(0.9 * base_zero.item(), rel=1e-5)

    ones = torch.ones(16, dtype=torch.long)
    base_one = FocalLoss(gamma=2.0)(logits, ones)
    weighted_one = FocalLoss(gamma=2.0, alpha=alpha)(logits, ones)
    assert weighted_one.item() == pytest.approx(0.1 * base_one.item(), rel=1e-5)


def test_alpha_wrong_length_raises():
    logits = torch.randn(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="alpha"):
        FocalLoss(alpha=[0.5])(logits, targets)


def test_scalar_alpha_multiclass_raises():
    logits = torch.randn(4, 3)
    targets = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError, match="binary"):
        FocalLoss(alpha=0.5)(logits, targets)


def test_alpha_out_of_range_raises():
    logits = torch.randn(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError):
        FocalLoss(alpha=1.5)(logits, targets)


def test_negative_gamma_raises():
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)


def test_bad_reduction_raises():
    with pytest.raises(ValueError):
        FocalLoss(reduction="average")


def test_reduction_none_and_sum():
    logits = torch.randn(5, 2)
    targets = torch.randint(0, 2, (5,))
    per_sample = FocalLoss(reduction="none")(logits, targets)
    assert per_sample.shape == (5,)
    total = FocalLoss(reduction="sum")(logits, targets)
    assert total.item() == pytest.approx(per_sample.sum().item(), rel=1e-6)


@pytest.mark.parametrize("bad_logits", [torch.randn(4), torch.randn(2, 2, 2)])
def test_invalid_logits_dim_raises(bad_logits):
    targets = torch.zeros(4, dtype=torch.long)
    with pytest.raises(ValueError):
        FocalLoss()(bad_logits, targets)


def test_invalid_targets_dim_raises():
    logits = torch.randn(4, 2)
    with pytest.raises(ValueError):
        FocalLoss()(logits, torch.zeros(4, 1, dtype=torch.long))


def test_batch_size_mismatch_raises():
    with pytest.raises(ValueError, match="mismatch"):
        FocalLoss()(torch.randn(4, 2), torch.zeros(3, dtype=torch.long))


def test_targets_out_of_range_raises():
    with pytest.raises(ValueError, match="range"):
        FocalLoss()(torch.randn(4, 2), torch.tensor([0, 1, 2, 0]))
    with pytest.raises(ValueError, match="range"):
        FocalLoss()(torch.randn(4, 2), torch.tensor([0, -1, 1, 0]))


def test_numerical_stability_with_extreme_logits():
    logits = torch.tensor([[1e4, -1e4], [-1e4, 1e4]])
    targets = torch.tensor([0, 1])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert torch.isfinite(loss)


def test_float_targets_are_cast():
    logits = torch.randn(4, 2)
    loss = FocalLoss()(logits, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.isfinite(loss)
