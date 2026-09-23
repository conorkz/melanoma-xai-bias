"""Tests for melanoma.training.metrics."""

import numpy as np
import pytest

from melanoma.training.metrics import binary_metrics, sensitivity_at_specificity


def test_perfect_predictions():
    metrics = binary_metrics([1, 1, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auc"] == 1.0


def test_known_confusion_matrix():
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert metrics["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5


def test_no_positives_gives_nan_sensitivity():
    metrics = binary_metrics([0, 0, 0], [0, 1, 0])
    assert np.isnan(metrics["sensitivity"])
    assert metrics["specificity"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["n_positive"] == 0


def test_no_negatives_gives_nan_specificity():
    metrics = binary_metrics([1, 1, 1], [1, 0, 1])
    assert np.isnan(metrics["specificity"])
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)


def test_auc_nan_without_probabilities():
    metrics = binary_metrics([1, 0], [1, 0])
    assert np.isnan(metrics["auc"])


def test_auc_nan_with_single_class():
    metrics = binary_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert np.isnan(metrics["auc"])


def test_auc_nan_with_non_finite_probs():
    metrics = binary_metrics([1, 0], [1, 0], [np.nan, 0.1])
    assert np.isnan(metrics["auc"])


def test_f1_nan_when_undefined():
    metrics = binary_metrics([0, 0], [0, 0])
    assert np.isnan(metrics["f1"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        binary_metrics([], [])


def test_non_binary_y_true_raises():
    with pytest.raises(ValueError):
        binary_metrics([0, 1, 2], [0, 1, 1])


def test_non_binary_y_pred_raises():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, 2])


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0, 1], [0.5])


def test_sensitivity_at_specificity_separable():
    y_true = [1] * 10 + [0] * 10
    y_prob = [0.9] * 10 + [0.1] * 10
    result = sensitivity_at_specificity(y_true, y_prob, target_specificity=0.95)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] >= 0.95


def test_sensitivity_at_specificity_single_class_is_nan():
    result = sensitivity_at_specificity([1, 1], [0.9, 0.8])
    assert np.isnan(result["sensitivity"])


def test_sensitivity_at_specificity_invalid_target():
    with pytest.raises(ValueError):
        sensitivity_at_specificity([1, 0], [0.9, 0.1], target_specificity=1.5)


def test_sensitivity_at_specificity_length_mismatch():
    with pytest.raises(ValueError):
        sensitivity_at_specificity([1, 0], [0.9])
