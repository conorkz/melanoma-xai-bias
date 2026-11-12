"""Tests for paired sensitivity comparisons."""

import numpy as np
import pandas as pd
import pytest

from melanoma.evaluation.paired import (
    difference_in_differences,
    paired_sensitivity_change,
)


def make_frame(predictions, y_true=None):
    y_true = y_true or ([1] * 20 + [0] * 20)
    return pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(len(y_true))],
            "y_true": y_true,
            "y_pred": predictions,
        }
    )


def test_strictly_better_condition_has_positive_delta():
    baseline = make_frame([0] * 20 + [0] * 20)
    improved = make_frame([1] * 20 + [0] * 20)
    result = paired_sensitivity_change(baseline, improved, n_boot=500)
    assert result["delta"] == pytest.approx(1.0)
    assert result["ci_low"] > 0
    assert result["p_value"] < 0.01


def test_identical_conditions_have_zero_delta():
    frame = make_frame([1, 0] * 20)
    result = paired_sensitivity_change(frame, frame, n_boot=200)
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_difference_in_differences_zero_for_equal_improvement():
    a1 = make_frame([0] * 20 + [0] * 20)
    a2 = make_frame([1] * 20 + [0] * 20)
    b1 = make_frame([0] * 20 + [0] * 20)
    b2 = make_frame([1] * 20 + [0] * 20)
    result = difference_in_differences(a1, a2, b1, b2, n_boot=200)
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_difference_in_differences_detects_asymmetric_change():
    a1 = make_frame([0] * 20 + [0] * 20)
    a2 = make_frame([1] * 20 + [0] * 20)
    b1 = make_frame([0] * 20 + [0] * 20)
    b2 = make_frame([0] * 20 + [0] * 20)
    result = difference_in_differences(a1, a2, b1, b2, n_boot=500)
    assert result["delta"] == pytest.approx(1.0)
    assert result["ci_low"] > 0
