"""Tests for calibration metrics."""

import numpy as np
import pandas as pd
import pytest

from melanoma.evaluation.calibration import (
    brier_score,
    calibration_summary,
    expected_calibration_error,
    reliability_curve,
)


def test_ece_zero_for_calibrated_constant_predictor():
    y_true = np.array([1] * 30 + [0] * 70)
    y_prob = np.full(100, 0.3)
    assert expected_calibration_error(y_true, y_prob) == 0.0


def test_ece_large_for_overconfident_predictor():
    y_true = np.array([1] * 50 + [0] * 50)
    y_prob = np.full(100, 0.9)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.4)


def test_brier_score_zero_for_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y_true, y_prob) == 0.0


def test_reliability_curve_has_all_bins_and_counts():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    curve = reliability_curve(y_true, y_prob, n_bins=10)
    assert len(curve) == 10
    assert curve["n"].sum() == 4
    assert curve.loc[curve["n"] > 0, "mean_prob"].notna().all()


def test_calibration_summary_adds_group_rows():
    predictions = pd.DataFrame(
        {
            "run_id": ["run1"] * 6,
            "config_id": ["ce"] * 6,
            "backbone": ["resnet50"] * 6,
            "seed": [1] * 6,
            "image_id": [f"img{i}" for i in range(6)],
            "y_true": [0, 0, 1, 0, 1, 1],
            "y_prob": [0.1, 0.2, 0.8, 0.3, 0.7, 0.9],
        }
    )
    tone = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(6)],
            "tone_group": ["light", "light", "light", "dark", "dark", "dark"],
        }
    )
    summary = calibration_summary(predictions, tone)
    assert set(summary["tone_group"]) == {"all", "light", "dark"}
    assert summary["n"].sum() == 12
