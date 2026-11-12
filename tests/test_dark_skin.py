"""Tests for the dark-skin robustness analysis."""

import numpy as np
import pandas as pd
import pytest

from melanoma.fairness.dark_skin import (
    binned_error_curve,
    continuous_error_model,
    paired_auc_difference,
    threshold_sensitivity,
)


def synthetic_frames(n_per_group=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    tone_rows = []
    contrast_rows = []
    image_id = 0
    for L in (45, 52, 58, 65, 75, 82):
        for _ in range(n_per_group):
            y_true = int(image_id % 2 == 0)
            prob = np.clip(0.35 + 0.006 * (L - 60) + rng.normal(0, 0.08), 0.01, 0.99)
            for run, seed_value in enumerate((1, 2, 3)):
                rows.append(
                    {
                        "run_id": f"run_{run}",
                        "config_id": "ce",
                        "backbone": "efficientnet_b0",
                        "seed": seed_value,
                        "image_id": f"img_{image_id}",
                        "y_true": y_true,
                        "y_prob": prob,
                        "y_pred": int(prob >= 0.5),
                    }
                )
            tone_rows.append({"image_id": f"img_{image_id}", "L": L})
            contrast_rows.append(
                {
                    "image_id": f"img_{image_id}",
                    "delta_L": 20.0 + rng.normal(0, 3),
                    "delta_E": 30.0 + rng.normal(0, 4),
                    "lesion_area": 800 + rng.integers(0, 400),
                }
            )
            image_id += 1
    predictions = pd.DataFrame(rows)
    tone = pd.DataFrame(tone_rows)
    contrast = pd.DataFrame(contrast_rows)
    return predictions, tone, contrast


def test_threshold_sensitivity_recovers_dark_auc_drop():
    predictions, tone, _ = synthetic_frames()
    result = threshold_sensitivity(predictions, tone, thresholds=(55, 58, 60))
    assert set(result["threshold"]) == {55, 58, 60}
    dark = result[result["tone_group"] == "dark"]["auc"].mean()
    light = result[result["tone_group"] == "light"]["auc"].mean()
    assert dark < light


def test_threshold_sensitivity_has_expected_columns():
    predictions, tone, _ = synthetic_frames()
    result = threshold_sensitivity(predictions, tone, thresholds=(58,))
    for column in ("threshold", "run_id", "tone_group", "n", "n_positive", "sensitivity", "auc", "fpr"):
        assert column in result.columns


def test_paired_auc_difference_zero_for_identical_scores():
    frame = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1],
            "y_prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )
    result = paired_auc_difference(frame, frame, n_boot=200)
    assert result["delta"] == pytest.approx(0.0)
    assert result["ci_low"] <= 0 <= result["ci_high"]


def test_continuous_model_returns_odds_ratios():
    predictions, tone, contrast = synthetic_frames(n_per_group=150)
    tone["age"] = 60
    tone["sex"] = "male"
    tone["localization"] = "back"
    tone["lesion_id"] = tone["image_id"]
    result = continuous_error_model(
        predictions,
        tone,
        contrast,
        n_boot=20,
    )
    assert {"outcome", "term", "odds_ratio", "ci_low", "ci_high", "p_value"} <= set(
        result.columns
    )
    assert set(result["outcome"]) == {"error", "false_positive", "false_negative"}
    assert np.isfinite(result["odds_ratio"]).all()
    assert (result["odds_ratio"] > 0).all()


def test_binned_error_curve_returns_requested_bins():
    predictions, tone, contrast = synthetic_frames()
    result = binned_error_curve(predictions, tone, contrast, n_bins=5)
    assert len(result) == 5
    assert result["n"].sum() == len(tone)
    assert result["error_rate"].between(0, 1).all()
