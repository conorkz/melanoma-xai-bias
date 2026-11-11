"""Tests for melanoma.fairness.metrics."""

import numpy as np
import pandas as pd
import pytest

from melanoma.fairness.metrics import disparity_summary, group_metrics


def test_group_metrics_known_values():
    frame = pd.DataFrame(
        {
            "group": ["light", "light", "light", "light", "dark", "dark", "dark", "dark"],
            "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
            "y_prob": [0.9, 0.8, 0.2, 0.1, 0.9, 0.1, 0.2, 0.1],
        }
    )
    result = group_metrics(frame, group_col="group", n_boot=20)
    light = result[result["group"] == "light"].iloc[0]
    dark = result[result["group"] == "dark"].iloc[0]
    assert light["sensitivity"] == 1.0
    assert dark["sensitivity"] == 0.5
    assert light["fpr"] == 0.0
    assert dark["fpr"] == 0.0
    assert light["n_positive"] == 2
    assert dark["n_positive"] == 2


def test_disparity_summary():
    frame = pd.DataFrame(
        {
            "group": ["light", "medium", "dark"],
            "sensitivity": [0.8, 0.6, 0.4],
        }
    )
    result = disparity_summary(frame, metric="sensitivity")
    assert result["max_min_gap"] == pytest.approx(0.4)
    assert result["light_dark_ratio"] == pytest.approx(2.0)


def test_disparity_summary_missing_dark():
    frame = pd.DataFrame({"group": ["light", "medium"], "sensitivity": [0.8, 0.6]})
    result = disparity_summary(frame, metric="sensitivity")
    assert result["max_min_gap"] == pytest.approx(0.2)
    assert np.isnan(result["light_dark_ratio"])


def test_group_metrics_missing_group_column_raises():
    frame = pd.DataFrame({"y_true": [1, 0], "y_prob": [0.9, 0.1]})
    with pytest.raises(ValueError, match="group"):
        group_metrics(frame)


def test_group_metrics_missing_probability_column_raises():
    frame = pd.DataFrame({"group": ["light", "light"], "y_true": [1, 0]})
    with pytest.raises(ValueError, match="y_prob"):
        group_metrics(frame)
