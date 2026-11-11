"""Tests for the XAI aggregation functions."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_xai_group_means():
    analyzer = load_script("analyze_xai")
    frame = pd.DataFrame(
        {
            "backbone": ["efficientnet_b0"] * 4,
            "tone_group": ["light", "light", "dark", "dark"],
            "image_id": ["a", "b", "c", "d"],
            "y_true": [1, 0, 1, 0],
            "correct": [1, 1, 0, 0],
            "attribution_mass": [0.6, 0.4, 0.2, 0.4],
            "pointing_game": [1.0, 1.0, 0.0, 1.0],
            "attribution_iou": [0.5, 0.3, 0.1, 0.2],
        }
    )
    summary = analyzer.summarize_xai(frame)
    light = summary[summary["tone_group"] == "light"].iloc[0]
    dark = summary[summary["tone_group"] == "dark"].iloc[0]
    assert light["attribution_mass_mean"] == pytest.approx(0.5)
    assert dark["attribution_mass_mean"] == pytest.approx(0.3)
    assert light["error_rate"] == 0.0
    assert dark["error_rate"] == 1.0
    assert light["n_positive"] == 1


def test_correlations_positive_for_tone_and_error():
    analyzer = load_script("analyze_xai")
    frame = pd.DataFrame(
        {
            "backbone": ["efficientnet_b0"] * 6,
            "tone_group": ["light", "light", "medium", "medium", "dark", "dark"],
            "attribution_mass": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "correct": [1, 1, 0, 1, 0, 0],
        }
    )
    result = analyzer.correlations(frame)
    row = result.iloc[0]
    assert row["attribution_vs_tone_rho"] > 0
    assert row["attribution_vs_error_rho"] > 0
    assert np.isfinite(row["attribution_vs_error_p"])
