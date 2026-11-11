"""Tests for the paper table generator."""

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(stem):
    path = PROJECT_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_table_contains_values_and_escaped_names():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "config_id": ["stage2_resnet50_ce_random"],
            "val_sensitivity_mean": [0.5],
            "val_sensitivity_std": [0.01],
            "sensitivity_mean": [0.6],
            "sensitivity_ci_low": [0.5],
            "sensitivity_ci_high": [0.7],
            "specificity_mean": [0.95],
            "balanced_accuracy_mean": [0.75],
            "auc_mean": [0.9],
        }
    )
    output = generator.stage_table(frame, "Caption", "tab:test")
    assert r"\toprule" in output
    assert r"stage2\_resnet50\_ce\_random" in output
    assert r"$\pm$" in output
    assert r"\label{tab:test}" in output


def test_short_name_for_each_backbone():
    generator = load_script("build_paper_tables")
    assert generator.short_name("stage2_resnet50_ce_random") == "ResNet-50 / CE"
    assert (
        generator.short_name("stage2_efficientnet_b0_focal2_cb")
        == "EfficientNet-B0 / Focal+CB"
    )
    assert generator.short_name("stage2_vit_s_ce_random_strong") == "ViT-S/16 / CE+strong"


def test_xai_table_contains_three_groups():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "tone_group": ["dark", "light", "medium"],
            "n": [129, 450, 450],
            "n_positive": [18, 36, 54],
            "attribution_mass_mean": [0.342, 0.456, 0.374],
            "pointing_game_mean": [0.449, 0.718, 0.613],
            "attribution_iou_mean": [0.208, 0.333, 0.284],
            "error_rate": [0.140, 0.076, 0.091],
        }
    )
    output = generator.xai_table(frame, "Caption", "tab:xai")
    assert r"\toprule" in output
    assert "light" in output
    assert "dark" in output
    assert r"\label{tab:xai}" in output


def test_fairness_table_contains_three_groups():
    generator = load_script("build_paper_tables")
    rows = []
    for config_id in ("stage2_resnet50_ce_random", "stage2_vit_s_focal2_cb"):
        for group, sensitivity, auc, fpr in (
            ("light", 0.5, 0.9, 0.05),
            ("medium", 0.6, 0.92, 0.05),
            ("dark", 0.4, 0.8, 0.1),
        ):
            rows.append(
                {
                    "config_id": config_id,
                    "tone_group": group,
                    "sensitivity_mean": sensitivity,
                    "sensitivity_std": 0.02,
                    "auc_mean": auc,
                    "fpr_mean": fpr,
                    "n": 43,
                    "n_positive": 6,
                }
            )
    frame = pd.DataFrame(rows)
    output = generator.fairness_table(frame, "Caption", "tab:fairness")
    assert r"\toprule" in output
    assert "ResNet-50 / CE" in output
    assert "ViT-S/16 / Focal+CB" in output
    assert r"\label{tab:fairness}" in output
    assert "43 / 6" in output
