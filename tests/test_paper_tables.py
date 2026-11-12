"""Tests for the paper table generator."""

import importlib.util
import math
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
            "sensitivity_std": [0.05],
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
    rows = []
    for backbone in ("efficientnet_b0", "resnet50"):
        for group, mass, pointing, iou, error, positive in (
            ("light", 0.456, 0.718, 0.333, 0.076, 36),
            ("medium", 0.374, 0.613, 0.284, 0.091, 54),
            ("dark", 0.342, 0.449, 0.208, 0.140, 18),
        ):
            rows.append(
                {
                    "backbone": backbone,
                    "tone_group": group,
                    "n": 450 if group != "dark" else 129,
                    "n_positive": positive,
                    "attribution_mass_mean": mass,
                    "pointing_game_mean": pointing,
                    "attribution_iou_mean": iou,
                    "error_rate": error,
                }
            )
    output = generator.xai_table(pd.DataFrame(rows), "Caption", "tab:xai")
    assert r"\toprule" in output
    assert r"\multirow{3}{*}{EfficientNet-B0}" in output
    assert "ResNet-50" in output
    assert "light" in output
    assert "dark" in output
    assert r"\label{tab:xai}" in output


def test_xai_correlation_table_formats_small_p():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "backbone": ["efficientnet_b0"],
            "n": [1027],
            "attribution_vs_tone_rho": [-0.169567],
            "attribution_vs_tone_p": [4.57e-08],
            "attribution_vs_error_rho": [-0.023207],
            "attribution_vs_error_p": [0.4575],
        }
    )
    output = generator.xai_correlation_table(frame, "Caption", "tab:xaicorr")
    assert "EfficientNet-B0" in output
    assert r"4.6\times10^{-8}" in output
    assert "0.458" in output
    assert r"\label{tab:xaicorr}" in output


def test_external_table_handles_undefined_sensitivity():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "backbone": ["resnet50", "resnet50"],
            "tone_group": ["II", "IV"],
            "n_mean": [804, 25],
            "n_positive_mean": [144, 0],
            "sensitivity_mean": [0.354, math.nan],
            "sensitivity_std": [0.05, math.nan],
            "specificity_mean": [0.891, 1.0],
            "specificity_std": [0.01, 0.0],
            "auc_mean": [0.739, math.nan],
            "auc_std": [0.02, math.nan],
        }
    )
    output = generator.external_table(frame, "Caption", "tab:external")
    assert "ResNet-50" in output
    assert "804" in output
    assert "144" in output
    assert "--" in output
    assert r"\label{tab:external}" in output


def test_dark_skin_table_contains_three_cutoffs():
    generator = load_script("build_paper_tables")
    rows = []
    for threshold in (55, 58, 60):
        for group, n, positive, auc, fpr in (
            ("light", 644, 81, 0.868, 0.050),
            ("medium", 840, 99, 0.911, 0.049),
            ("dark", 43, 6, 0.805, 0.126),
        ):
            rows.append(
                {
                    "threshold": threshold,
                    "tone_group": group,
                    "n": n,
                    "n_positive": positive,
                    "sensitivity": 0.5,
                    "auc": auc,
                    "fpr": fpr,
                }
            )
    output = generator.dark_skin_table(pd.DataFrame(rows), "Caption", "tab:darkskin")
    assert "55" in output
    assert "58" in output
    assert "60" in output
    assert r"\label{tab:darkskin}" in output


def test_adjusted_model_table_contains_both_error_types():
    generator = load_script("build_paper_tables")
    rows = []
    for outcome in ("false_positive", "false_negative"):
        for backbone in ("resnet50", "efficientnet_b0", "vit_s"):
            for term, odds in (("L", 0.8), ("delta_L", 2.0)):
                rows.append(
                    {
                        "outcome": outcome,
                        "backbone": backbone,
                        "term": term,
                        "odds_ratio": odds,
                        "ci_low": odds - 0.1,
                        "ci_high": odds + 0.1,
                    }
                )
    output = generator.adjusted_model_table(
        pd.DataFrame(rows), "Caption", "tab:adjusted"
    )
    assert "False positive" in output
    assert "False negative" in output
    assert r"\label{tab:adjusted}" in output


def test_calibration_table_contains_ece_and_brier():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "config_id": [
                "stage2_resnet50_ce_random",
                "stage2_resnet50_focal2_cb",
            ],
            "ece_mean": [0.046, 0.203],
            "ece_std": [0.013, 0.041],
            "brier_mean": [0.074, 0.133],
            "brier_std": [0.004, 0.016],
        }
    )
    output = generator.calibration_table(frame, "Caption", "tab:calibration")
    assert "0.046" in output
    assert "0.203" in output
    assert "Brier" in output


def test_ranking_table_contains_did_rows():
    generator = load_script("build_paper_tables")
    frame = pd.DataFrame(
        {
            "test": ["paired", "did"],
            "comparison": ["focal2_cb_vs_ce", "resnet50_minus_vit_s"],
            "backbone": ["resnet50", "resnet50_vs_vit_s"],
            "delta_mean": [-0.116, -0.199],
            "ci_low_min": [-0.244, -0.437],
            "ci_high_max": [0.031, -0.011],
            "p_max": [0.248, 0.032],
        }
    )
    output = generator.ranking_table(frame, "Caption", "tab:ranking")
    assert "Difference in differences" in output
    assert "ResNet-50 minus ViT-S/16" in output


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
