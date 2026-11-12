#!/usr/bin/env python
"""Seed-level inference for the Stage 2 benchmark.

The experimental unit for a training intervention is the seed, not the test
image. This script reports per-seed metrics, seed-level 95% t intervals for the
mean, paired condition changes, difference-in-differences, AUC comparisons, and
sensitivity at a common 95% test specificity.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.training.metrics import sensitivity_at_specificity
from melanoma.utils import ensure_dir

PATTERN = re.compile(r"^stage2_(resnet50|efficientnet_b0|vit_s)_(.+)_seed([123])$")
BACKBONES = ["resnet50", "efficientnet_b0", "vit_s"]
CONDITIONS = ["ce_random", "focal2_cb", "ce_random_strong"]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Seed-level inference")
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--out-dir", default="results/seed_intervals")
    return parser.parse_args(argv)


def load_runs(root):
    runs = {}
    for path in sorted(root.glob("*/predictions_test.csv")):
        frame = pd.read_csv(path)
        run_id = frame["run_id"].iloc[0]
        match = PATTERN.match(run_id)
        if match is None:
            continue
        backbone, condition, seed = match.groups()
        runs[(backbone, condition, int(seed))] = frame
    return runs


def metrics(frame):
    y_true = frame["y_true"].to_numpy()
    y_prob = frame["y_prob"].to_numpy()
    y_pred = frame["y_pred"].to_numpy()
    positives = y_true == 1
    negatives = y_true == 0
    oracle = sensitivity_at_specificity(y_true, y_prob, 0.95)
    return {
        "n": len(frame),
        "n_positive": int(positives.sum()),
        "sensitivity": float(y_pred[positives].mean()),
        "specificity": float(1.0 - y_pred[negatives].mean()),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity_at_95_test_spec": float(oracle["sensitivity"]),
    }


def t_summary(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    else:
        half = float("nan")
    return mean, sd, mean - half, mean + half


def paired_frame(frame_a, frame_b):
    merged = frame_a[["image_id", "y_true", "y_prob", "y_pred"]].merge(
        frame_b[["image_id", "y_prob", "y_pred"]],
        on="image_id",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        raise ValueError("No overlapping images")
    return merged


def paired_changes(frame_a, frame_b):
    merged = paired_frame(frame_a, frame_b)
    y_true = merged["y_true"].to_numpy()
    pred_a = merged["y_pred_a"].to_numpy()
    pred_b = merged["y_pred_b"].to_numpy()
    prob_a = merged["y_prob_a"].to_numpy()
    prob_b = merged["y_prob_b"].to_numpy()
    positives = y_true == 1
    return {
        "delta_sensitivity": float(pred_b[positives].mean() - pred_a[positives].mean()),
        "delta_auc": float(roc_auc_score(y_true, prob_b) - roc_auc_score(y_true, prob_a)),
        "n": len(merged),
        "n_positive": int(positives.sum()),
    }


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(resolve(args.predictions_root))
    rows = []
    for (backbone, condition, seed), frame in runs.items():
        result = metrics(frame)
        rows.append(
            {
                "backbone": backbone,
                "condition": condition,
                "seed": seed,
                **result,
            }
        )
    per_seed = pd.DataFrame(rows).sort_values(["backbone", "condition", "seed"])
    out_dir = ensure_dir(resolve(args.out_dir))
    per_seed.to_csv(out_dir / "stage2_per_seed.csv", index=False)

    summary_rows = []
    for (backbone, condition), group in per_seed.groupby(["backbone", "condition"]):
        for metric_name in (
            "sensitivity",
            "specificity",
            "auc",
            "sensitivity_at_95_test_spec",
        ):
            mean, sd, low, high = t_summary(group[metric_name].to_numpy())
            summary_rows.append(
                {
                    "backbone": backbone,
                    "condition": condition,
                    "metric": metric_name,
                    "mean": mean,
                    "sd": sd,
                    "t_low": low,
                    "t_high": high,
                    "n_seeds": len(group),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "stage2_summary.csv", index=False)

    paired_rows = []
    for backbone in BACKBONES:
        for seed in (1, 2, 3):
            ce = runs.get((backbone, "ce_random", seed))
            for condition in ("focal2_cb", "ce_random_strong"):
                other = runs.get((backbone, condition, seed))
                if ce is None or other is None:
                    continue
                result = paired_changes(ce, other)
                paired_rows.append(
                    {
                        "backbone": backbone,
                        "condition": condition,
                        "seed": seed,
                        **result,
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out_dir / "paired_per_seed.csv", index=False)

    paired_summary_rows = []
    for (backbone, condition), group in paired.groupby(["backbone", "condition"]):
        for metric_name in ("delta_sensitivity", "delta_auc"):
            mean, sd, low, high = t_summary(group[metric_name].to_numpy())
            paired_summary_rows.append(
                {
                    "backbone": backbone,
                    "condition": condition,
                    "metric": metric_name,
                    "mean": mean,
                    "sd": sd,
                    "t_low": low,
                    "t_high": high,
                    "n_seeds": len(group),
                }
            )
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(out_dir / "paired_summary.csv", index=False)

    did_rows = []
    for seed in (1, 2, 3):
        for first, second in (
            ("resnet50", "vit_s"),
            ("efficientnet_b0", "vit_s"),
            ("resnet50", "efficientnet_b0"),
        ):
            frames = {
                key: runs.get((key[0], key[1], seed))
                for key in (
                    (first, "ce_random"),
                    (first, "focal2_cb"),
                    (second, "ce_random"),
                    (second, "focal2_cb"),
                )
            }
            if any(frame is None for frame in frames.values()):
                continue
            first_change = paired_changes(
                frames[(first, "ce_random")], frames[(first, "focal2_cb")]
            )
            second_change = paired_changes(
                frames[(second, "ce_random")], frames[(second, "focal2_cb")]
            )
            did_rows.append(
                {
                    "architecture_a": first,
                    "architecture_b": second,
                    "seed": seed,
                    "delta_sensitivity": first_change["delta_sensitivity"]
                    - second_change["delta_sensitivity"],
                    "delta_auc": first_change["delta_auc"] - second_change["delta_auc"],
                }
            )
    did = pd.DataFrame(did_rows)
    did.to_csv(out_dir / "did_per_seed.csv", index=False)

    did_summary_rows = []
    for (first, second), group in did.groupby(["architecture_a", "architecture_b"]):
        for metric_name in ("delta_sensitivity", "delta_auc"):
            mean, sd, low, high = t_summary(group[metric_name].to_numpy())
            did_summary_rows.append(
                {
                    "architecture_a": first,
                    "architecture_b": second,
                    "metric": metric_name,
                    "mean": mean,
                    "sd": sd,
                    "t_low": low,
                    "t_high": high,
                    "n_seeds": len(group),
                }
            )
    did_summary = pd.DataFrame(did_summary_rows)
    did_summary.to_csv(out_dir / "did_summary.csv", index=False)

    auc_rows = []
    for seed in (1, 2, 3):
        for condition in ("ce_random", "focal2_cb"):
            for first, second in (
                ("resnet50", "vit_s"),
                ("efficientnet_b0", "vit_s"),
                ("resnet50", "efficientnet_b0"),
            ):
                frame_a = runs.get((first, condition, seed))
                frame_b = runs.get((second, condition, seed))
                if frame_a is None or frame_b is None:
                    continue
                merged = paired_frame(frame_a, frame_b)
                y_true = merged["y_true"].to_numpy()
                auc_a = roc_auc_score(y_true, merged["y_prob_a"])
                auc_b = roc_auc_score(y_true, merged["y_prob_b"])
                auc_rows.append(
                    {
                        "condition": condition,
                        "architecture_a": first,
                        "architecture_b": second,
                        "seed": seed,
                        "delta_auc": float(auc_a - auc_b),
                    }
                )
    auc_differences = pd.DataFrame(auc_rows)
    auc_differences.to_csv(out_dir / "auc_differences.csv", index=False)

    auc_summary_rows = []
    for (condition, first, second), group in auc_differences.groupby(
        ["condition", "architecture_a", "architecture_b"]
    ):
        mean, sd, low, high = t_summary(group["delta_auc"].to_numpy())
        auc_summary_rows.append(
            {
                "condition": condition,
                "architecture_a": first,
                "architecture_b": second,
                "mean": mean,
                "sd": sd,
                "t_low": low,
                "t_high": high,
                "n_seeds": len(group),
            }
        )
    auc_summary = pd.DataFrame(auc_summary_rows)
    auc_summary.to_csv(out_dir / "auc_summary.csv", index=False)

    lines = [
        "# Seed-level inference",
        "",
        "The seed is the experimental unit. Intervals are 95% t intervals across three seeds.",
        "",
        "## Stage 2 sensitivity",
        "",
        "| Configuration | Mean | SD | 95% t interval |",
        "|---|---|---|---|",
    ]
    sens = summary[summary["metric"] == "sensitivity"]
    for _, row in sens.iterrows():
        lines.append(
            f"| {row['backbone']} / {row['condition']} | {row['mean']:.3f} | "
            f"{row['sd']:.3f} | {row['t_low']:.3f} to {row['t_high']:.3f} |"
        )
    lines += [
        "",
        "## Paired sensitivity changes",
        "",
        "| Architecture | Condition | Mean | SD | 95% t interval |",
        "|---|---|---|---|---|",
    ]
    paired_sens = paired_summary[paired_summary["metric"] == "delta_sensitivity"]
    for _, row in paired_sens.iterrows():
        lines.append(
            f"| {row['backbone']} | {row['condition']} | {row['mean']:+.3f} | "
            f"{row['sd']:.3f} | {row['t_low']:+.3f} to {row['t_high']:+.3f} |"
        )
    lines += [
        "",
        "## Difference in differences",
        "",
        "| Comparison | Mean | SD | 95% t interval |",
        "|---|---|---|---|",
    ]
    did_sens = did_summary[did_summary["metric"] == "delta_sensitivity"]
    for _, row in did_sens.iterrows():
        lines.append(
            f"| {row['architecture_a']} minus {row['architecture_b']} | "
            f"{row['mean']:+.3f} | {row['sd']:.3f} | "
            f"{row['t_low']:+.3f} to {row['t_high']:+.3f} |"
        )
    (out_dir / "seed_intervals.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(sens[["backbone", "condition", "mean", "sd", "t_low", "t_high"]].to_string(index=False))
    print(paired_sens[["backbone", "condition", "mean", "sd", "t_low", "t_high"]].to_string(index=False))
    print(did_sens[["architecture_a", "architecture_b", "mean", "sd", "t_low", "t_high"]].to_string(index=False))
    print(f"Saved seed-level inference to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
