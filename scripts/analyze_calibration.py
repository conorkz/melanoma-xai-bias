#!/usr/bin/env python
"""Calibration analysis for the Stage 2 runs."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.evaluation.calibration import (
    apply_temperature,
    brier_score,
    calibration_summary,
    expected_calibration_error,
    fit_temperature,
    prior_shift_correction,
    reliability_curve,
)
from melanoma.utils import ensure_dir

BACKBONES = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "vit_s": "ViT-S/16",
}
COLORS = {
    "efficientnet_b0": "#0072B2",
    "resnet50": "#D55E00",
    "vit_s": "#009E73",
}
TONES = ["light", "medium", "dark"]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Calibration analysis")
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--tone-table", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--out-dir", default="results/calibration")
    parser.add_argument("--split", default="results/split_ham10000_lesion_seed42.csv")
    return parser.parse_args(argv)


def load_predictions(root):
    frames = []
    for path in sorted(root.glob("*/predictions_test.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No predictions under {root}")
    frame = pd.concat(frames, ignore_index=True)
    frame["backbone"] = frame["run_id"].str.extract(
        r"^stage\d+_(resnet50|efficientnet_b0|vit_s)_"
    )[0]
    return frame


def effective_training_prior(run_dir, config_id, train_prior):
    """Effective melanoma prior seen by the loss during training.

    A class-balanced sampler sets the sampled prior to 0.5. A focal alpha on
    the positive class multiplies the positive odds by alpha / (1 - alpha).
    The training config is read from run.json when available.
    """
    training = {}
    run_json = run_dir / "run.json"
    if run_json.exists():
        training = (json.loads(run_json.read_text()).get("config") or {}).get(
            "training"
        ) or {}
    sampler = training.get(
        "sampler", "class_balanced" if "_cb" in config_id else "random"
    )
    loss = training.get("loss", "focal" if "focal" in config_id else "ce")
    odds = 1.0 if sampler == "class_balanced" else train_prior / (1.0 - train_prior)
    alpha = training.get("focal_alpha") if loss == "focal" else None
    if alpha is not None and 0.0 < float(alpha) < 1.0:
        odds *= float(alpha) / (1.0 - float(alpha))
    return odds / (1.0 + odds)


def corrected_calibration(root, split_path):
    """ECE and Brier before and after prior-shift and temperature correction.

    Class-balanced runs are moved from the sampler prior (0.5) to the training
    prevalence; random-sampler runs need no prior correction. A temperature is
    then fitted on the validation split and applied to the test split. Both
    corrections are monotone, so AUC and the validation-selected 95%
    specificity operating point are unchanged.
    """
    split = pd.read_csv(split_path)
    train = split[split["split"] == "train"]
    train_prior = float((train["dx"] == "mel").mean())
    rows = []
    for run_dir in sorted(root.iterdir()):
        val_path = run_dir / "predictions_val.csv"
        test_path = run_dir / "predictions_test.csv"
        if not (val_path.exists() and test_path.exists()):
            continue
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        config_id = str(test["config_id"].iloc[0])
        sampler_prior = effective_training_prior(run_dir, config_id, train_prior)

        def corrected(frame):
            return prior_shift_correction(frame["y_prob"], sampler_prior, train_prior)

        val_prior = corrected(val)
        test_prior = corrected(test)
        temperature = fit_temperature(val["y_true"], val_prior)
        test_scaled = apply_temperature(test_prior, temperature)
        y = test["y_true"].to_numpy()
        rows.append(
            {
                "run_id": test["run_id"].iloc[0],
                "config_id": config_id,
                "seed": int(test["seed"].iloc[0]),
                "sampler_prior": sampler_prior,
                "train_prior": train_prior,
                "temperature": temperature,
                "ece_raw": expected_calibration_error(y, test["y_prob"]),
                "ece_prior": expected_calibration_error(y, test_prior),
                "ece_prior_ts": expected_calibration_error(y, test_scaled),
                "brier_raw": brier_score(y, test["y_prob"]),
                "brier_prior": brier_score(y, test_prior),
                "brier_prior_ts": brier_score(y, test_scaled),
            }
        )
    return pd.DataFrame(rows)


def plot_calibration(predictions, summary, output):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axis = axes[0]
    ce = predictions[predictions["config_id"].str.endswith("ce_random")]
    for backbone in BACKBONES:
        subset = ce[ce["backbone"] == backbone]
        if subset.empty:
            continue
        curve = reliability_curve(subset["y_true"], subset["y_prob"], n_bins=10)
        curve = curve[curve["n"] > 0]
        axis.plot(
            curve["mean_prob"],
            curve["observed_rate"],
            marker="o",
            linewidth=1.6,
            markersize=4,
            color=COLORS[backbone],
            label=BACKBONES[backbone],
        )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed melanoma rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(color="#DDDDDD", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("A. Reliability, CE baseline")

    axis = axes[1]
    width = 0.24
    positions = np.arange(len(TONES))
    for index, backbone in enumerate(BACKBONES):
        subset = summary[
            (summary["backbone"] == backbone)
            & (summary["config_id"].str.endswith("ce_random"))
            & (summary["tone_group"].isin(TONES))
        ]
        means = []
        errors = []
        for tone in TONES:
            values = subset[subset["tone_group"] == tone]["ece"].dropna()
            means.append(values.mean() if not values.empty else np.nan)
            errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
        axis.bar(
            positions + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[backbone],
            edgecolor="#222222",
            linewidth=0.6,
            label=BACKBONES[backbone],
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(TONES)
    axis.set_ylabel("Expected calibration error")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title("B. Calibration by lightness group")

    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    ensure_dir(output.parent)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    args = parse_args(argv)
    predictions = load_predictions(resolve(args.predictions_root))
    tone = pd.read_csv(resolve(args.tone_table))[["image_id", "tone_group"]]
    summary = calibration_summary(predictions, tone)

    out_dir = ensure_dir(resolve(args.out_dir))
    summary.to_csv(out_dir / "calibration_summary.csv", index=False)
    predictions.to_csv(out_dir / "predictions_for_calibration.csv", index=False)

    figure = plot_calibration(
        predictions, summary, resolve("results/figures/calibration_reliability.png")
    )

    overall = (
        summary[summary["tone_group"] == "all"]
        .groupby("config_id")[["ece", "brier"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    overall.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in overall.columns
    ]
    overall.to_csv(out_dir / "calibration_overall.csv", index=False)

    lines = [
        "# Calibration",
        "",
        "Expected calibration error (10 bins) and Brier score on the test split.",
        "",
        "| Configuration | ECE | Brier |",
        "|---|---|---|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['config_id']} | {row['ece_mean']:.3f} | {row['brier_mean']:.3f} |"
        )
    (out_dir / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    corrected = corrected_calibration(
        resolve(args.predictions_root), resolve(args.split)
    )
    corrected.to_csv(out_dir / "calibration_corrected_runs.csv", index=False)
    metrics = [c for c in corrected.columns if c.startswith(("ece_", "brier_"))]
    corrected_summary = (
        corrected.groupby("config_id")[metrics + ["temperature"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    corrected_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in corrected_summary.columns
    ]
    corrected_summary.to_csv(out_dir / "calibration_corrected.csv", index=False)
    lines += [
        "",
        "## After prior-shift and temperature correction",
        "",
        "| Configuration | ECE raw | ECE prior | ECE prior+TS | T |",
        "|---|---|---|---|---|",
    ]
    for _, row in corrected_summary.iterrows():
        lines.append(
            f"| {row['config_id']} | {row['ece_raw_mean']:.3f} | "
            f"{row['ece_prior_mean']:.3f} | {row['ece_prior_ts_mean']:.3f} | "
            f"{row['temperature_mean']:.2f} |"
        )
    (out_dir / "calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(overall.to_string(index=False))
    print(corrected_summary[["config_id", "ece_raw_mean", "ece_prior_mean", "ece_prior_ts_mean", "brier_raw_mean", "brier_prior_ts_mean", "temperature_mean"]].to_string(index=False))
    print(figure)
    print(f"Saved calibration results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
