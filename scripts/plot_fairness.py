#!/usr/bin/env python
"""Plot the fairness metrics by architecture, condition, and lightness group."""

import argparse
import re
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

from melanoma.utils import ensure_dir

BACKBONES = ["efficientnet_b0", "resnet50", "vit_s"]
LABELS = {
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "vit_s": "ViT-S/16",
}
COLORS = {
    "efficientnet_b0": "#0072B2",
    "resnet50": "#D55E00",
    "vit_s": "#009E73",
}
CONDITIONS = [
    ("ce_random", "CE baseline"),
    ("focal2_cb", "Focal loss + class-balanced"),
    ("ce_random_strong", "CE + strong augmentation"),
]
TONES = ["light", "medium", "dark"]
METRICS = [
    ("auc", "AUC", (0.65, 0.95)),
    ("fpr", "False positive rate", (0.0, 0.22)),
]
PATTERN = re.compile(r"^stage2_(resnet50|efficientnet_b0|vit_s)_(.+)$")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot fairness metrics")
    parser.add_argument(
        "--runs", default="results/fairness/lightness/fairness_runs.csv"
    )
    parser.add_argument("--out", default="results/figures/fairness_by_tone.png")
    return parser.parse_args(argv)


def split_config(frame):
    parsed = frame["config_id"].str.extract(PATTERN)
    frame = frame.copy()
    frame["backbone"] = parsed[0]
    frame["condition"] = parsed[1]
    return frame


def seed_summary(frame):
    return (
        frame.groupby(["backbone", "condition", "tone_group"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )


def plot(frame, output):
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    positions = np.arange(len(TONES))

    for row, (metric, label, limits) in enumerate(METRICS):
        for column, (condition, title) in enumerate(CONDITIONS):
            axis = axes[row, column]
            subset = frame[frame["condition"] == condition]
            summary = seed_summary(subset.assign(value=subset[metric]))
            for backbone in BACKBONES:
                architecture = summary[summary["backbone"] == backbone].set_index(
                    "tone_group"
                )
                if architecture.empty:
                    continue
                means = [architecture.loc[tone, "mean"] for tone in TONES]
                errors = [
                    architecture.loc[tone, "std"]
                    if np.isfinite(architecture.loc[tone, "std"])
                    else 0.0
                    for tone in TONES
                ]
                axis.errorbar(
                    positions,
                    means,
                    yerr=errors,
                    marker="o",
                    markersize=5,
                    linewidth=1.6,
                    capsize=3,
                    color=COLORS[backbone],
                    label=LABELS[backbone],
                )
            axis.set_xticks(positions)
            axis.set_xticklabels(TONES)
            axis.set_ylim(*limits)
            axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
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
    frame = split_config(pd.read_csv(resolve(args.runs)))
    output = plot(frame, resolve(args.out))
    print(output)
    print(output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
