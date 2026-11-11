#!/usr/bin/env python
"""Plot the XAI metrics by architecture and lightness group."""

import argparse
import math
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

ARCHITECTURES = ["efficientnet_b0", "resnet50", "vit_s"]
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
TONES = ["light", "medium", "dark"]
METRICS = [
    ("attribution_mass", "Attribution mass inside lesion"),
    ("pointing_game", "Pointing game accuracy"),
]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plot XAI metrics by tone group")
    parser.add_argument("--metrics", default="results/xai/xai_all_metrics.csv")
    parser.add_argument("--out", default="results/figures/xai_by_tone.png")
    return parser.parse_args(argv)


def seed_means(frame, metric):
    per_seed = (
        frame.groupby(["backbone", "tone_group", "seed"])[metric]
        .mean()
        .reset_index()
    )
    summary = (
        per_seed.groupby(["backbone", "tone_group"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary


def plot(frame, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    width = 0.24
    positions = np.arange(len(TONES))

    for axis, (metric, title) in zip(axes, METRICS):
        summary = seed_means(frame, metric)
        for index, backbone in enumerate(ARCHITECTURES):
            subset = summary[summary["backbone"] == backbone].set_index("tone_group")
            means = [subset.loc[tone, "mean"] for tone in TONES]
            errors = [
                subset.loc[tone, "std"] if not math.isnan(subset.loc[tone, "std"]) else 0.0
                for tone in TONES
            ]
            axis.bar(
                positions + (index - 1) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                error_kw={"elinewidth": 1, "ecolor": "#333333"},
                label=LABELS[backbone],
                color=COLORS[backbone],
                edgecolor="#222222",
                linewidth=0.6,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([
            "light\n$n=450$",
            "medium\n$n=450$",
            "dark\n$n=129$",
        ])
        axis.set_ylabel(title)
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
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
    frame = pd.read_csv(resolve(args.metrics))
    output = plot(frame, resolve(args.out))
    print(output)
    print(output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
