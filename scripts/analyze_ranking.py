#!/usr/bin/env python
"""Paired tests for the architecture ranking reversal."""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.evaluation.paired import (
    difference_in_differences,
    paired_sensitivity_change,
)
from melanoma.utils import ensure_dir

BACKBONES = ["resnet50", "efficientnet_b0", "vit_s"]
LABELS = {
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_s": "ViT-S/16",
}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Paired ranking tests")
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--out-dir", default="results/ranking")
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args(argv)


def load_runs(root):
    pattern = re.compile(
        r"^stage2_(resnet50|efficientnet_b0|vit_s)_(.+)_seed([123])$"
    )
    runs = {}
    for path in sorted(root.glob("*/predictions_test.csv")):
        frame = pd.read_csv(path)
        run_id = frame["run_id"].iloc[0]
        match = pattern.match(run_id)
        if match is None:
            continue
        backbone, condition, seed = match.groups()
        runs[(backbone, condition, int(seed))] = frame
    return runs


def main(argv=None):
    args = parse_args(argv)
    runs = load_runs(resolve(args.predictions_root))
    rows = []
    for backbone in BACKBONES:
        for seed in (1, 2, 3):
            ce = runs.get((backbone, "ce_random", seed))
            focal = runs.get((backbone, "focal2_cb", seed))
            strong = runs.get((backbone, "ce_random_strong", seed))
            if ce is None:
                continue
            if focal is not None:
                result = paired_sensitivity_change(
                    ce, focal, n_boot=args.n_boot
                )
                rows.append(
                    {
                        "test": "paired",
                        "comparison": "focal2_cb_vs_ce",
                        "backbone": backbone,
                        "seed": seed,
                        **result,
                    }
                )
            if strong is not None:
                result = paired_sensitivity_change(
                    ce, strong, n_boot=args.n_boot
                )
                rows.append(
                    {
                        "test": "paired",
                        "comparison": "ce_random_strong_vs_ce",
                        "backbone": backbone,
                        "seed": seed,
                        **result,
                    }
                )

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
            result = difference_in_differences(
                frames[(first, "ce_random")],
                frames[(first, "focal2_cb")],
                frames[(second, "ce_random")],
                frames[(second, "focal2_cb")],
                n_boot=args.n_boot,
            )
            rows.append(
                {
                    "test": "did",
                    "comparison": f"{first}_minus_{second}",
                    "backbone": f"{first}_vs_{second}",
                    "seed": seed,
                    **result,
                }
            )

    if not rows:
        print("No ranking tests produced.", file=sys.stderr)
        return 1
    frame = pd.DataFrame(rows)
    frame.to_csv(ensure_dir(resolve(args.out_dir)) / "ranking_tests.csv", index=False)

    summary = (
        frame.groupby(["test", "comparison", "backbone"])
        .agg(
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            ci_low_min=("ci_low", "min"),
            ci_high_max=("ci_high", "max"),
            p_max=("p_value", "max"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(resolve(args.out_dir) / "ranking_summary.csv", index=False)

    lines = [
        "# Paired ranking tests",
        "",
        "Sensitivity changes paired by test image. Intervals are 95% bootstrap intervals.",
        "",
        "| Test | Comparison | Architecture | Delta | 95% CI | p (max across seeds) | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['test']} | {row['comparison']} | {row['backbone']} | "
            f"{row['delta_mean']:.3f} | "
            f"{row['ci_low_min']:.3f} to {row['ci_high_max']:.3f} | "
            f"{row['p_max']:.3f} | {int(row['seeds'])} |"
        )
    (resolve(args.out_dir) / "ranking.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"Saved ranking tests to {resolve(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
