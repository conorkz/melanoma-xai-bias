#!/usr/bin/env python
"""Group-wise fairness analysis for test predictions."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.fairness.metrics import disparity_summary, group_metrics
from melanoma.utils import ensure_dir


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze group-wise fairness")
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--tone-table", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--out-dir", default="results/fairness")
    parser.add_argument("--group-column", default="tone_group")
    parser.add_argument("--run-prefix", default="stage2")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_tone_table(path, group_column):
    frame = pd.read_csv(path)
    required = {"image_id", group_column}
    if not required <= set(frame.columns):
        raise ValueError(f"Tone table must contain {sorted(required)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Tone table contains duplicate image_id rows")
    return frame[["image_id", group_column]]


def load_predictions(root, run_prefix):
    frames = []
    for path in sorted(Path(root).glob("*/predictions_test.csv")):
        frame = pd.read_csv(path)
        required = {"image_id", "y_true", "y_prob", "config_id", "seed", "run_id", "threshold"}
        if not required <= set(frame.columns):
            continue
        if run_prefix and not str(frame["run_id"].iloc[0]).startswith(run_prefix):
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No predictions_test.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def summarize(runs, group_column):
    aggregate = (
        runs.groupby(["config_id", group_column])
        .agg(
            n_seeds=("seed", "nunique"),
            n=("n", "first"),
            n_positive=("n_positive", "first"),
            sensitivity_mean=("sensitivity", "mean"),
            sensitivity_std=("sensitivity", "std"),
            sensitivity_low_mean=("sensitivity_low", "mean"),
            sensitivity_high_mean=("sensitivity_high", "mean"),
            specificity_mean=("specificity", "mean"),
            fpr_mean=("fpr", "mean"),
            fnr_mean=("fnr", "mean"),
            auc_mean=("auc", "mean"),
        )
        .reset_index()
    )
    return aggregate


def format_value(value):
    if value is None:
        return "nan"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}" if np.isfinite(value) else "nan"
    return str(value)


def to_markdown(aggregate, disparity, group_column):
    columns = [
        "config_id",
        group_column,
        "n_seeds",
        "n",
        "n_positive",
        "sensitivity_mean",
        "sensitivity_std",
        "fpr_mean",
        "auc_mean",
    ]
    lines = [
        "# Tone-based fairness analysis",
        "",
        f"Groups are image-based tone strata, not clinical Fitzpatrick labels. Group column: `{group_column}`.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in aggregate.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    lines += [
        "",
        "## Disparity",
        "",
        "| config_id | max_min_gap | light_dark_ratio |",
        "|---|---|---|",
    ]
    for _, row in disparity.iterrows():
        lines.append(
            f"| {row['config_id']} | {format_value(row['max_min_gap'])} | "
            f"{format_value(row['light_dark_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    tone = load_tone_table(resolve(args.tone_table), args.group_column)
    predictions = load_predictions(resolve(args.predictions_root), args.run_prefix)
    merged = predictions.merge(tone, on="image_id", how="inner", validate="many_to_one")
    if merged.empty:
        print("No overlapping image ids between predictions and tone table", file=sys.stderr)
        return 1

    results = []
    for (run_id, config_id, seed), subset in merged.groupby(["run_id", "config_id", "seed"]):
        threshold = float(subset["threshold"].iloc[0])
        metrics = group_metrics(
            subset,
            group_col=args.group_column,
            threshold=threshold,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        metrics["run_id"] = run_id
        metrics["config_id"] = config_id
        metrics["seed"] = seed
        results.append(metrics)

    runs = pd.concat(results, ignore_index=True)
    aggregate = summarize(runs, args.group_column)

    disparity_rows = []
    for config_id, subset in aggregate.groupby("config_id"):
        summary = disparity_summary(
            subset,
            group_col=args.group_column,
            metric="sensitivity_mean",
        )
        summary["config_id"] = config_id
        disparity_rows.append(summary)
    disparity = pd.DataFrame(disparity_rows)

    out_dir = ensure_dir(resolve(args.out_dir))
    runs.to_csv(out_dir / "fairness_runs.csv", index=False)
    aggregate.to_csv(out_dir / "fairness_summary.csv", index=False)
    disparity.to_csv(out_dir / "fairness_disparity.csv", index=False)
    (out_dir / "fairness.md").write_text(
        to_markdown(aggregate, disparity, args.group_column), encoding="utf-8"
    )

    print(
        aggregate[["config_id", args.group_column, "n", "n_positive"]]
        .drop_duplicates()
        .to_string(index=False)
    )
    print(f"Saved fairness results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
