#!/usr/bin/env python
"""Aggregate per-run metrics into the RQ1 benchmark table."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.utils import ensure_dir

GROUP_COLUMNS = ["config_id", "backbone", "loss", "sampler", "augmentation", "img_size"]
METRICS = ["sensitivity", "specificity", "balanced_accuracy", "auc"]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate training runs into a benchmark table")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--prefix", default="rq1_benchmark")
    parser.add_argument("--run-prefix", default=None, help="Only include run ids with this prefix")
    return parser.parse_args(argv)


def load_runs(runs_dir, run_prefix=None):
    """Read every metrics.json under runs_dir that has a completed test split."""
    rows = []
    for metrics_path in sorted(Path(runs_dir).glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        run_id = data.get("run_id", metrics_path.parent.name)
        if run_prefix is not None and not str(run_id).startswith(run_prefix):
            continue
        test = data.get("test")
        if not test:
            continue
        at_target = test.get("at_target")
        if not isinstance(at_target, dict):
            continue
        bootstrap = test.get("bootstrap", {}) or {}
        sensitivity_ci = bootstrap.get("sensitivity", {}) or {}
        row = {
            "run_id": run_id,
            "config_id": data.get("config_id", data.get("run_name", metrics_path.parent.name)),
            "seed": data.get("seed"),
            "backbone": data.get("backbone"),
            "loss": data.get("loss"),
            "sampler": data.get("sampler"),
            "augmentation": data.get("augmentation"),
            "img_size": data.get("img_size"),
            "best_epoch": data.get("best_epoch"),
            "n_test": at_target.get("n"),
            "n_positive": at_target.get("n_positive"),
            "sensitivity_ci_low": sensitivity_ci.get("low"),
            "sensitivity_ci_high": sensitivity_ci.get("high"),
        }
        for metric in METRICS:
            row[metric] = at_target.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_runs(runs):
    """Group runs by configuration and report mean and standard deviation."""
    if runs.empty:
        raise ValueError("No completed runs found")
    for column in GROUP_COLUMNS:
        if column not in runs.columns:
            runs[column] = None

    aggregations = {
        "n_seeds": ("seed", "nunique"),
        "n_runs": ("run_id", "count"),
        "n_test": ("n_test", "first"),
        "n_positive": ("n_positive", "first"),
        "sensitivity_ci_low": ("sensitivity_ci_low", "mean"),
        "sensitivity_ci_high": ("sensitivity_ci_high", "mean"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    grouped = runs.groupby(GROUP_COLUMNS, dropna=False)
    aggregate = grouped.agg(**aggregations).reset_index()
    return aggregate.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def format_value(value):
    if value is None:
        return "nan"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def to_markdown(aggregate, title="RQ1 benchmark"):
    """Render a compact markdown table without a tabulate dependency."""
    columns = list(aggregate.columns)
    lines = [
        f"# {title}",
        "",
        "Test split. The operating threshold was selected on validation at the target specificity.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in aggregate.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    runs_dir = resolve(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"Runs directory not found: {runs_dir}", file=sys.stderr)
        return 1

    runs = load_runs(runs_dir, run_prefix=args.run_prefix)
    if runs.empty:
        print(f"No completed runs with a test split found under {runs_dir}", file=sys.stderr)
        return 1

    aggregate = aggregate_runs(runs)
    out_dir = ensure_dir(resolve(args.out_dir))
    runs_path = out_dir / f"{args.prefix}_runs.csv"
    aggregate_path = out_dir / f"{args.prefix}.csv"
    markdown_path = out_dir / f"{args.prefix}.md"

    runs.to_csv(runs_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    markdown_path.write_text(to_markdown(aggregate), encoding="utf-8")

    prefix_text = f" (run prefix {args.run_prefix})" if args.run_prefix else ""
    print(f"Collected {len(runs)} runs across {len(aggregate)} configurations{prefix_text}")
    print(f"Runs: {runs_path}")
    print(f"Benchmark CSV: {aggregate_path}")
    print(f"Benchmark table: {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
