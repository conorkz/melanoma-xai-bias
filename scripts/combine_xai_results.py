#!/usr/bin/env python
"""Combine the per-architecture XAI outputs into one set of result files."""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_xai import correlations, summarize_xai, to_markdown
from melanoma.utils import ensure_dir

ARCHITECTURES = ("efficientnet_b0", "resnet50", "vit_s")
METRIC_COLUMNS = (
    "backbone",
    "seed",
    "run_id",
    "image_id",
    "tone_group",
    "y_true",
    "y_prob",
    "y_pred",
    "correct",
    "attribution_mass",
    "pointing_game",
    "attribution_iou",
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Combine per-architecture XAI results")
    parser.add_argument("--root", default="results/xai")
    parser.add_argument("--architectures", default=",".join(ARCHITECTURES))
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def load_architectures(root, architectures):
    frames = []
    missing = []
    for architecture in architectures:
        path = root / architecture / "xai_metrics.csv"
        if not path.exists():
            missing.append(architecture)
            continue
        frame = pd.read_csv(path)
        if "backbone" not in frame.columns:
            frame["backbone"] = architecture
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No XAI metrics found under {root}")
    return pd.concat(frames, ignore_index=True), missing


def main(argv=None):
    args = parse_args(argv)
    root = resolve(args.root)
    architectures = [item.strip() for item in args.architectures.split(",") if item.strip()]
    frame, missing = load_architectures(root, architectures)
    frame = frame[[column for column in METRIC_COLUMNS if column in frame.columns]]

    summary = summarize_xai(frame)
    correlation = correlations(frame)

    out_dir = ensure_dir(resolve(args.out_dir) if args.out_dir else root)
    frame.to_csv(out_dir / "xai_all_metrics.csv", index=False)
    summary.to_csv(out_dir / "xai_all_summary.csv", index=False)
    correlation.to_csv(out_dir / "xai_all_correlations.csv", index=False)
    (out_dir / "xai.md").write_text(to_markdown(summary, correlation), encoding="utf-8")

    if missing:
        print(f"Missing architectures: {', '.join(missing)}", file=sys.stderr)
    print(summary.to_string(index=False))
    print(correlation.to_string(index=False))
    print(f"Saved combined XAI results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
