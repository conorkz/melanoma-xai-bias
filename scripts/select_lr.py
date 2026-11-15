#!/usr/bin/env python
"""Pick each architecture's learning rate from the Stage 3a runs by validation AUC."""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"^(?P<prefix>[^_]+)_(?P<backbone>resnet50|efficientnet_b0|vit_s)_.*_seed\d+$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--prefix", default="stage3lr")
    return parser.parse_args(argv)


def collect(runs_dir, prefix):
    rows = []
    for path in sorted(Path(runs_dir).glob(f"{prefix}_*/metrics.json")):
        match = PATTERN.match(path.parent.name)
        if not match or match["prefix"] != prefix:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run": path.parent.name,
                "backbone": match["backbone"],
                "lr": float(metrics["lr"]),
                "val_auc": float(metrics["val"]["at_0.5"]["auc"]),
                "best_epoch": metrics.get("best_epoch"),
            }
        )
    return rows


def select(rows):
    best = {}
    for row in rows:
        current = best.get(row["backbone"])
        if current is None or row["val_auc"] > current["val_auc"]:
            best[row["backbone"]] = row
    return best


def main(argv=None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = PROJECT_ROOT / runs_dir
    rows = collect(runs_dir, args.prefix)
    if not rows:
        print(f"No {args.prefix} runs under {runs_dir}", file=sys.stderr)
        return 1
    for row in sorted(rows, key=lambda r: (r["backbone"], r["lr"])):
        print(f"{row['backbone']:16s} lr={row['lr']:.0e} val_auc={row['val_auc']:.4f} best_epoch={row['best_epoch']}")
    print()
    for backbone, row in select(rows).items():
        print(f"{backbone}: --set training.lr={row['lr']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
