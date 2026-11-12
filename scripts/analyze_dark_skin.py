#!/usr/bin/env python
"""Dark-skin robustness analysis: thresholds, continuous tone, and contrast."""

import argparse
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

from melanoma.fairness.dark_skin import (
    assign_tone_groups,
    binned_error_by_contrast,
    binned_error_curve,
    continuous_error_model,
    paired_auc_difference,
    threshold_sensitivity,
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
TONE_COLORS = {"light": "#E8C39E", "medium": "#B97A56", "dark": "#5C3A2E"}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Dark-skin robustness analysis")
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--tone-table", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--metadata", default="data/raw/HAM10000_metadata.csv")
    parser.add_argument("--contrast", default="results/contrast_ham10000.csv")
    parser.add_argument("--out-dir", default="results/dark_skin")
    parser.add_argument("--thresholds", default="55,58,60")
    parser.add_argument("--n-boot", type=int, default=500)
    return parser.parse_args(argv)


def load_ce_predictions(root):
    frames = []
    for path in sorted(root.glob("*_ce_random_seed[123]")):
        predictions = path / "predictions_test.csv"
        if predictions.exists():
            frames.append(pd.read_csv(predictions))
    if not frames:
        raise FileNotFoundError(f"No CE predictions under {root}")
    frame = pd.concat(frames, ignore_index=True)
    frame["backbone"] = frame["run_id"].str.extract(
        r"^stage2_(resnet50|efficientnet_b0|vit_s)_"
    )[0]
    return frame


def load_tone(path, metadata_path):
    tone = pd.read_csv(resolve(path))
    metadata = pd.read_csv(resolve(metadata_path))
    metadata = metadata[["image_id", "age", "sex", "localization", "lesion_id"]]
    tone = tone.merge(metadata, on="image_id", how="left")
    return tone


def auc_difference_table(predictions, tone, thresholds, n_boot):
    base = predictions.merge(tone[["image_id", "L"]], on="image_id", how="left")
    base = base.dropna(subset=["L"])
    rows = []
    for threshold in thresholds:
        assigned = assign_tone_groups(base, threshold)
        for run_id, run in assigned.groupby("run_id", sort=False):
            groups = {
                name: run[run["tone_group"] == name]
                for name in ("light", "medium", "dark")
            }
            for other in ("light", "medium"):
                if groups["dark"].empty or groups[other].empty:
                    continue
                result = paired_auc_difference(
                    groups["dark"], groups[other], n_boot=n_boot
                )
                rows.append(
                    {
                        "threshold": threshold,
                        "run_id": run_id,
                        "backbone": run["backbone"].iloc[0],
                        "seed": int(run["seed"].iloc[0]),
                        "comparison": f"dark_vs_{other}",
                        "delta_auc": result["delta"],
                        "ci_low": result["ci_low"],
                        "ci_high": result["ci_high"],
                        "p_value": result["p_value"],
                        "n_boot_valid": result["n_boot_valid"],
                    }
                )
    return pd.DataFrame(rows)


def plot_robustness(thresholds, models, output):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # Panel A: group AUC by cutoff, averaged across architectures and seeds.
    axis = axes[0]
    for group in ("light", "medium", "dark"):
        subset = thresholds[thresholds["tone_group"] == group]
        summary = subset.groupby("threshold")["auc"].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            summary["threshold"],
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=TONE_COLORS[group],
            label=group.capitalize(),
        )
    axis.set_xlabel("Dark-stratum $L^*$ cutoff")
    axis.set_ylabel("Test AUC")
    axis.set_xticks(sorted(thresholds["threshold"].unique()))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("A. AUC by cutoff")

    # Panel B: false positive rate by cutoff.
    axis = axes[1]
    for group in ("light", "medium", "dark"):
        subset = thresholds[thresholds["tone_group"] == group]
        summary = subset.groupby("threshold")["fpr"].agg(["mean", "std"]).reset_index()
        axis.errorbar(
            summary["threshold"],
            summary["mean"],
            yerr=summary["std"],
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=TONE_COLORS[group],
            label=group.capitalize(),
        )
    axis.set_xlabel("Dark-stratum $L^*$ cutoff")
    axis.set_ylabel("False positive rate")
    axis.set_xticks(sorted(thresholds["threshold"].unique()))
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("B. False positives by cutoff")

    # Panel C: adjusted odds ratios for L* and lesion contrast.
    axis = axes[2]
    terms = ["L", "delta_L"]
    labels = {"L": "Perilesional $L^*$", "delta_L": "Lesion contrast"}
    positions = []
    ticks = []
    labels_out = []
    position = 0
    error_models = models[models["outcome"] == "error"]
    for backbone in BACKBONES:
        subset = error_models[
            (error_models["backbone"] == backbone) & (error_models["term"].isin(terms))
        ]
        for term in terms:
            row = subset[subset["term"] == term]
            if row.empty:
                continue
            row = row.iloc[0]
            axis.errorbar(
                row["odds_ratio"],
                position,
                xerr=[
                    [row["odds_ratio"] - row["ci_low"]],
                    [row["ci_high"] - row["odds_ratio"]],
                ],
                marker="o",
                capsize=3,
                color=COLORS[backbone],
                linewidth=1.5,
            )
            positions.append(position)
            ticks.append(position)
            labels_out.append(f"{BACKBONES[backbone]} / {labels[term]}")
            position += 1
        position += 0.5
    axis.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
    axis.set_xscale("log")
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels_out, fontsize=7)
    axis.set_xlabel("Adjusted odds ratio (95% CI)")
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title("C. Adjusted error model")

    figure.tight_layout()
    output = Path(output)
    ensure_dir(output.parent)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return output


def main(argv=None):
    args = parse_args(argv)
    thresholds = tuple(int(item) for item in args.thresholds.split(",") if item)
    predictions = load_ce_predictions(resolve(args.predictions_root))
    tone = load_tone(args.tone_table, args.metadata)
    contrast = pd.read_csv(resolve(args.contrast))

    sensitivity = threshold_sensitivity(predictions, tone, thresholds)
    differences = auc_difference_table(
        predictions, tone, thresholds, n_boot=args.n_boot
    )
    models = continuous_error_model(
        predictions, tone, contrast, n_boot=args.n_boot
    )
    curve = binned_error_curve(predictions, tone, contrast, n_bins=10)
    contrast_curve = binned_error_by_contrast(
        predictions, tone, contrast, n_bins=5
    )

    out_dir = ensure_dir(resolve(args.out_dir))
    sensitivity.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
    differences.to_csv(out_dir / "auc_differences.csv", index=False)
    models.to_csv(out_dir / "continuous_model.csv", index=False)
    curve.to_csv(out_dir / "error_by_lightness.csv", index=False)
    contrast_curve.to_csv(out_dir / "error_by_contrast.csv", index=False)

    figure = plot_robustness(
        sensitivity,
        models,
        resolve("results/figures/dark_skin_robustness.png"),
    )

    lines = [
        "# Dark-skin robustness",
        "",
        "Group metrics at three L* cutoffs, adjusted error models, and continuous trends.",
        "",
        "## AUC by cutoff",
        "",
        "| Cutoff | Group | n | Melanomas | AUC | FPR |",
        "|---|---|---|---|---|---|",
    ]
    summary = (
        sensitivity.groupby(["threshold", "tone_group"])
        .agg(
            n=("n", "first"),
            n_positive=("n_positive", "first"),
            auc=("auc", "mean"),
            fpr=("fpr", "mean"),
        )
        .reset_index()
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['threshold'])} | {row['tone_group']} | {int(row['n'])} | "
            f"{int(row['n_positive'])} | {row['auc']:.3f} | {row['fpr']:.3f} |"
        )
    lines += [
        "",
        "## Adjusted odds ratios",
        "",
        "| Outcome | Architecture | Term | OR | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in models[models["outcome"] == "error"].iterrows():
        lines.append(
            f"| error | {BACKBONES.get(row['backbone'], row['backbone'])} | {row['term']} | "
            f"{row['odds_ratio']:.3f} | {row['ci_low']:.3f} to {row['ci_high']:.3f} | "
            f"{row['p_value']:.3f} |"
        )
    lines += [
        "",
        "## False positive and false negative models",
        "",
        "| Outcome | Architecture | Term | OR | 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in models[
        models["outcome"].isin(["false_positive", "false_negative"])
        & models["term"].isin(["L", "delta_L"])
    ].iterrows():
        lines.append(
            f"| {row['outcome']} | {BACKBONES.get(row['backbone'], row['backbone'])} | "
            f"{row['term']} | {row['odds_ratio']:.3f} | "
            f"{row['ci_low']:.3f} to {row['ci_high']:.3f} | {row['p_value']:.3f} |"
        )
    (out_dir / "dark_skin.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print(
        models[
            models["term"].isin(["L", "delta_L"])
            & models["outcome"].isin(["error", "false_positive", "false_negative"])
        ].to_string(index=False)
    )
    print(figure)
    print(f"Saved dark-skin analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
