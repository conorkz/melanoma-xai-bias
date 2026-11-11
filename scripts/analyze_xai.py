#!/usr/bin/env python
"""Quantitative XAI analysis on the HAM10000 test split."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.data.transforms import build_transforms
from melanoma.models.factory import build_model
from melanoma.training.trainer import load_checkpoint
from melanoma.utils import ensure_dir, get_device
from melanoma.xai.gradcam import compute_gradcam, get_target_layer
from melanoma.xai.metrics import (
    attribution_iou,
    attribution_mass_in_mask,
    pointing_game,
)

BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_s": "vit_small_patch16_224",
}
TONE_ORDER = {"light": 0, "medium": 1, "dark": 2}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quantitative XAI analysis")
    parser.add_argument(
        "--checkpoints-root",
        default="outputs/kaggle/rq1-stage-2/checkpoints/melanoma-xai-bias/checkpoints",
    )
    parser.add_argument(
        "--predictions-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument("--tone-table", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--images-dir", default="data/HAM10000/images")
    parser.add_argument(
        "--masks-dir",
        default="data/raw/ham10000_masks/HAM10000_segmentations_lesion_tschandl",
    )
    parser.add_argument("--out-dir", default="results/xai")
    parser.add_argument("--backbones", default="resnet50,efficientnet_b0,vit_s")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images per tone group (debug)",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def load_tone_table(path):
    frame = pd.read_csv(path)
    if "image_id" not in frame.columns or "tone_group" not in frame.columns:
        raise ValueError("Tone table must contain image_id and tone_group")
    return frame[["image_id", "tone_group"]]


def load_predictions(path):
    frame = pd.read_csv(path)
    required = {"image_id", "y_true", "y_prob", "y_pred"}
    if not required <= set(frame.columns):
        raise ValueError(f"Predictions must contain {sorted(required)}")
    return frame


def summarize_xai(frame):
    summary = (
        frame.groupby(["backbone", "tone_group"])
        .agg(
            n=("image_id", "size"),
            n_positive=("y_true", "sum"),
            attribution_mass_mean=("attribution_mass", "mean"),
            pointing_game_mean=("pointing_game", "mean"),
            attribution_iou_mean=("attribution_iou", "mean"),
            error_rate=("correct", lambda values: 1.0 - values.mean()),
        )
        .reset_index()
    )
    return summary


def _spearman(x, y):
    result = spearmanr(x, y)
    statistic = getattr(result, "statistic", getattr(result, "correlation", np.nan))
    pvalue = getattr(result, "pvalue", np.nan)
    return float(statistic), float(pvalue)


def correlations(frame):
    rows = []
    for backbone, subset in frame.groupby("backbone"):
        valid = subset.dropna(subset=["attribution_mass"])
        if len(valid) < 3:
            continue
        error_rho, error_p = _spearman(valid["attribution_mass"], 1 - valid["correct"])
        tone_valid = valid.assign(
            tone_index=valid["tone_group"].map(TONE_ORDER)
        ).dropna(subset=["tone_index"])
        if len(tone_valid) >= 3:
            tone_rho, tone_p = _spearman(
                tone_valid["attribution_mass"], tone_valid["tone_index"]
            )
        else:
            tone_rho, tone_p = float("nan"), float("nan")
        rows.append(
            {
                "backbone": backbone,
                "n": len(valid),
                "attribution_vs_error_rho": error_rho,
                "attribution_vs_error_p": error_p,
                "attribution_vs_tone_rho": tone_rho,
                "attribution_vs_tone_p": tone_p,
            }
        )
    return pd.DataFrame(rows)


def format_value(value):
    if value is None:
        return "nan"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}" if np.isfinite(value) else "nan"
    return str(value)


def to_markdown(summary, correlation):
    columns = [
        "backbone",
        "tone_group",
        "n",
        "n_positive",
        "attribution_mass_mean",
        "pointing_game_mean",
        "attribution_iou_mean",
        "error_rate",
    ]
    lines = [
        "# Quantitative XAI",
        "",
        "Attribution mass, pointing game, and attribution IoU for the HAM10000 test split.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    lines += [
        "",
        "## Correlations",
        "",
        "| backbone | n | rho (error) | p (error) | rho (tone) | p (tone) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in correlation.iterrows():
        lines.append(
            f"| {row['backbone']} | {row['n']} | "
            f"{format_value(row['attribution_vs_error_rho'])} | "
            f"{format_value(row['attribution_vs_error_p'])} | "
            f"{format_value(row['attribution_vs_tone_rho'])} | "
            f"{format_value(row['attribution_vs_tone_p'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)
    checkpoints_root = resolve(args.checkpoints_root)
    predictions_root = resolve(args.predictions_root)
    images_dir = resolve(args.images_dir)
    masks_dir = resolve(args.masks_dir)
    tone = load_tone_table(resolve(args.tone_table))

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    transform = build_transforms(args.img_size, train=False)

    records = []
    for backbone_key in backbones:
        model_name = BACKBONES.get(backbone_key)
        if model_name is None:
            print(f"Unknown backbone {backbone_key}", file=sys.stderr)
            continue
        for seed in seeds:
            run_id = f"stage2_{backbone_key}_ce_random_seed{seed}"
            checkpoint = checkpoints_root / f"{run_id}.pt"
            predictions_path = predictions_root / run_id / "predictions_test.csv"
            if not checkpoint.exists() or not predictions_path.exists():
                print(f"Skipping {run_id}: missing checkpoint or predictions")
                continue

            model = build_model(
                model_name,
                num_classes=2,
                pretrained=False,
                img_size=args.img_size,
            )
            load_checkpoint(checkpoint, model, device=device)
            model.to(device)
            model.eval()
            target_layer, reshape = get_target_layer(model, backbone=model_name)

            predictions = load_predictions(predictions_path)
            predictions = predictions.merge(tone, on="image_id", how="left")
            predictions["tone_group"] = predictions["tone_group"].fillna("unknown")
            if args.max_images is not None:
                pieces = []
                for _, group in predictions.groupby("tone_group"):
                    pieces.append(
                        group.sample(
                            n=min(args.max_images, len(group)), random_state=0
                        )
                    )
                predictions = pd.concat(pieces).reset_index(drop=True)

            for _, row in tqdm(predictions.iterrows(), total=len(predictions), desc=run_id):
                image_path = images_dir / f"{row['image_id']}.jpg"
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                tensor = transform(image=image_rgb)["image"].unsqueeze(0)
                target_class = int(row["y_pred"])
                heatmap = compute_gradcam(
                    model,
                    tensor,
                    target_layer,
                    target_class=target_class,
                    reshape_transform=reshape,
                )
                mask_path = masks_dir / f"{row['image_id']}_segmentation.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue

                mass = attribution_mass_in_mask(heatmap, mask)
                pointing = pointing_game(heatmap, mask)
                iou = attribution_iou(heatmap, mask)
                records.append(
                    {
                        "backbone": backbone_key,
                        "seed": seed,
                        "run_id": run_id,
                        "image_id": row["image_id"],
                        "tone_group": row["tone_group"],
                        "y_true": int(row["y_true"]),
                        "y_prob": float(row["y_prob"]),
                        "y_pred": target_class,
                        "correct": int(target_class == int(row["y_true"])),
                        "attribution_mass": mass,
                        "pointing_game": float(pointing) if np.isfinite(pointing) else np.nan,
                        "attribution_iou": iou,
                    }
                )

    if not records:
        print("No XAI records produced. Check checkpoints and predictions.", file=sys.stderr)
        return 1

    frame = pd.DataFrame(records)
    summary = summarize_xai(frame)
    correlation = correlations(frame)

    out_dir = ensure_dir(resolve(args.out_dir))
    frame.to_csv(out_dir / "xai_metrics.csv", index=False)
    summary.to_csv(out_dir / "xai_summary.csv", index=False)
    correlation.to_csv(out_dir / "xai_correlations.csv", index=False)
    (out_dir / "xai.md").write_text(to_markdown(summary, correlation), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Saved XAI results to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
