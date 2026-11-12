#!/usr/bin/env python
"""Evaluate trained checkpoints on an external dermoscopic dataset."""

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.data.transforms import build_transforms
from melanoma.models.factory import build_model
from melanoma.training.metrics import binary_metrics, sensitivity_at_specificity
from melanoma.training.trainer import load_checkpoint
from melanoma.utils import ensure_dir, get_device

BACKBONES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit_s": "vit_small_patch16_224",
    "vit_small_patch16_224": "vit_small_patch16_224",
}
FITZPATRICK_ORDER = ["I", "II", "III", "IV", "V", "VI"]


class ExternalImages(Dataset):
    def __init__(self, frame, images_dir, transform):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.images_dir / f"{row['image_id']}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=image)["image"]
        return tensor, index


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="External dataset evaluation")
    parser.add_argument("--metadata", default="data/external/hiba/hiba-skin-lesions.csv")
    parser.add_argument("--images-dir", default="data/external/hiba/images")
    parser.add_argument(
        "--runs-root",
        default="outputs/kaggle/rq1-stage-2/melanoma-xai-bias/extracted/outputs/runs",
    )
    parser.add_argument(
        "--checkpoints-root",
        default="outputs/kaggle/rq1-stage-2/checkpoints/melanoma-xai-bias/checkpoints",
    )
    parser.add_argument("--run-pattern", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backbones", default="efficientnet_b0,resnet50,vit_s")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--conditions", default="ce_random,focal2_cb,ce_random_strong"
    )
    parser.add_argument("--image-type", default="dermoscopic")
    parser.add_argument("--id-column", default="isic_id")
    parser.add_argument("--label-column", default="diagnosis")
    parser.add_argument("--positive-label", default="melanoma")
    parser.add_argument("--tone-column", default="fitzpatrick_skin_type")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=None, help="Debug subset size")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge new rows into an existing external_metrics.csv",
    )
    parser.add_argument("--out-dir", default="results/external/hiba")
    return parser.parse_args(argv)


def load_metadata(args):
    frame = pd.read_csv(resolve(args.metadata))
    if args.image_type is not None:
        frame = frame[frame["image_type"] == args.image_type]
    frame = frame.rename(columns={args.id_column: "image_id"})
    frame["y_true"] = (frame[args.label_column] == args.positive_label).astype(int)
    frame[args.tone_column] = frame[args.tone_column].fillna("unknown")
    frame = frame[["image_id", "y_true", args.tone_column]].rename(
        columns={args.tone_column: "tone_group"}
    )
    if frame.empty:
        raise ValueError("No images remain after filtering")
    if args.max_images is not None:
        frame = frame.head(args.max_images)
    return frame.reset_index(drop=True)


def parse_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def run_ids(args, backbones, seeds):
    conditions = parse_list(args.conditions)
    return [
        f"stage2_{backbone}_{condition}_seed{seed}"
        for backbone in backbones
        for condition in conditions
        for seed in seeds
    ]


def predict(model, loader, device):
    model.eval()
    probabilities = np.full(len(loader.dataset), np.nan, dtype=float)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities[indices.numpy()] = probs.detach().cpu().numpy()
    return probabilities


def metrics_for_group(frame, threshold, group=None):
    subset = frame if group is None else frame[frame["tone_group"] == group]
    y_true = subset["y_true"].to_numpy()
    y_prob = subset["y_prob"].to_numpy()
    predictions = (y_prob >= threshold).astype(int)
    metrics = binary_metrics(y_true, predictions, y_prob)
    metrics["fpr"] = (
        1.0 - metrics["specificity"] if np.isfinite(metrics["specificity"]) else np.nan
    )
    oracle = sensitivity_at_specificity(y_true, y_prob, 0.95)
    metrics["sensitivity_at_95_spec"] = oracle["sensitivity"]
    metrics["oracle_threshold"] = oracle["threshold"]
    metrics["tone_group"] = "all" if group is None else group
    return metrics


def summarize(per_run):
    numeric = [
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "auc",
        "fpr",
        "sensitivity_at_95_spec",
        "oracle_threshold",
        "n",
        "n_positive",
    ]
    summary = (
        per_run.groupby(["backbone", "condition", "tone_group"])[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)
    metadata = load_metadata(args)
    backbones = parse_list(args.backbones)
    seeds = parse_list(args.seeds, int)
    runs_root = resolve(args.runs_root)
    checkpoints_root = resolve(args.checkpoints_root)
    transform = build_transforms(args.img_size, train=False)
    dataset = ExternalImages(metadata, resolve(args.images_dir), transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    records = []
    predictions = []
    for run_id in run_ids(args, backbones, seeds):
        run_dir = runs_root / run_id
        checkpoint_path = checkpoints_root / f"{run_id}.pt"
        metrics_path = run_dir / "metrics.json"
        if not (run_dir.exists() and checkpoint_path.exists() and metrics_path.exists()):
            print(f"Skipping {run_id}: missing run directory, checkpoint, or metrics")
            continue

        run_meta = json.loads(metrics_path.read_text())
        backbone_key = run_meta["backbone"]
        model_name = BACKBONES.get(backbone_key)
        if model_name is None:
            print(f"Skipping {run_id}: unknown backbone {backbone_key}")
            continue

        threshold = float(run_meta["threshold"])
        model = build_model(
            model_name,
            num_classes=2,
            pretrained=False,
            img_size=args.img_size,
        )
        load_checkpoint(checkpoint_path, model, device=device)
        model.to(device)

        probabilities = predict(model, loader, device)
        run_frame = metadata.copy()
        run_frame["y_prob"] = probabilities
        run_frame["run_id"] = run_id
        run_frame["backbone"] = backbone_key
        run_frame["seed"] = int(run_meta["seed"])
        run_frame["threshold"] = threshold
        predictions.append(run_frame)

        for group in ["all"] + sorted(run_frame["tone_group"].unique()):
            metrics = metrics_for_group(
                run_frame, threshold, None if group == "all" else group
            )
            metrics.update(
                {
                    "run_id": run_id,
                    "config_id": run_meta["config_id"],
                    "backbone": backbone_key,
                    "seed": int(run_meta["seed"]),
                    "threshold": threshold,
                }
            )
            records.append(metrics)
        print(f"Finished {run_id}")

        model.cpu()
        del model
        gc.collect()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    if not records:
        print("No external evaluations produced.", file=sys.stderr)
        return 1

    per_run = pd.DataFrame(records)
    per_run["condition"] = per_run["config_id"].str.extract(
        r"^stage2_(?:resnet50|efficientnet_b0|vit_s)_(.+)$"
    )[0]
    per_run = per_run[
        [
            "run_id",
            "config_id",
            "backbone",
            "condition",
            "seed",
            "tone_group",
            "n",
            "n_positive",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
            "auc",
            "fpr",
            "threshold",
            "sensitivity_at_95_spec",
            "oracle_threshold",
        ]
    ]
    new_predictions = pd.concat(predictions, ignore_index=True)

    out_dir = ensure_dir(resolve(args.out_dir))
    if args.merge:
        metrics_path = out_dir / "external_metrics.csv"
        if metrics_path.exists():
            previous = pd.read_csv(metrics_path)
            per_run = pd.concat([previous, per_run], ignore_index=True)
            per_run = per_run.drop_duplicates(
                subset=["run_id", "tone_group"], keep="last"
            )
        predictions_path = out_dir / "external_predictions.csv"
        if predictions_path.exists():
            previous = pd.read_csv(predictions_path)
            new_predictions = pd.concat(
                [previous, new_predictions], ignore_index=True
            )
            new_predictions = new_predictions.drop_duplicates(
                subset=["run_id", "image_id"], keep="last"
            )

    summary = summarize(per_run)
    per_run.to_csv(out_dir / "external_metrics.csv", index=False)
    summary.to_csv(out_dir / "external_summary.csv", index=False)
    new_predictions.to_csv(out_dir / "external_predictions.csv", index=False)

    print(summary.to_string(index=False))
    print(f"Saved external evaluation to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
