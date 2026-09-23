#!/usr/bin/env python
"""Render Grad-CAM heatmaps for sample lesions using a trained checkpoint."""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.config import load_config
from melanoma.data.metadata import load_metadata
from melanoma.data.transforms import build_transforms
from melanoma.models.factory import build_model
from melanoma.training.trainer import load_checkpoint
from melanoma.utils import ensure_dir, get_device
from melanoma.viz.plots import save_heatmap_grid
from melanoma.xai.gradcam import compute_gradcam, get_target_layer, overlay_heatmap


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Grad-CAM demo for the smoke test checkpoint")
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--backbone", default=None, help="Override config backbone")
    parser.add_argument("--img-size", type=int, default=None, help="Override config image size")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--classes", nargs="+", default=["mel", "nv"])
    parser.add_argument("--out-dir", default="outputs/gradcam")
    parser.add_argument("--device", default="cpu", help="cpu is recommended for attribution")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def pick_samples(frame, classes, samples_per_class, seed):
    pieces = []
    for class_name in classes:
        subset = frame[frame["dx"] == class_name]
        if subset.empty:
            raise ValueError(f"No images found for class '{class_name}' in metadata")
        pieces.append(subset.sample(min(samples_per_class, len(subset)), random_state=seed))
    return pd.concat(pieces).reset_index(drop=True)


def main(argv=None):
    args = parse_args(argv)
    config = load_config(resolve(args.config))
    backbone = args.backbone or config.model.backbone
    img_size = args.img_size or config.training.img_size
    checkpoint = resolve(args.checkpoint or f"checkpoints/{config.run_name}.pt")
    out_dir = ensure_dir(resolve(args.out_dir))
    device = get_device(args.device)

    if not checkpoint.exists():
        print(f"Checkpoint not found at {checkpoint}. Run scripts/smoke_test.py first.", file=sys.stderr)
        return 1

    metadata_csv = resolve(config.data.metadata_csv)
    images_dir = resolve(config.data.images_dir)
    frame = load_metadata(metadata_csv)
    samples = pick_samples(frame, args.classes, args.samples_per_class, args.seed)

    model = build_model(backbone, num_classes=config.model.num_classes, pretrained=False)
    load_checkpoint(checkpoint, model, device=device)
    model.to(device)
    model.eval()
    target_layer, reshape = get_target_layer(model, backbone=backbone)

    transform = build_transforms(img_size, train=False)
    grid_items = []
    saved_files = []
    for _, row in samples.iterrows():
        image_path = images_dir / f"{row['image_id']}.jpg"
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(image=image_rgb)["image"].unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.softmax(model(tensor.to(device)), dim=1)[0, 1].cpu())
        heatmap = compute_gradcam(model, tensor, target_layer, target_class=None, reshape_transform=reshape)
        overlay = overlay_heatmap(image_rgb, heatmap)
        single_path = out_dir / f"{row['image_id']}_gradcam.png"
        cv2.imwrite(str(single_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        saved_files.append(single_path)
        label = "melanoma" if row["dx"] == "mel" else "benign"
        grid_items.append(
            {"image": overlay, "title": f"{label} | P(mel)={probability:.2f}"}
        )

    if not grid_items:
        print("No heatmaps generated, check dataset paths.", file=sys.stderr)
        return 1

    grid_path = save_heatmap_grid(grid_items, out_dir / "gradcam_grid.png", cols=3, title="Grad-CAM on smoke test model")
    print(f"Saved {len(saved_files)} heatmaps to {out_dir}")
    print(f"Grid: {grid_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
