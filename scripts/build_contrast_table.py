#!/usr/bin/env python
"""Build a per-image lesion-skin contrast table for HAM10000."""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.fairness.contrast import lesion_contrast
from melanoma.utils import ensure_dir


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build HAM10000 contrast table")
    parser.add_argument("--images-dir", default="data/HAM10000/images")
    parser.add_argument(
        "--masks-dir",
        default="data/raw/ham10000_masks/HAM10000_segmentations_lesion_tschandl",
    )
    parser.add_argument("--tone-table", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--out", default="results/contrast_ham10000.csv")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--ring-pixels", type=int, default=11)
    return parser.parse_args(argv)


def build_contrast_table(args):
    images_dir = resolve(args.images_dir)
    masks_dir = resolve(args.masks_dir)
    tone = pd.read_csv(resolve(args.tone_table))
    image_ids = tone["image_id"].tolist()
    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    rows = []
    missing = 0
    for image_id in tqdm(image_ids, desc="contrast"):
        image_path = images_dir / f"{image_id}.jpg"
        mask_path = masks_dir / f"{image_id}_segmentation.png"
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            missing += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        features = lesion_contrast(image, mask, ring_pixels=args.ring_pixels)
        features["image_id"] = image_id
        rows.append(features)

    frame = pd.DataFrame(rows)
    columns = [
        "image_id",
        "lesion_L",
        "skin_L",
        "delta_L",
        "delta_E",
        "lesion_area",
        "skin_pixels",
    ]
    frame = frame[columns]
    print(f"images: {len(frame)}, missing: {missing}")
    return frame


def main(argv=None):
    args = parse_args(argv)
    frame = build_contrast_table(args)
    out_path = resolve(args.out)
    ensure_dir(out_path.parent)
    frame.to_csv(out_path, index=False)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
