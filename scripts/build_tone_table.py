#!/usr/bin/env python
"""Compute an ITA skin tone proxy for HAM10000 images."""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from melanoma.fairness.tone import compute_ita, ita_group, lightness_group
from melanoma.utils import ensure_dir


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compute ITA skin tone proxy")
    parser.add_argument("--metadata", default="data/raw/HAM10000_metadata.csv")
    parser.add_argument("--images-dir", default="data/HAM10000/images")
    parser.add_argument(
        "--masks-dir",
        default="data/raw/ham10000_masks/HAM10000_segmentations_lesion_tschandl",
    )
    parser.add_argument("--out", default="results/tone_ita_ham10000.csv")
    parser.add_argument("--limit", type=int, default=None, help="Debug: limit the number of images")
    return parser.parse_args(argv)


def build_tone_table(image_ids, images_dir, masks_dir):
    rows = []
    for image_id in tqdm(image_ids, desc="ITA"):
        image_path = images_dir / f"{image_id}.jpg"
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            rows.append(
                {
                    "image_id": image_id,
                    "L": float("nan"),
                    "b": float("nan"),
                    "ita": float("nan"),
                    "n_pixels": 0,
                    "ita_group": "unknown",
                    "tone_group": "unknown",
                }
            )
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_path = masks_dir / f"{image_id}_segmentation.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        result = compute_ita(image_rgb, mask)
        result["image_id"] = image_id
        result["ita_group"] = ita_group(result["ita"])
        result["tone_group"] = lightness_group(result["L"])
        rows.append(result)
    table = pd.DataFrame(rows)
    return table[["image_id", "L", "b", "ita", "n_pixels", "ita_group", "tone_group"]]


def main(argv=None):
    args = parse_args(argv)
    metadata_path = resolve(args.metadata)
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}", file=sys.stderr)
        return 1
    frame = pd.read_csv(metadata_path)
    if "image_id" not in frame.columns:
        print("Metadata must contain an image_id column", file=sys.stderr)
        return 1
    image_ids = frame["image_id"].astype(str).tolist()
    if args.limit is not None:
        image_ids = image_ids[: args.limit]

    images_dir = resolve(args.images_dir)
    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1
    masks_dir = resolve(args.masks_dir)

    table = build_tone_table(image_ids, images_dir, masks_dir)
    output = resolve(args.out)
    ensure_dir(output.parent)
    table.to_csv(output, index=False)

    print(table["ita_group"].value_counts(dropna=False).to_string())
    print(f"Saved {len(table)} rows to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
