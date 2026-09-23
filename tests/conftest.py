"""Shared fixtures: synthetic datasets that need no network or real data."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def make_image(seed, size=32, grayscale=False):
    rng = np.random.default_rng(seed)
    if grayscale:
        return rng.integers(0, 255, (size, size), dtype=np.uint8)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def make_synthetic_dataset(
    root,
    n_lesions_per_class=2,
    images_per_lesion=1,
    img_size=32,
    classes=None,
    seed=0,
):
    """Create a tiny HAM10000-shaped dataset on disk and return its parts."""
    classes = classes or CLASSES
    root = Path(root)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counter = 0
    for lesion_index in range(n_lesions_per_class * len(classes)):
        class_name = classes[lesion_index % len(classes)]
        lesion_id = f"lesion_{class_name}_{lesion_index}"
        for _ in range(images_per_lesion):
            image_id = f"img_{counter:04d}"
            image = make_image(seed + counter, img_size)
            assert cv2.imwrite(str(images_dir / f"{image_id}.jpg"), image)
            rows.append(
                {
                    "image_id": image_id,
                    "dx": class_name,
                    "lesion_id": lesion_id,
                    "age": 40 + (counter % 40),
                    "sex": "male",
                    "localization": "back",
                }
            )
            counter += 1
    frame = pd.DataFrame(rows)
    metadata_csv = root / "metadata.csv"
    frame.to_csv(metadata_csv, index=False)
    return {
        "frame": frame,
        "images_dir": images_dir,
        "metadata_csv": metadata_csv,
        "root": root,
        "img_size": img_size,
    }


@pytest.fixture
def synthetic_dataset(tmp_path):
    return make_synthetic_dataset(tmp_path / "synthetic")


@pytest.fixture
def small_frame(synthetic_dataset):
    return synthetic_dataset["frame"]
