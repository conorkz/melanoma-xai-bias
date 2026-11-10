"""PyTorch dataset for HAM10000 dermoscopic images."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from melanoma.data.metadata import CLASS_NAMES, MALIGNANT_CLASS

TASKS = ("binary", "multiclass")


class HAM10000Dataset(Dataset):
    """Dataset over HAM10000 images with binary or multiclass labels.

    In strict mode any missing image file raises immediately at construction
    and any unreadable file raises at access time. In non-strict mode missing
    files are dropped at construction and the number dropped is recorded in
    self.n_missing.
    """

    def __init__(self, frame: pd.DataFrame, images_dir, transform=None, task="binary", strict=True):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if frame.empty:
            raise ValueError("Cannot build a dataset from an empty metadata frame")
        images_dir = Path(images_dir)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if "image_id" not in frame.columns or "dx" not in frame.columns:
            raise ValueError("Frame must contain 'image_id' and 'dx' columns")

        self.images_dir = images_dir
        self.transform = transform
        self.task = task
        self.class_names = list(CLASS_NAMES) if task == "multiclass" else ["benign", "melanoma"]

        records = []
        image_ids = []
        missing = 0
        for image_id, dx in zip(frame["image_id"], frame["dx"]):
            path = images_dir / f"{image_id}.jpg"
            if not path.exists():
                if strict:
                    raise FileNotFoundError(f"Image file missing: {path}")
                missing += 1
                continue
            records.append((path, dx))
            image_ids.append(image_id)
        if not records:
            raise ValueError("No usable images found for this dataset")
        self.n_missing = missing
        self.records = records
        self.image_ids = image_ids

        if task == "binary":
            self.labels = [1 if dx == MALIGNANT_CLASS else 0 for _, dx in records]
        else:
            mapping = {name: index for index, name in enumerate(self.class_names)}
            self.labels = [mapping[dx] for _, dx in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, _ = self.records[index]
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unreadable image file: {path}")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        return image, self.labels[index]

    def sample_weights(self):
        """Per-sample weights for class-balanced sampling."""
        return torch.DoubleTensor(np.asarray(self.labels, dtype=np.float64))
