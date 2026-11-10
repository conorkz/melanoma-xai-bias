"""Tests for melanoma.data.dataset."""

import cv2
import numpy as np
import pytest
import torch

from melanoma.data.dataset import HAM10000Dataset
from melanoma.data.metadata import CLASS_NAMES
from melanoma.data.transforms import build_transforms

from conftest import make_image


def test_binary_labels(synthetic_dataset):
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"], task="binary")
    assert len(dataset) == len(synthetic_dataset["frame"])
    assert all(
        label == (row["dx"] == "mel")
        for label, (_, row) in zip(dataset.labels, synthetic_dataset["frame"].iterrows())
    )
    assert dataset.class_names == ["benign", "melanoma"]


def test_multiclass_labels(synthetic_dataset):
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"], task="multiclass")
    assert dataset.class_names == list(CLASS_NAMES)
    for label, (_, row) in zip(dataset.labels, synthetic_dataset["frame"].iterrows()):
        assert label == CLASS_NAMES.index(row["dx"])


def test_getitem_with_transform(synthetic_dataset):
    transform = build_transforms(32, train=False)
    dataset = HAM10000Dataset(
        synthetic_dataset["frame"], synthetic_dataset["images_dir"], transform=transform
    )
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)


def test_getitem_without_transform_returns_unit_range(synthetic_dataset):
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"])
    image, _ = dataset[0]
    assert image.shape[0] == 3
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_missing_image_raises_in_strict_mode(synthetic_dataset):
    path = synthetic_dataset["images_dir"] / f"{synthetic_dataset['frame'].iloc[0]['image_id']}.jpg"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"], strict=True)


def test_missing_image_dropped_in_non_strict_mode(synthetic_dataset):
    path = synthetic_dataset["images_dir"] / f"{synthetic_dataset['frame'].iloc[0]['image_id']}.jpg"
    path.unlink()
    dataset = HAM10000Dataset(
        synthetic_dataset["frame"], synthetic_dataset["images_dir"], strict=False
    )
    assert dataset.n_missing == 1
    assert len(dataset) == len(synthetic_dataset["frame"]) - 1


def test_all_images_missing_non_strict_raises(synthetic_dataset):
    for path in synthetic_dataset["images_dir"].glob("*.jpg"):
        path.unlink()
    with pytest.raises(ValueError, match="No usable images"):
        HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"], strict=False)


def test_missing_images_dir_raises(synthetic_dataset):
    with pytest.raises(FileNotFoundError):
        HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"] / "nope")


def test_empty_frame_raises(synthetic_dataset):
    with pytest.raises(ValueError):
        HAM10000Dataset(
            synthetic_dataset["frame"].iloc[0:0],
            synthetic_dataset["images_dir"],
        )


def test_invalid_task_raises(synthetic_dataset):
    with pytest.raises(ValueError, match="task"):
        HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"], task="regression")


def test_required_columns_raise(synthetic_dataset):
    with pytest.raises(ValueError):
        HAM10000Dataset(
            synthetic_dataset["frame"].drop(columns=["dx"]), synthetic_dataset["images_dir"]
        )


def test_corrupt_image_raises_at_getitem(synthetic_dataset):
    image_id = synthetic_dataset["frame"].iloc[0]["image_id"]
    (synthetic_dataset["images_dir"] / f"{image_id}.jpg").write_bytes(b"this is not a jpeg")
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"])
    with pytest.raises(RuntimeError, match="Unreadable"):
        dataset[0]


def test_grayscale_image_is_expanded(synthetic_dataset):
    image_id = synthetic_dataset["frame"].iloc[0]["image_id"]
    gray = make_image(999, size=32, grayscale=True)
    cv2.imwrite(str(synthetic_dataset["images_dir"] / f"{image_id}.jpg"), gray)
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"])
    image, _ = dataset[0]
    assert image.shape[0] == 3


def test_sample_weights_match_labels(synthetic_dataset):
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"])
    weights = dataset.sample_weights()
    assert isinstance(weights, torch.Tensor)
    assert weights.tolist() == [float(label) for label in dataset.labels]


def test_image_ids_match_frame_order(synthetic_dataset):
    dataset = HAM10000Dataset(synthetic_dataset["frame"], synthetic_dataset["images_dir"])
    assert dataset.image_ids == synthetic_dataset["frame"]["image_id"].tolist()


def test_image_ids_skip_missing_images(synthetic_dataset):
    frame = synthetic_dataset["frame"].copy()
    missing_id = frame.iloc[0]["image_id"]
    (synthetic_dataset["images_dir"] / f"{missing_id}.jpg").unlink()
    dataset = HAM10000Dataset(frame, synthetic_dataset["images_dir"], strict=False)
    assert missing_id not in dataset.image_ids
    assert len(dataset.image_ids) == len(dataset)
