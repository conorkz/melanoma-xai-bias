"""Tests for melanoma.data.metadata."""

import pandas as pd
import pytest

from melanoma.data.metadata import (
    class_counts,
    load_metadata,
    to_binary_labels,
    to_class_indices,
)

from conftest import make_synthetic_dataset


def test_load_valid_metadata(synthetic_dataset):
    frame = load_metadata(synthetic_dataset["metadata_csv"])
    assert len(frame) == len(synthetic_dataset["frame"])
    assert set(frame["dx"]) == set(synthetic_dataset["frame"]["dx"])


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_metadata("/nonexistent/metadata.csv")


def test_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"image_id": ["a"], "dx": ["mel"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lesion_id"):
        load_metadata(path)


def test_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("image_id,dx,lesion_id\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_metadata(path)


def test_duplicate_image_ids_raise(tmp_path):
    path = tmp_path / "dupes.csv"
    pd.DataFrame(
        {"image_id": ["a", "a"], "dx": ["mel", "nv"], "lesion_id": ["l1", "l2"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_metadata(path)


def test_unknown_labels_raise(tmp_path):
    path = tmp_path / "bad_label.csv"
    pd.DataFrame(
        {"image_id": ["a"], "dx": ["melanoma"], "lesion_id": ["l1"]}
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown diagnosis"):
        load_metadata(path)


def test_labels_are_normalized(tmp_path):
    path = tmp_path / "dirty.csv"
    pd.DataFrame(
        {"image_id": ["a", "b"], "dx": [" MEL ", "Nv"], "lesion_id": ["l1", "l2"]}
    ).to_csv(path, index=False)
    frame = load_metadata(path)
    assert list(frame["dx"]) == ["mel", "nv"]


def test_binary_labels(small_frame):
    labels = to_binary_labels(small_frame)
    assert set(labels.unique()) == {0, 1}
    assert labels[small_frame["dx"] == "mel"].eq(1).all()
    assert labels[small_frame["dx"] != "mel"].eq(0).all()


def test_class_indices_are_contiguous(small_frame):
    indices = to_class_indices(small_frame)
    assert indices.min() == 0
    assert indices.max() == small_frame["dx"].nunique() - 1


def test_class_counts_sorted_ascending(small_frame):
    counts = class_counts(small_frame)
    assert list(counts.values) == sorted(counts.values)
