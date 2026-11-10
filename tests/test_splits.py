"""Tests for melanoma.data.splits."""

import numpy as np
import pandas as pd
import pytest

from melanoma.data.splits import make_splits, split_summary

from conftest import make_synthetic_dataset


def test_three_way_split_without_group_leakage(synthetic_dataset):
    splits = make_splits(synthetic_dataset["frame"], seed=7)
    assert set(splits["split"]) == {"train", "val", "test"}
    groups = {name: set(part["lesion_id"]) for name, part in splits.groupby("split")}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert sum(len(part) for _, part in splits.groupby("split")) == len(splits)


def test_split_fractions_are_approximate(synthetic_dataset):
    splits = make_splits(synthetic_dataset["frame"], val_fraction=0.2, test_fraction=0.3, seed=3)
    total = len(splits)
    assert abs((splits["split"] == "val").sum() / total - 0.2) < 0.2
    assert abs((splits["split"] == "test").sum() / total - 0.3) < 0.2


def test_split_is_deterministic(synthetic_dataset):
    first = make_splits(synthetic_dataset["frame"], seed=11)
    second = make_splits(synthetic_dataset["frame"], seed=11)
    assert first["split"].tolist() == second["split"].tolist()


def test_split_changes_with_seed(synthetic_dataset):
    first = make_splits(synthetic_dataset["frame"], seed=1)
    second = make_splits(synthetic_dataset["frame"], seed=2)
    assert first["split"].tolist() != second["split"].tolist()


@pytest.mark.parametrize(
    "val_fraction,test_fraction",
    [(0.0, 0.2), (0.2, 0.0), (-0.1, 0.2), (0.2, 1.0), (0.6, 0.5)],
)
def test_invalid_fractions_raise(synthetic_dataset, val_fraction, test_fraction):
    with pytest.raises(ValueError):
        make_splits(synthetic_dataset["frame"], val_fraction=val_fraction, test_fraction=test_fraction)


def test_missing_group_column_raises(synthetic_dataset):
    with pytest.raises(ValueError, match="not found"):
        make_splits(synthetic_dataset["frame"], group_col="patient_id")


def test_nan_groups_raise(synthetic_dataset):
    frame = synthetic_dataset["frame"].copy()
    frame.loc[0, "lesion_id"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        make_splits(frame)


def test_too_few_groups_raise():
    frame = pd.DataFrame(
        {"image_id": ["a"], "dx": ["mel"], "lesion_id": ["only_one"]}
    )
    with pytest.raises(ValueError, match="at least 3"):
        make_splits(frame)


def test_empty_frame_raises():
    frame = pd.DataFrame({"image_id": [], "dx": [], "lesion_id": []})
    with pytest.raises(ValueError):
        make_splits(frame)


def test_does_not_mutate_input(synthetic_dataset):
    frame = synthetic_dataset["frame"].copy(deep=True)
    make_splits(frame)
    assert "split" not in frame.columns


def test_split_summary_counts(synthetic_dataset):
    splits = make_splits(synthetic_dataset["frame"], val_fraction=0.2, test_fraction=0.2, seed=5)
    summary = split_summary(splits)
    assert list(summary.index) == ["train", "val", "test"]
    assert summary["total"].sum() == len(splits)
    for class_name in synthetic_dataset["frame"]["dx"].unique():
        assert summary[class_name].sum() == (splits["dx"] == class_name).sum()


def test_split_summary_missing_split_column_raises(synthetic_dataset):
    with pytest.raises(ValueError, match="split"):
        split_summary(synthetic_dataset["frame"])


def test_split_summary_missing_class_column_raises(synthetic_dataset):
    splits = make_splits(synthetic_dataset["frame"])
    with pytest.raises(ValueError, match="dx"):
        split_summary(splits.drop(columns=["dx"]))
