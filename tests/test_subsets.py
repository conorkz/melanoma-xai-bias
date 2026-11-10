"""Tests for melanoma.data.subsets."""

import pytest

from melanoma.data.subsets import stratified_subset

from conftest import make_synthetic_dataset


def test_stratified_subset_keeps_lesions_intact(synthetic_dataset):
    frame = synthetic_dataset["frame"]
    subset = stratified_subset(frame, size=len(frame) // 2, seed=0)
    assert 0 < len(subset) <= len(frame)
    for lesion_id in subset["lesion_id"].unique():
        original = frame[frame["lesion_id"] == lesion_id]
        chosen = subset[subset["lesion_id"] == lesion_id]
        assert len(chosen) == len(original)


def test_stratified_subset_covers_every_class(synthetic_dataset):
    frame = synthetic_dataset["frame"]
    subset = stratified_subset(frame, size=len(frame) // 2, seed=0)
    assert set(subset["dx"]) == set(frame["dx"])


def test_stratified_subset_returns_copy_when_size_large(synthetic_dataset):
    frame = synthetic_dataset["frame"]
    subset = stratified_subset(frame, size=len(frame) + 10, seed=0)
    assert len(subset) == len(frame)
    assert subset is not frame


def test_stratified_subset_none_returns_copy(synthetic_dataset):
    frame = synthetic_dataset["frame"]
    subset = stratified_subset(frame, size=None, seed=0)
    assert len(subset) == len(frame)
    assert subset is not frame


def test_stratified_subset_too_small_raises(synthetic_dataset):
    with pytest.raises(ValueError):
        stratified_subset(synthetic_dataset["frame"], size=2, seed=0)


def test_stratified_subset_keeps_multi_image_lesions(tmp_path):
    dataset = make_synthetic_dataset(
        tmp_path / "multi", n_lesions_per_class=3, images_per_lesion=2
    )
    frame = dataset["frame"]
    subset = stratified_subset(frame, size=len(frame) // 2, seed=0)
    for lesion_id in subset["lesion_id"].unique():
        assert len(subset[subset["lesion_id"] == lesion_id]) == 2
