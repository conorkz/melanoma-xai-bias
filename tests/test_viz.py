"""Tests for melanoma.viz.plots."""

import numpy as np
import pytest
import torch

from melanoma.viz.plots import (
    denormalize_tensor,
    plot_class_distribution,
    plot_roc_curve,
    plot_sample_grid,
    plot_training_history,
    save_heatmap_grid,
)


def test_denormalize_tensor_range_and_shape():
    tensor = torch.randn(3, 16, 16)
    image = denormalize_tensor(tensor)
    assert image.shape == (16, 16, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0
    assert image.max() <= 255


def test_plot_class_distribution_writes_file(tmp_path, small_frame):
    path = plot_class_distribution(small_frame, tmp_path / "figures" / "counts.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_class_distribution_empty_raises(tmp_path):
    import pandas as pd

    with pytest.raises(ValueError):
        plot_class_distribution(pd.DataFrame({"dx": []}), tmp_path / "x.png")


def test_plot_sample_grid_writes_file(tmp_path, synthetic_dataset):
    path = plot_sample_grid(
        synthetic_dataset["frame"],
        synthetic_dataset["images_dir"],
        tmp_path / "grid.png",
        n_per_class=2,
    )
    assert path.exists()


def test_plot_training_history_writes_file(tmp_path):
    history = {
        "train_loss": [1.0, 0.8, 0.6],
        "val_metrics": [
            {"balanced_accuracy": 0.5},
            {"balanced_accuracy": float("nan")},
            {"balanced_accuracy": 0.7},
        ],
    }
    path = plot_training_history(history, tmp_path / "curves.png")
    assert path.exists()


def test_plot_training_history_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_training_history({"train_loss": [], "val_metrics": []}, tmp_path / "x.png")


def test_save_heatmap_grid_writes_file(tmp_path):
    samples = [
        {"image": np.zeros((16, 16, 3), dtype=np.uint8), "title": "a"},
        {"image": np.ones((16, 16, 3), dtype=np.uint8), "title": "b"},
    ]
    path = save_heatmap_grid(samples, tmp_path / "grid.png", cols=2)
    assert path.exists()


def test_save_heatmap_grid_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        save_heatmap_grid([], tmp_path / "grid.png")


def test_plot_roc_curve_writes_file(tmp_path):
    path = plot_roc_curve([1, 1, 0, 0], [0.9, 0.7, 0.2, 0.1], tmp_path / "roc.png")
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_roc_curve_single_class_raises(tmp_path):
    with pytest.raises(ValueError):
        plot_roc_curve([1, 1], [0.9, 0.8], tmp_path / "roc.png")
