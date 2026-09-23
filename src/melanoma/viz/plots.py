"""Plotting helpers for scripts and the EDA notebook."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from melanoma.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from melanoma.utils import ensure_dir


def denormalize_tensor(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Convert a normalized CHW tensor into an HWC uint8 RGB image."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std) + np.asarray(mean)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def plot_class_distribution(frame, path, title="HAM10000 class distribution"):
    """Bar chart of diagnosis counts on a logarithmic scale."""
    counts = frame["dx"].value_counts()
    if counts.empty:
        raise ValueError("Cannot plot class distribution of an empty frame")
    ensure_dir(Path(path).parent)
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    sns.barplot(x=counts.index, y=counts.values, ax=axis, color="#3b6ea5")
    axis.set_yscale("log")
    axis.set_ylabel("images (log scale)")
    axis.set_title(title)
    for position, value in enumerate(counts.values):
        axis.text(position, value, str(value), ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return Path(path)


def plot_sample_grid(frame, images_dir, path, n_per_class=5, seed=0, title="Samples per class"):
    """Grid with one row per class and a few sample images per row."""
    ensure_dir(Path(path).parent)
    classes = sorted(frame["dx"].unique())
    figure, axes = plt.subplots(len(classes), n_per_class, figsize=(2.2 * n_per_class, 2.4 * len(classes)), dpi=120)
    axes = np.atleast_2d(axes)
    for row, class_name in enumerate(classes):
        subset = frame[frame["dx"] == class_name]
        samples = subset.sample(min(n_per_class, len(subset)), random_state=seed)
        for column in range(n_per_class):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(samples):
                image_path = Path(images_dir) / f"{samples.iloc[column]['image_id']}.jpg"
                image = cv2.imread(str(image_path))
                if image is not None:
                    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(class_name, loc="left", fontsize=9)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return Path(path)


def plot_training_history(history, path):
    """Two panel chart of training loss and validation balanced accuracy."""
    if "train_loss" not in history or not history["train_loss"]:
        raise ValueError("History has no training losses to plot")
    ensure_dir(Path(path).parent)
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(epochs, history["train_loss"], marker="o", color="#3b6ea5")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].set_title("Training loss")
    balanced = [metrics.get("balanced_accuracy", np.nan) for metrics in history["val_metrics"]]
    axes[1].plot(epochs, balanced, marker="o", color="#a5533b")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("balanced accuracy")
    axes[1].set_title("Validation balanced accuracy")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return Path(path)


def save_heatmap_grid(samples, path, cols=3, title=None):
    """Save a grid of (image, heatmap, title) dicts with the heatmap overlaid."""
    if not samples:
        raise ValueError("No samples to plot")
    rows = int(np.ceil(len(samples) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for axis, sample in zip(axes, samples):
        axis.imshow(sample["image"])
        axis.set_title(sample.get("title", ""), fontsize=9)
        axis.axis("off")
    for axis in axes[len(samples):]:
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    ensure_dir(Path(path).parent)
    figure.savefig(path)
    plt.close(figure)
    return Path(path)
