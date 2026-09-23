"""Class-balanced sampling and class weight utilities."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_balanced_weights(labels):
    """Per-sample weights proportional to the inverse class frequency."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("Cannot compute weights for empty labels")
    counts = np.bincount(labels)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Classes missing from labels: {missing}")
    return 1.0 / counts[labels]


def make_weighted_sampler(labels, num_samples=None, generator=None, replacement=True):
    """Build a WeightedRandomSampler from class labels."""
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    weights = class_balanced_weights(labels)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights) if num_samples is None else num_samples,
        replacement=replacement,
        generator=generator,
    )


def compute_class_weights(labels, mode="inverse"):
    """Class weights for loss weighting. Mode 'inverse' is normalized inverse frequency."""
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("Cannot compute class weights for empty labels")
    counts = np.bincount(labels)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Classes missing from labels: {missing}")
    if mode != "inverse":
        raise ValueError(f"Unknown mode {mode!r}, expected 'inverse'")
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)
