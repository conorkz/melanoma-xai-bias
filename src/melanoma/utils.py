"""Shared utilities: seeds, devices, directories."""

import random
from pathlib import Path

import numpy as np
import torch


def get_device(preferred="auto"):
    """Resolve a torch device. 'auto' picks MPS, then CUDA, then CPU."""
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(preferred)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS device requested but not available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but not available")
    return device


def seed_everything(seed):
    """Seed python, numpy, and torch random number generators."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    """Create a directory (and parents) if missing, and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
