"""Quantitative attribution metrics for lesion masks and model faithfulness."""

import cv2
import numpy as np
import torch


def _binary_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.max() > 1:
        return mask > 127
    return mask > 0


def attribution_mass_in_mask(heatmap, mask) -> float:
    """Fraction of total attribution mass that falls inside the mask."""
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    binary = _binary_mask(mask, heatmap.shape)
    total = float(heatmap.sum())
    if total <= 0:
        return float("nan")
    return float(heatmap[binary].sum()) / total


def pointing_game(heatmap, mask):
    """True when the maximum attribution falls inside the mask."""
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    if not np.isfinite(heatmap).all():
        raise ValueError("heatmap contains non-finite values")
    if heatmap.max() <= 0:
        return float("nan")
    binary = _binary_mask(mask, heatmap.shape)
    row, column = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return bool(binary[row, column])


def attribution_iou(heatmap, mask, fraction: float = 0.25) -> float:
    """IoU between the top-fraction attribution pixels and the mask."""
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    binary = _binary_mask(mask, heatmap.shape)
    flat = heatmap.ravel()
    k = max(1, int(round(fraction * flat.size)))
    top = np.argpartition(flat, -k)[-k:]
    attribution = np.zeros(flat.size, dtype=bool)
    attribution[top] = True
    attribution = attribution.reshape(heatmap.shape)
    union = np.logical_or(attribution, binary).sum()
    if union == 0:
        return float("nan")
    intersection = np.logical_and(attribution, binary).sum()
    return float(intersection / union)


def _target_probability(model, image_tensor, target_class: int, device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)
        return float(probabilities[0, target_class].cpu())


def deletion_insertion_auc(
    model,
    image_tensor,
    heatmap,
    target_class: int,
    steps: int = 20,
    device="cpu",
):
    """Deletion and insertion AUC for a single image and heatmap.

    The deletion curve removes the top-k attribution pixels and replaces them
    with the per-channel image mean. The insertion curve starts from the mean
    image and restores the top-k pixels. Returns both curves and their AUCs.
    """
    if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
        raise ValueError(
            f"image_tensor must be (1, C, H, W), got {tuple(image_tensor.shape)}"
        )
    heatmap = np.asarray(heatmap, dtype=np.float64)
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    if heatmap.shape != (image_tensor.size(2), image_tensor.size(3)):
        raise ValueError(
            f"heatmap shape {heatmap.shape} does not match image "
            f"{(image_tensor.size(2), image_tensor.size(3))}"
        )
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    device = torch.device(device)
    model.to(device)
    image = image_tensor.to(device)
    mean_image = image.mean(dim=(2, 3), keepdim=True).expand_as(image).clone()
    order = np.argsort(heatmap.ravel())[::-1]
    fractions = np.linspace(0.0, 1.0, steps + 1)

    deletion_probabilities = []
    insertion_probabilities = []
    for fraction in fractions:
        k = int(round(fraction * order.size))
        selected = np.zeros(order.size, dtype=bool)
        selected[order[:k]] = True
        mask = torch.from_numpy(selected.reshape(heatmap.shape)).to(device)

        deleted = image.clone()
        deleted[0, :, mask] = mean_image[0, :, mask]
        inserted = mean_image.clone()
        inserted[0, :, mask] = image[0, :, mask]

        deletion_probabilities.append(_target_probability(model, deleted, target_class, device))
        insertion_probabilities.append(_target_probability(model, inserted, target_class, device))

    if hasattr(np, "trapezoid"):
        trapezoid = np.trapezoid
    else:
        trapezoid = np.trapz
    return {
        "fractions": fractions.tolist(),
        "deletion": deletion_probabilities,
        "insertion": insertion_probabilities,
        "deletion_auc": float(trapezoid(deletion_probabilities, fractions)),
        "insertion_auc": float(trapezoid(insertion_probabilities, fractions)),
    }
