"""Lesion-skin contrast features from dermoscopic images and lesion masks."""

import cv2
import numpy as np


def _lab(image_rgb):
    """Convert RGB uint8 to CIELAB with L in 0-100 and a/b in -127..127."""
    rgb = np.asarray(image_rgb, dtype=np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def _binary_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask.max() > 1:
        return mask > 127
    return mask > 0


def _nan_result(lesion_area=0):
    return {
        "lesion_L": float("nan"),
        "skin_L": float("nan"),
        "delta_L": float("nan"),
        "delta_E": float("nan"),
        "lesion_area": int(lesion_area),
        "skin_pixels": 0,
    }


def lesion_contrast(image_rgb, mask, ring_pixels: int = 11):
    """Compute median CIELAB contrast between the lesion and a perilesional ring.

    ``delta_L`` is skin minus lesion, so a positive value means the lesion is
    darker than the surrounding skin. ``delta_E`` is the CIELAB 76 color
    difference between the two median colors. Very dark and very bright skin
    pixels are excluded because they are usually hair, rulers, or specular
    highlights rather than skin.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"image_rgb must be HxWx3, got shape {image.shape}")
    if ring_pixels < 1:
        raise ValueError(f"ring_pixels must be >= 1, got {ring_pixels}")

    lab = _lab(image[:, :, :3])
    binary = _binary_mask(mask, lab.shape[:2])
    lesion_area = int(binary.sum())
    if lesion_area == 0:
        return _nan_result()

    lesion_pixels = lab[binary]
    lesion_L = float(np.median(lesion_pixels[:, 0]))
    lesion_a = float(np.median(lesion_pixels[:, 1]))
    lesion_b = float(np.median(lesion_pixels[:, 2]))

    size = 2 * ring_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    dilated = cv2.dilate(binary.astype(np.uint8), kernel) > 0
    ring = dilated & ~binary
    skin_pixels = lab[ring]
    if skin_pixels.size == 0:
        return _nan_result(lesion_area)

    valid = (skin_pixels[:, 0] >= 10.0) & (skin_pixels[:, 0] <= 95.0)
    skin_pixels = skin_pixels[valid]
    if skin_pixels.size == 0:
        return _nan_result(lesion_area)

    skin_L = float(np.median(skin_pixels[:, 0]))
    skin_a = float(np.median(skin_pixels[:, 1]))
    skin_b = float(np.median(skin_pixels[:, 2]))

    return {
        "lesion_L": lesion_L,
        "skin_L": skin_L,
        "delta_L": skin_L - lesion_L,
        "delta_E": float(
            np.sqrt(
                (lesion_L - skin_L) ** 2
                + (lesion_a - skin_a) ** 2
                + (lesion_b - skin_b) ** 2
            )
        ),
        "lesion_area": lesion_area,
        "skin_pixels": int(skin_pixels.shape[0]),
    }
