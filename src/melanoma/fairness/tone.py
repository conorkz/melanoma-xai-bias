"""Skin tone proxy from non-lesional skin using the Individual Typology Angle."""

import cv2
import numpy as np

ITA_GROUPS = ("light", "medium", "dark", "unknown")


def compute_ita(image_rgb, lesion_mask=None, min_pixels: int = 500) -> dict:
    """Compute median L*, b*, and ITA from non-lesional skin.

    The image must be HxWx3 RGB. If a lesion mask is given, the mask is
    dilated slightly and excluded from the skin sample. Very dark and very
    bright pixels are excluded to avoid vignette and specular highlights.
    Returns a dict with L, b, ita, and n_pixels. ITA is NaN when no valid
    skin pixels remain.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    b_channel = lab[:, :, 2]

    if lesion_mask is None:
        skin = np.ones(lightness.shape, dtype=bool)
    else:
        mask = np.asarray(lesion_mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != lightness.shape:
            mask = cv2.resize(
                mask,
                (lightness.shape[1], lightness.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if mask.max() > 1:
            binary = (mask > 127).astype(np.uint8)
        else:
            binary = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        lesion = cv2.dilate(binary, kernel)
        skin = lesion == 0

    valid = skin & (lightness >= 10.0) & (lightness <= 95.0)
    n_pixels = int(valid.sum())
    if n_pixels < min_pixels:
        return {
            "L": float("nan"),
            "b": float("nan"),
            "ita": float("nan"),
            "n_pixels": n_pixels,
        }

    median_l = float(np.median(lightness[valid]))
    median_b = float(np.median(b_channel[valid]))
    if median_b <= 0:
        ita = float("nan")
    else:
        ita = float(np.degrees(np.arctan((median_l - 50.0) / median_b)))
    return {"L": median_l, "b": median_b, "ita": ita, "n_pixels": n_pixels}


def ita_group(ita) -> str:
    """Map an ITA value to a coarse tone group.

    The thresholds approximate the light, intermediate, and dark ITA bands.
    These are image-based strata, not clinical Fitzpatrick labels.
    """
    if ita is None or not np.isfinite(ita):
        return "unknown"
    if ita > 41.0:
        return "light"
    if ita > 10.0:
        return "medium"
    return "dark"


def lightness_group(lightness) -> str:
    """Map median skin L* to a coarse lightness group.

    These thresholds are specific to the HAM10000 lightness distribution.
    They are image-based strata, not clinical Fitzpatrick labels. The L*
    proxy is more robust than ITA on dermatoscope images with color casts.
    """
    if lightness is None or not np.isfinite(lightness):
        return "unknown"
    if lightness > 70.0:
        return "light"
    if lightness > 55.0:
        return "medium"
    return "dark"
