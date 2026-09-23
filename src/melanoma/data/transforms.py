"""Albumentations transform pipelines for training and evaluation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    img_size: int,
    train: bool,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    strong: bool = False,
):
    """Build an albumentations pipeline returning normalized CHW tensors.

    Train pipelines add geometric and photometric augmentation. The 'strong'
    regime increases photometric magnitudes to simulate domain shift.
    """
    if not isinstance(img_size, int) or img_size <= 0:
        raise ValueError(f"img_size must be a positive integer, got {img_size!r}")
    scale = 2.0 if strong else 1.0
    ops = [A.Resize(img_size, img_size)]
    if train:
        ops.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2 * scale,
                    contrast=0.2 * scale,
                    saturation=0.2 * scale,
                    hue=0.05 * scale,
                    p=0.7,
                ),
            ]
        )
        if strong:
            ops.append(A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-20, 20), p=0.5))
    ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(ops)
