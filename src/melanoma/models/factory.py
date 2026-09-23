"""Backbone factory built on timm."""

import timm

SUPPORTED_BACKBONES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]


def build_model(backbone: str, num_classes: int = 2, pretrained: bool = True, drop_rate: float = 0.0):
    """Create a timm model configured as a classifier.

    Raises ValueError for invalid num_classes, negative drop_rate, or an
    unknown backbone name.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2, got {num_classes}")
    if not 0.0 <= drop_rate <= 1.0:
        raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")
    if not isinstance(backbone, str) or not backbone:
        raise ValueError(f"backbone must be a non-empty string, got {backbone!r}")
    try:
        return timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate
        )
    except RuntimeError as exc:
        raise ValueError(f"Unknown or unsupported backbone '{backbone}': {exc}") from exc


def as_feature_extractor(model):
    """Return a copy-style wrapper that outputs pooled features (no classifier)."""
    try:
        model.reset_classifier(0)
    except (AttributeError, RuntimeError) as exc:
        raise ValueError("Model does not support reset_classifier for feature extraction") from exc
    return model
