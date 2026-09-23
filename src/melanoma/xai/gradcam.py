"""Grad-CAM heatmaps via Captum with support for CNN and ViT backbones."""

import cv2
import numpy as np
import torch
from captum.attr import LayerAttribution, LayerGradCam

GRADCAM_TARGETS = {
    "resnet18": "layer4[-1]",
    "resnet50": "layer4[-1]",
    "efficientnet_b0": "conv_head",
    "convnext_tiny": "stages[-1].blocks[-1]",
    "vit_small_patch16_224": "blocks[-1].norm1",
    "swin_tiny_patch4_window7_224": None,
}


def vit_reshape_transform(tensor):
    """Reshape transformer tokens [B, N, C] into spatial maps [B, C, H, W]."""
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D token tensor, got shape {tuple(tensor.shape)}")
    n_tokens = tensor.size(1)
    if n_tokens < 2:
        raise ValueError(f"Need at least 2 tokens (class token plus patches), got {n_tokens}")
    grid = int(round((n_tokens - 1) ** 0.5))
    if grid * grid != n_tokens - 1:
        raise ValueError(f"Cannot infer square token grid from {n_tokens} tokens")
    result = tensor[:, 1:, :].reshape(tensor.size(0), grid, grid, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def resolve_layer(model, path):
    """Resolve a dotted layer path with optional integer indexing, e.g. 'layer4[-1]'."""
    current = model
    for part in path.split("."):
        name, index = part, None
        if part.endswith("]") and "[" in part:
            name, index_text = part[:-1].split("[", 1)
            try:
                index = int(index_text)
            except ValueError as exc:
                raise ValueError(f"Invalid layer index in '{part}'") from exc
        if not name or not hasattr(current, name):
            raise ValueError(f"Layer '{part}' not found in model")
        current = getattr(current, name)
        if index is not None:
            try:
                current = current[index]
            except (IndexError, TypeError) as exc:
                raise ValueError(f"Layer index out of range in '{part}'") from exc
    return current


def get_target_layer(model, backbone=None, layer_path=None):
    """Return (target_layer, reshape_transform) for a model.

    An explicit layer_path wins over the built-in per-backbone mapping.
    """
    if layer_path:
        return resolve_layer(model, layer_path), None
    if backbone is None or backbone not in GRADCAM_TARGETS:
        raise ValueError(
            f"No built-in Grad-CAM target for backbone {backbone!r}. "
            "Pass layer_path explicitly."
        )
    target = GRADCAM_TARGETS[backbone]
    if target is None:
        raise ValueError(f"Grad-CAM target for {backbone!r} is not supported yet")
    reshape = vit_reshape_transform if backbone.startswith("vit_") else None
    return resolve_layer(model, target), reshape


def compute_gradcam(
    model,
    input_tensor,
    target_layer,
    target_class=None,
    reshape_transform=None,
):
    """Compute a normalized Grad-CAM heatmap for a single image batch.

    Returns a float32 array of shape (H, W) with values in [0, 1]. The input
    must be a 4D tensor of shape (1, C, H, W).
    """
    if input_tensor.dim() != 4:
        raise ValueError(f"input_tensor must be 4D (B, C, H, W), got {tuple(input_tensor.shape)}")
    if input_tensor.size(0) != 1:
        raise ValueError(f"compute_gradcam expects batch size 1, got {input_tensor.size(0)}")

    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    if target_class is None:
        with torch.no_grad():
            target_class = int(model(input_tensor).argmax(dim=1).item())

    attribution = LayerGradCam(model, target_layer)
    token_style = reshape_transform is not None
    with torch.enable_grad():
        try:
            heatmap = attribution.attribute(
                input_tensor,
                target=target_class,
                relu_attributions=True,
                attr_dim_summation=not token_style,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"Grad-CAM attribution failed: {exc}") from exc

    if token_style:
        if heatmap.dim() != 3:
            raise RuntimeError(
                f"Expected token-style attribution (B, N, C), got {tuple(heatmap.shape)}"
            )
        heatmap = reshape_transform(heatmap)
        if heatmap.dim() != 4:
            raise ValueError(
                "reshape_transform must map token attributions to (B, C, H, W), "
                f"got shape {tuple(heatmap.shape)}"
            )
        heatmap = heatmap.mean(dim=1, keepdim=True)
    elif heatmap.dim() == 4:
        heatmap = heatmap.mean(dim=1, keepdim=True)
    elif heatmap.dim() == 3:
        raise ValueError(
            "Layer returned token-style attributions; provide a reshape_transform"
        )
    else:
        raise RuntimeError(f"Unexpected attribution shape {tuple(heatmap.shape)}")

    resized = LayerAttribution.interpolate(
        heatmap.detach(), (input_tensor.size(2), input_tensor.size(3)), interpolate_mode="bilinear"
    )
    array = resized[0, 0].cpu().numpy().astype(np.float32)
    array = np.maximum(array, 0.0)
    max_value = float(array.max())
    if max_value > 0:
        array = array / max_value
    return array


def overlay_heatmap(image_rgb, heatmap, alpha: float = 0.45):
    """Blend a heatmap over an RGB image using the JET colormap."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    image_rgb = np.asarray(image_rgb)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got shape {image_rgb.shape}")
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {heatmap.shape}")
    if heatmap.shape != image_rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored_bgr = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    blended = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)
