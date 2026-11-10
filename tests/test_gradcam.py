"""Tests for melanoma.xai.gradcam."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from melanoma.models.factory import build_model
from melanoma.xai.gradcam import (
    GRADCAM_TARGETS,
    compute_gradcam,
    get_target_layer,
    overlay_heatmap,
    resolve_layer,
    vit_reshape_transform,
)


@pytest.fixture(scope="module")
def resnet():
    model = build_model("resnet18", num_classes=2, pretrained=False)
    model.eval()
    return model


def test_resolve_layer_valid(resnet):
    layer = resolve_layer(resnet, "layer4[-1]")
    assert layer is resnet.layer4[-1]
    assert resolve_layer(resnet, "conv1") is resnet.conv1


def test_resolve_layer_bad_name_raises(resnet):
    with pytest.raises(ValueError):
        resolve_layer(resnet, "not_a_layer")


def test_resolve_layer_index_out_of_range_raises(resnet):
    with pytest.raises(ValueError):
        resolve_layer(resnet, "layer4[99]")


def test_resolve_layer_bad_index_text_raises(resnet):
    with pytest.raises(ValueError):
        resolve_layer(resnet, "layer4[abc]")


def test_get_target_layer_known_backbone(resnet):
    layer, reshape = get_target_layer(resnet, backbone="resnet18")
    assert layer is resnet.layer4[-1]
    assert reshape is None


def test_get_target_layer_explicit_path(resnet):
    layer, reshape = get_target_layer(resnet, backbone=None, layer_path="layer3[-1]")
    assert layer is resnet.layer3[-1]
    assert reshape is None


def test_get_target_layer_unknown_backbone_raises(resnet):
    with pytest.raises(ValueError):
        get_target_layer(resnet, backbone="mystery_net")


def test_get_target_layer_unsupported_backbone_raises(resnet):
    assert GRADCAM_TARGETS["swin_tiny_patch4_window7_224"] is None
    with pytest.raises(ValueError):
        get_target_layer(resnet, backbone="swin_tiny_patch4_window7_224")


def test_compute_gradcam_shape_and_range(resnet):
    layer, _ = get_target_layer(resnet, backbone="resnet18")
    heatmap = compute_gradcam(resnet, torch.randn(1, 3, 64, 64), layer)
    assert heatmap.shape == (64, 64)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_compute_gradcam_nonzero_for_positive_attribution():
    class TinyCAM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
            self.fc = nn.Linear(1, 2)
            with torch.no_grad():
                self.conv.weight.fill_(1.0)
                self.conv.bias.zero_()
                self.fc.weight.copy_(torch.tensor([[1.0], [-1.0]]))
                self.fc.bias.zero_()

        def forward(self, x):
            features = self.conv(x)
            pooled = features.mean(dim=(2, 3))
            return self.fc(pooled)

    model = TinyCAM()
    heatmap = compute_gradcam(model, torch.ones(1, 3, 8, 8), model.conv, target_class=0)
    assert heatmap.shape == (8, 8)
    assert heatmap.max() > 0.0


def test_compute_gradcam_explicit_target_classes(resnet):
    layer, _ = get_target_layer(resnet, backbone="resnet18")
    for target in (0, 1):
        heatmap = compute_gradcam(resnet, torch.randn(1, 3, 64, 64), layer, target_class=target)
        assert heatmap.shape == (64, 64)


def test_compute_gradcam_rejects_batch(resnet):
    layer, _ = get_target_layer(resnet, backbone="resnet18")
    with pytest.raises(ValueError):
        compute_gradcam(resnet, torch.randn(2, 3, 64, 64), layer)


def test_compute_gradcam_rejects_3d_input(resnet):
    layer, _ = get_target_layer(resnet, backbone="resnet18")
    with pytest.raises(ValueError):
        compute_gradcam(resnet, torch.randn(3, 64, 64), layer)


def test_vit_reshape_transform():
    tokens = torch.randn(2, 197, 8)
    reshaped = vit_reshape_transform(tokens)
    assert reshaped.shape == (2, 8, 14, 14)


def test_vit_reshape_invalid_token_count_raises():
    with pytest.raises(ValueError):
        vit_reshape_transform(torch.randn(2, 100, 8))


def test_vit_reshape_2d_input_raises():
    with pytest.raises(ValueError):
        vit_reshape_transform(torch.randn(2, 197))


def test_vit_gradcam_end_to_end():
    model = build_model("vit_small_patch16_224", num_classes=2, pretrained=False)
    layer, reshape = get_target_layer(model, backbone="vit_small_patch16_224")
    heatmap = compute_gradcam(
        model, torch.randn(1, 3, 224, 224), layer, reshape_transform=reshape
    )
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_overlay_heatmap_output(resnet):
    image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    heatmap = np.random.default_rng(1).random((16, 16)).astype(np.float32)
    overlay = overlay_heatmap(image, heatmap, alpha=0.5)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay, image)


def test_overlay_heatmap_invalid_alpha():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    heatmap = np.zeros((16, 16), dtype=np.float32)
    for bad_alpha in (-0.1, 1.5):
        with pytest.raises(ValueError):
            overlay_heatmap(image, heatmap, alpha=bad_alpha)


def test_overlay_heatmap_invalid_shapes():
    with pytest.raises(ValueError):
        overlay_heatmap(np.zeros((16, 16), dtype=np.uint8), np.zeros((16, 16), dtype=np.float32))
    with pytest.raises(ValueError):
        overlay_heatmap(np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((16, 16, 2), dtype=np.float32))
