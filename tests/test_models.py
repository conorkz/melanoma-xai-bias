"""Tests for melanoma.models.factory."""

import pytest
import torch

from melanoma.models.factory import as_feature_extractor, build_model


def test_forward_shape_binary():
    model = build_model("resnet18", num_classes=2, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 2)


def test_forward_shape_multiclass():
    model = build_model("resnet18", num_classes=7, pretrained=False)
    output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 7)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError):
        build_model("resnet9999", pretrained=False)


@pytest.mark.parametrize("bad_classes", [0, 1, -3])
def test_invalid_num_classes_raises(bad_classes):
    with pytest.raises(ValueError):
        build_model("resnet18", num_classes=bad_classes, pretrained=False)


@pytest.mark.parametrize("bad_drop", [-0.1, 1.5])
def test_invalid_drop_rate_raises(bad_drop):
    with pytest.raises(ValueError):
        build_model("resnet18", drop_rate=bad_drop, pretrained=False)


@pytest.mark.parametrize("bad_backbone", ["", None, 123])
def test_invalid_backbone_type_raises(bad_backbone):
    with pytest.raises(ValueError):
        build_model(bad_backbone, pretrained=False)


def test_feature_extractor_output():
    model = as_feature_extractor(build_model("resnet18", num_classes=2, pretrained=False))
    features = model(torch.randn(2, 3, 64, 64))
    assert features.shape == (2, 512)
