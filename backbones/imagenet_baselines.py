"""
Classic ImageNet-pretrained baselines (VGG-16, ResNet-50) for downstream datasets.

These are meant for "classic reviewer" comparisons:
  - backbone pretrained on ImageNet
  - downstream head trained on target dataset (e.g., CIFAR-100)
  - MSP (1 - max softmax prob) as uncertainty for risk–coverage (AURC)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _freeze_all(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def build_resnet50_linear_probe(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    ResNet-50 backbone pretrained on ImageNet + linear probe head for num_classes.
    """
    from torchvision.models import resnet50

    weights = None
    if pretrained:
        try:
            from torchvision.models import ResNet50_Weights

            weights = ResNet50_Weights.IMAGENET1K_V2
        except Exception:
            # Older torchvision: weights arg may not exist; will fall back to pretrained=True.
            weights = None

    try:
        model = resnet50(weights=weights)
    except TypeError:
        model = resnet50(pretrained=pretrained)

    _freeze_all(model)
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def build_vgg16_linear_probe(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    VGG-16 backbone pretrained on ImageNet + linear probe head for num_classes.
    """
    from torchvision.models import vgg16

    weights = None
    if pretrained:
        try:
            from torchvision.models import VGG16_Weights

            weights = VGG16_Weights.IMAGENET1K_V1
        except Exception:
            weights = None

    try:
        model = vgg16(weights=weights)
    except TypeError:
        model = vgg16(pretrained=pretrained)

    _freeze_all(model)
    # VGG classifier last layer is a Linear.
    last = model.classifier[-1]
    if not isinstance(last, nn.Linear):
        raise RuntimeError("Unexpected VGG16 classifier head layout; expected last layer to be nn.Linear.")
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    for p in model.classifier[-1].parameters():
        p.requires_grad = True
    return model

