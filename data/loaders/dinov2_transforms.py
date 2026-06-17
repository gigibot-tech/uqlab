"""
DINOv2 input transforms.

Our DINOv2 wrapper forwards tensors directly as `pixel_values` to the
Transformers model. That means we must match the expected preprocessing:
  - 224x224 spatial resolution (via resize + crop)
  - ImageNet normalization

If we accidentally evaluate a DINOv2-head checkpoint on 32x32 CIFAR tensors
normalized with CIFAR statistics, accuracy collapses and UQ metrics become
meaningless (risk–coverage curves saturate near 1.0 risk).
"""

from __future__ import annotations


def dinov2_eval_transform():
    """Deterministic eval transform (resize + center crop + ImageNet norm)."""
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )


def dinov2_train_transform():
    """Train transform for DINOv2-head fine-tuning on downstream labels."""
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

