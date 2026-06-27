"""
Data Preprocessing and Transforms - Unified transform definitions.

This module consolidates all data preprocessing functionality:
- CIFAR-10 standard transforms
- DINOv2 transforms (224x224 with ImageNet normalization)
- Data augmentation strategies
- Transform utilities

Consolidates from:
- src/data/dinov2_transforms.py
- Various transform definitions scattered across the codebase
"""

from typing import Tuple

import torch
from torchvision import transforms


class RepeatChannels:
    """Picklable 1→N channel repeat (DataLoader workers require picklable transforms)."""

    def __init__(self, times: int = 3) -> None:
        self.times = times

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(self.times, 1, 1)


# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# MNIST normalization (grayscale, repeated to 3ch in transforms)
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_MEAN_3CH = (0.1307, 0.1307, 0.1307)
MNIST_STD_3CH = (0.3081, 0.3081, 0.3081)


def get_mnist_transforms(*, repeat_channels: int = 3) -> transforms.Compose:
    """MNIST → tensor, optional 3-channel repeat, resize to 32 for ResNet path."""
    steps = [transforms.ToTensor(), transforms.Resize((32, 32))]
    if repeat_channels == 3:
        steps.append(RepeatChannels(3))
        steps.append(transforms.Normalize(MNIST_MEAN_3CH, MNIST_STD_3CH))
    else:
        steps.append(transforms.Normalize(MNIST_MEAN, MNIST_STD))
    return transforms.Compose(steps)


def get_dataset_image_transform(dataset_name: str) -> transforms.Compose:
    """End-to-end image transforms keyed by dataset registry name."""
    name = (dataset_name or "cifar10").lower()
    if name in ("mnist", "fashion_mnist"):
        return get_mnist_transforms(repeat_channels=3)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ImageNet normalization constants (for DINOv2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_cifar10_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get CIFAR-10 transforms for training and testing.
    
    Args:
        augment: Whether to use data augmentation for training
        
    Returns:
        train_transform, test_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    return train_transform, test_transform


def get_dinov2_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get DINOv2-compatible transforms (224x224 with ImageNet normalization).
    
    DINOv2 requires:
    - 224x224 spatial resolution (via resize + crop)
    - ImageNet normalization
    
    If we accidentally evaluate a DINOv2-head checkpoint on 32x32 CIFAR tensors
    normalized with CIFAR statistics, accuracy collapses and UQ metrics become
    meaningless (risk–coverage curves saturate near 1.0 risk).
    
    Args:
        augment: Whether to use data augmentation for training
        
    Returns:
        train_transform, test_transform
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, test_transform


def get_augmentation_transforms(
    augmentation_type: str = 'standard',
    image_size: int = 32,
) -> transforms.Compose:
    """
    Get augmentation transforms for different strategies.
    
    Args:
        augmentation_type: Type of augmentation ('standard', 'strong', 'autoaugment')
        image_size: Target image size
        
    Returns:
        Augmentation transform
    """
    if augmentation_type == 'standard':
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    elif augmentation_type == 'strong':
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(15),
        ])
    
    elif augmentation_type == 'autoaugment':
        return transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        ])
    
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


def get_test_time_augmentation_transforms(
    n_augmentations: int = 5,
    image_size: int = 32,
) -> list:
    """
    Get test-time augmentation transforms for uncertainty estimation.
    
    Args:
        n_augmentations: Number of augmentation variants
        image_size: Target image size
        
    Returns:
        List of augmentation transforms
    """
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    augmentation_transforms = []
    
    for i in range(n_augmentations):
        if i == 0:
            # First one is no augmentation
            augmentation_transforms.append(base_transform)
        else:
            # Add slight variations
            augmentation_transforms.append(transforms.Compose([
                transforms.RandomCrop(image_size, padding=2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
            ]))
    
    return augmentation_transforms


def denormalize_cifar10(tensor):
    """
    Denormalize CIFAR-10 tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = tensor.new_tensor(CIFAR10_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(CIFAR10_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def denormalize_imagenet(tensor):
    """
    Denormalize ImageNet tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    mean = tensor.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


class MixUp:
    """
    MixUp data augmentation.
    
    Reference: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize MixUp.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        Apply MixUp to a batch.
        
        Args:
            x: Input batch [B, C, H, W]
            y: Labels [B]
            
        Returns:
            mixed_x, mixed_y, lam
        """
        import torch
        import numpy as np
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix data augmentation.
    
    Reference: CutMix: Regularization Strategy to Train Strong Classifiers (Yun et al., 2019)
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x, y):
        """
        Apply CutMix to a batch.
        
        Args:
            x: Input batch [B, C, H, W]
            y: Labels [B]
            
        Returns:
            mixed_x, mixed_y, lam
        """
        import torch
        import numpy as np
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get random box
        _, _, H, W = x.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def get_transform_by_name(name: str, **kwargs) -> transforms.Compose:
    """
    Get transform by name for easy configuration.
    
    Args:
        name: Transform name ('cifar10', 'dinov2', 'cifar10_augmented', etc.)
        **kwargs: Additional arguments
        
    Returns:
        Transform
    """
    if name == 'cifar10':
        train_transform, _ = get_cifar10_transforms(augment=False)
        return train_transform
    
    elif name == 'cifar10_augmented':
        train_transform, _ = get_cifar10_transforms(augment=True)
        return train_transform
    
    elif name == 'dinov2':
        train_transform, _ = get_dinov2_transforms(augment=False)
        return train_transform
    
    elif name == 'dinov2_augmented':
        train_transform, _ = get_dinov2_transforms(augment=True)
        return train_transform
    
    else:
        raise ValueError(f"Unknown transform name: {name}")

# Made with Bob
