"""
Utility functions for uncertainty classification experiments.

Provides helper functions for:
- Random seed management
- Device selection (CPU/CUDA/MPS)
- Data transformations for DINOv2
"""

import random
import torch
import numpy as np
from torchvision import transforms


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device(requested: str) -> torch.device:
    """
    Automatically select the best available device.
    
    Priority: requested device > CUDA > MPS > CPU
    
    Args:
        requested: Device string ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: Selected device
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dino_transform() -> transforms.Compose:
    """
    Create standard DINOv2 image transformation pipeline.
    
    Uses ImageNet normalization as DINOv2 was trained on ImageNet.
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

# Made with Bob
