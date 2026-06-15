"""
Data Layer - Unified data loading and preprocessing.

This module consolidates all data-related functionality:
- Dataset loaders (CIFAR-10, CIFAR-10N, SVHN)
- Preprocessing and transforms
- Dataset statistics and analysis
- Split management for experiments
"""

from uqlab.data_loaders import (
    CIFAR10NDataset,
    CIFAR10NLabelView,
    NOISE_TYPE_ALIASES,
    get_cifar10n_loaders,
    normalize_noise_type,
)
from .loaders import (
    SplitSpec,
    sample_indices_for_fast_pilot,
    extract_features_for_indices,
    EmbeddingOrganizer,
)
from .preprocessing import (
    get_cifar10_transforms,
    get_dinov2_transforms,
    get_augmentation_transforms,
)
from .stats import (
    compute_dataset_statistics,
    analyze_label_distribution,
    compute_noise_statistics,
)

__all__ = [
    # Loaders
    "CIFAR10NDataset",
    "CIFAR10NLabelView",
    "NOISE_TYPE_ALIASES",
    "normalize_noise_type",
    "get_cifar10n_loaders",
    "SplitSpec",
    "sample_indices_for_fast_pilot",
    "extract_features_for_indices",
    "EmbeddingOrganizer",
    # Preprocessing
    "get_cifar10_transforms",
    "get_dinov2_transforms",
    "get_augmentation_transforms",
    # Statistics
    "compute_dataset_statistics",
    "analyze_label_distribution",
    "compute_noise_statistics",
]

# Made with Bob
