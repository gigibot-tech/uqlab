"""
Data Layer - Unified data loading and preprocessing.

This module consolidates all data-related functionality:
- Dataset loaders (CIFAR-10, CIFAR-10N, SVHN)
- Preprocessing and transforms
- Dataset statistics and analysis
- Split management for experiments
"""

# Import from loaders directory
from uqlab.data.loaders import (
    CIFAR10NDataset,
    CIFAR10NLabelView,
    NOISE_TYPE_ALIASES,
    get_cifar10n_loaders,
    normalize_noise_type,
)

# Import from loaders.py file (legacy, should be in evaluation/classification)
# These are actually in uq_classification.data_loader
try:
    from uqlab.evaluation.classification.data_loader import (
        SplitSpec,
        sample_indices_for_fast_pilot,
        extract_features_for_indices,
        EmbeddingOrganizer,
    )
except ImportError:
    # Fallback: these might not be needed for all use cases
    SplitSpec = None
    sample_indices_for_fast_pilot = None
    extract_features_for_indices = None
    EmbeddingOrganizer = None
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
