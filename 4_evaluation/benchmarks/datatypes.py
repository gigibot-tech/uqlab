"""
Core data types for UQ benchmarking.
Adapted from uq_disentanglement_comparison package.
"""

from dataclasses import dataclass, field
from typing import List, Callable, Iterable, Union, Optional

import numpy as np


@dataclass
class Dataset:
    """
    Standard dataset format for UQ experiments.
    Works with both Keras and PyTorch models.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    is_regression: bool = False
    
    # Additional metadata for noise tracking
    noise_mask: Optional[np.ndarray] = None  # Boolean mask: True = noisy label
    clean_labels: Optional[np.ndarray] = None  # Original clean labels (if available)
    noise_rate: float = 0.0  # Percentage of noisy labels


@dataclass
class UqModel:
    """Configuration for a UQ method."""
    model_function: Callable
    uq_name: str  # 'gaussian_logits', 'information_theoretic', 'dualxda'
    epochs: int
    framework: str = 'keras'  # 'keras' or 'pytorch'


@dataclass
class ExperimentConfig:
    """Configuration for a benchmark experiment."""
    dataset_name: str
    dataset: Optional[Dataset]
    models: List[UqModel]
    meta_experiments: List[str]  # ['label_noise', 'dataset_size', 'ood_class']


@dataclass
class UncertaintyResults:
    """
    Results from UQ evaluation.
    Used for plotting accuracy, aleatoric and epistemic uncertainty
    while changing a parameter (e.g., noise rate, dataset size).
    """
    accuracies: List[float] = field(default_factory=list)
    aleatoric_uncertainties: List[float] = field(default_factory=list)
    epistemic_uncertainties: List[float] = field(default_factory=list)
    changed_parameter_values: List[float] = field(default_factory=list)

    def append_values(self, accuracy, aleatoric_uncertainty, epistemic_uncertainty, parameter):
        """Add a single result point."""
        self.accuracies.append(accuracy)
        self.aleatoric_uncertainties.append(aleatoric_uncertainty)
        self.epistemic_uncertainties.append(epistemic_uncertainty)
        self.changed_parameter_values.append(parameter)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            'accuracies': list(self.accuracies),
            'aleatoric_uncertainties': list(self.aleatoric_uncertainties),
            'epistemic_uncertainties': list(self.epistemic_uncertainties),
            'changed_parameter_values': list(self.changed_parameter_values)
        }

# Made with Bob
