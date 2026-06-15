"""
Shared Type Definitions - Common types, enums, and type aliases.

This module provides centralized type definitions used across all layers:
- Enums for categorical values
- Type aliases for clarity
- Common data structures
- Protocol definitions for interfaces
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

# ============================================================================
# Type Aliases
# ============================================================================

# Tensor types
TensorLike = Union[torch.Tensor, NDArray[np.float32]]
DeviceType = Union[str, torch.device]

# Path types
PathLike = Union[str, Path]

# Data types
LabelArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]

# Metrics types
MetricsDict = Dict[str, float]
SignalDict = Dict[str, FloatArray]

# Config types
ConfigDict = Dict[str, Any]


# ============================================================================
# Enums
# ============================================================================

class NoiseType(str, Enum):
    """CIFAR-10N noise types."""
    CLEAN = "clean"
    WORST = "worst"
    AGGRE = "aggre"
    RANDOM1 = "random1"
    RANDOM2 = "random2"
    RANDOM3 = "random3"


class SplitType(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    EVAL_CLEAN = "eval_clean"
    EVAL_ALEATORIC = "eval_aleatoric"
    EVAL_EPISTEMIC = "eval_epistemic"


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    DINOV2_MLP = "dinov2_mlp"
    RESNET18_MCDROPOUT = "resnet18_mcdropout"
    RESNET50_MCDROPOUT = "resnet50_mcdropout"
    CNN_MCDROPOUT = "cnn_mcdropout"
    SIMPLE_CNN = "simple_cnn"


class TrainingMode(str, Enum):
    """Training modes."""
    FEATURE_SPACE = "feature_space"
    END_TO_END = "end_to_end"


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""
    MC_DROPOUT = "mc_dropout"
    DEEP_ENSEMBLE = "deep_ensemble"
    MSP = "msp"
    TEMPERATURE_SCALING = "temperature_scaling"


class SignalType(str, Enum):
    """Uncertainty signal types."""
    # Probabilistic signals
    MSP_UNCERTAINTY = "msp_uncertainty"
    PREDICTIVE_ENTROPY = "predictive_entropy"
    MUTUAL_INFO = "mutual_info"
    
    # Attribution-based signals
    COHERENCE = "coherence"
    INVERSE_COHERENCE = "inverse_coherence"
    
    # Logit-based signals
    DOMINANCE = "dominance"
    INVERSE_MASS = "inverse_mass"
    INVERSE_LOGIT_MAGNITUDE = "inverse_logit_magnitude"
    
    # Variance-based
    PREDICTIVE_VARIANCE = "predictive_variance"


class MetricType(str, Enum):
    """Evaluation metric types."""
    ACCURACY = "accuracy"
    AUROC = "auroc"
    AUROC_EPISTEMIC = "auroc_epistemic"
    AUROC_ALEATORIC = "auroc_aleatoric"
    UDE = "ude"
    ECE = "ece"
    BRIER_SCORE = "brier_score"
    NLL = "nll"


class ExperimentStatus(str, Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SweepType(str, Enum):
    """Experiment sweep types."""
    DATASET_SIZE = "dataset_size"
    LABEL_NOISE = "label_noise"
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"


# ============================================================================
# Protocol Definitions (Interfaces)
# ============================================================================

class ModelProtocol(Protocol):
    """Protocol for model interface."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        ...
    
    def train(self, mode: bool = True) -> None:
        """Set training mode."""
        ...
    
    def eval(self) -> None:
        """Set evaluation mode."""
        ...


class DataLoaderProtocol(Protocol):
    """Protocol for data loader interface."""
    
    def __iter__(self):
        """Iterate over batches."""
        ...
    
    def __len__(self) -> int:
        """Number of batches."""
        ...


class CallbackProtocol(Protocol):
    """Protocol for training callbacks."""
    
    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        ...
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict) -> None:
        """Called at the end of each epoch."""
        ...
    
    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """Called after each batch."""
        ...


# ============================================================================
# Common Data Structures
# ============================================================================

class EvaluationGroup:
    """Evaluation group specification."""
    
    def __init__(
        self,
        name: str,
        indices: List[int],
        is_noisy: Optional[BoolArray] = None,
        clean_labels: Optional[LabelArray] = None,
        noisy_labels: Optional[LabelArray] = None,
    ):
        self.name = name
        self.indices = indices
        self.is_noisy = is_noisy
        self.clean_labels = clean_labels
        self.noisy_labels = noisy_labels
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __repr__(self) -> str:
        return f"EvaluationGroup(name={self.name}, size={len(self)})"


class PredictionResult:
    """Model prediction results."""
    
    def __init__(
        self,
        logits: FloatArray,
        probabilities: FloatArray,
        predictions: LabelArray,
        uncertainties: Optional[SignalDict] = None,
    ):
        self.logits = logits
        self.probabilities = probabilities
        self.predictions = predictions
        self.uncertainties = uncertainties or {}
    
    @property
    def num_samples(self) -> int:
        return len(self.predictions)
    
    @property
    def num_classes(self) -> int:
        return self.probabilities.shape[1] if len(self.probabilities.shape) > 1 else 1


# ============================================================================
# Constants
# ============================================================================

# Signal names in canonical order
SIGNAL_NAMES = [
    "msp_uncertainty",
    "predictive_entropy",
    "mutual_info",
    "inverse_coherence",
    "dominance",
    "inverse_mass",
    "inverse_logit_magnitude",
]

# Signal display labels
SIGNAL_LABELS = {
    "msp_uncertainty": "MSP",
    "predictive_entropy": "Predictive Entropy",
    "mutual_info": "Mutual Information",
    "inverse_coherence": "Inverse Coherence",
    "dominance": "Dominance",
    "inverse_mass": "Inverse Mass",
    "inverse_logit_magnitude": "Inverse Logit Magnitude",
}

# Epistemic signals (respond to dataset size, stable to noise)
EPISTEMIC_SIGNALS = ["inverse_mass", "dominance", "inverse_logit_magnitude"]

# Aleatoric signals (respond to noise, stable to dataset size)
ALEATORIC_SIGNALS = ["inverse_coherence"]

# Group order for visualizations
GROUP_ORDER = ("clean", "aleatoric_like", "epistemic_like")

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Default color schemes
COLOR_SCHEMES = {
    "groups": {
        "clean": "#2ecc71",
        "aleatoric_like": "#e74c3c",
        "epistemic_like": "#3498db",
    },
    "signals": {
        "msp_uncertainty": "#e74c3c",
        "predictive_entropy": "#9b59b6",
        "mutual_info": "#3498db",
        "inverse_coherence": "#e67e22",
        "dominance": "#1abc9c",
        "inverse_mass": "#f39c12",
        "inverse_logit_magnitude": "#34495e",
    },
    "metrics": {
        "accuracy": "#2ecc71",
        "auroc": "#3498db",
        "ude": "#e74c3c",
    },
}

# Made with Bob