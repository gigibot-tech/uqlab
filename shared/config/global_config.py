"""
Shared Configuration - Global settings, paths, and constants.

This module provides centralized configuration management:
- Global paths and directories
- Default hyperparameters
- System settings
- Environment configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..types import ModelArchitecture, NoiseType, TrainingMode, UncertaintyMethod


# ============================================================================
# Path Configuration
# ============================================================================

@dataclass
class PathConfig:
    """Global path configuration."""
    
    # Root directories
    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_root: Path = field(default_factory=lambda: Path("./data"))
    results_root: Path = field(default_factory=lambda: Path("./results"))
    cache_root: Path = field(default_factory=lambda: Path("./cache"))
    
    # Dataset paths
    cifar10n_root: Path = field(default_factory=lambda: Path("./data/cifar10n"))
    cifar10_root: Path = field(default_factory=lambda: Path("./data/cifar10"))
    
    # Cache directories
    feature_cache_dir: Path = field(default_factory=lambda: Path("./cache/features"))
    model_cache_dir: Path = field(default_factory=lambda: Path("./cache/models"))
    
    # Results directories
    experiments_dir: Path = field(default_factory=lambda: Path("./results/experiments"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("./results/checkpoints"))
    logs_dir: Path = field(default_factory=lambda: Path("./results/logs"))
    
    def __post_init__(self):
        """Ensure all paths are Path objects and create directories."""
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr = getattr(self, attr_name)
                if isinstance(attr, (str, Path)):
                    path = Path(attr)
                    setattr(self, attr_name, path)
    
    def create_directories(self, exist_ok: bool = True) -> None:
        """Create all configured directories."""
        dirs = [
            self.data_root,
            self.results_root,
            self.cache_root,
            self.feature_cache_dir,
            self.model_cache_dir,
            self.experiments_dir,
            self.checkpoints_dir,
            self.logs_dir,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=exist_ok)


# ============================================================================
# Data Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Dataset settings
    dataset_name: str = "cifar10n"
    noise_type: NoiseType = NoiseType.WORST
    num_classes: int = 10
    
    # Split configuration
    under_supported_classes: List[int] = field(default_factory=lambda: [3, 5])
    under_train_per_class: int = 10
    regular_train_per_class: int = 500
    eval_per_group: int = 600
    
    # Noise injection
    aleatoric_noise_percentage: float = 0.0  # 0-100
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Image preprocessing
    image_size: int = 32
    normalize_mean: tuple = (0.4914, 0.4822, 0.4465)
    normalize_std: tuple = (0.2470, 0.2435, 0.2616)


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    architecture: ModelArchitecture = ModelArchitecture.DINOV2_MLP
    training_mode: TrainingMode = TrainingMode.FEATURE_SPACE
    
    # Backbone settings
    backbone_name: str = "dinov2_vitb14"
    freeze_backbone: bool = True
    use_pretrained: bool = True
    
    # Head settings
    hidden_dim: int = 256
    num_hidden_layers: int = 1
    dropout: float = 0.2
    use_batch_norm: bool = False
    
    # Uncertainty settings
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.MC_DROPOUT
    mc_passes: int = 20
    ensemble_size: int = 5


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Optimization
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Batch sizes
    train_batch_size: int = 256
    eval_batch_size: int = 512
    feature_batch_size: int = 64
    
    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_epochs: int = 2
    min_lr: float = 1e-6
    
    # Regularization
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    save_best_only: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    
    # Metrics
    compute_auroc: bool = True
    compute_ude: bool = True
    compute_ece: bool = True
    compute_brier: bool = True
    
    # Uncertainty quantification
    mc_passes: int = 20
    temperature: float = 1.0
    
    # Signal computation
    compute_all_signals: bool = True
    top_k_signals: int = 10
    
    # Calibration
    calibration_bins: int = 15


# ============================================================================
# System Configuration
# ============================================================================

@dataclass
class SystemConfig:
    """System and runtime configuration."""
    
    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    gpu_id: int = 0
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Progress tracking
    use_tqdm: bool = True
    tqdm_ncols: int = 100
    
    # Debugging
    debug_mode: bool = False
    profile_memory: bool = False


# ============================================================================
# Global Configuration
# ============================================================================

@dataclass
class GlobalConfig:
    """Complete global configuration."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def __post_init__(self):
        """Initialize configuration."""
        self.paths.create_directories()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> GlobalConfig:
        """Create configuration from dictionary."""
        return cls(
            paths=PathConfig(**config_dict.get("paths", {})),
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            system=SystemConfig(**config_dict.get("system", {})),
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "paths": self.paths.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "system": self.system.__dict__,
        }


# ============================================================================
# Default Configuration Instance
# ============================================================================

# Global default configuration
DEFAULT_CONFIG = GlobalConfig()


# ============================================================================
# Configuration Helpers
# ============================================================================

def get_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> None:
    """Update global configuration with keyword arguments."""
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)


def reset_config() -> None:
    """Reset configuration to defaults."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = GlobalConfig()


# Made with Bob