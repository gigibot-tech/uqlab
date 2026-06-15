"""
Configuration schema with validation using dataclasses.

This module provides type-safe configuration classes with validation,
while maintaining backward compatibility with dict-based configs.

Usage:
    # New way (with validation):
    config = ExperimentConfig.from_dict(config_dict)
    config.validate()
    
    # Old way (still works):
    config_dict = {...}  # Use directly as before
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class DataConfig:
    """Data loading and sampling configuration."""
    
    noise_type: str = "worse_label"
    under_supported_classes: str = "3,5"
    under_train_per_class: int = 50
    regular_train_per_class: int = 300
    eval_per_group: int = 600
    
    def validate(self) -> None:
        """Validate data configuration."""
        # Validate noise type
        valid_noise_types = ["worse_label", "aggre_label", "random_label1", "random_label2", "random_label3"]
        if self.noise_type not in valid_noise_types:
            raise ValueError(
                f"Invalid noise_type: {self.noise_type}. "
                f"Must be one of {valid_noise_types}"
            )
        
        # Validate sample counts
        if self.under_train_per_class <= 0:
            raise ValueError(f"under_train_per_class must be > 0, got {self.under_train_per_class}")
        
        if self.regular_train_per_class <= 0:
            raise ValueError(f"regular_train_per_class must be > 0, got {self.regular_train_per_class}")
        
        if self.eval_per_group <= 0:
            raise ValueError(f"eval_per_group must be > 0, got {self.eval_per_group}")
        
        # Validate under_supported_classes format
        if self.under_supported_classes.startswith("random:"):
            try:
                num_classes = int(self.under_supported_classes.split(":")[1])
                if not 0 < num_classes < 10:
                    raise ValueError(f"Random class count must be between 1 and 9, got {num_classes}")
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f"Invalid random format: {self.under_supported_classes}. "
                    f"Expected 'random:N' where N is 1-9"
                ) from e
        else:
            # Validate comma-separated class IDs
            try:
                class_ids = [int(x.strip()) for x in self.under_supported_classes.split(",")]
                if not all(0 <= cid < 10 for cid in class_ids):
                    raise ValueError("Class IDs must be between 0 and 9")
                if len(class_ids) != len(set(class_ids)):
                    raise ValueError("Duplicate class IDs found")
            except ValueError as e:
                raise ValueError(
                    f"Invalid under_supported_classes: {self.under_supported_classes}. "
                    f"Expected comma-separated integers (0-9) or 'random:N'"
                ) from e


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    dinov2_model: str = "small"
    hidden_dim: int = 256
    dropout: float = 0.2
    use_untrained_resnet: bool = False
    
    def validate(self) -> None:
        """Validate model configuration."""
        valid_models = ["small", "base", "large", "giant"]
        if self.dinov2_model not in valid_models:
            raise ValueError(
                f"Invalid dinov2_model: {self.dinov2_model}. "
                f"Must be one of {valid_models}"
            )
        
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    
    epochs: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    train_batch_size: int = 256
    feature_batch_size: int = 64
    
    def validate(self) -> None:
        """Validate training configuration."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        
        if self.train_batch_size <= 0:
            raise ValueError(f"train_batch_size must be > 0, got {self.train_batch_size}")
        
        if self.feature_batch_size <= 0:
            raise ValueError(f"feature_batch_size must be > 0, got {self.feature_batch_size}")


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    mc_passes: int = 20
    top_k: int = 10
    attribution_method: str = "dualxda"
    
    def validate(self) -> None:
        """Validate evaluation configuration."""
        if self.mc_passes <= 0:
            raise ValueError(f"mc_passes must be > 0, got {self.mc_passes}")
        
        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {self.top_k}")
        
        valid_methods = ["dualxda"]
        if self.attribution_method not in valid_methods:
            raise ValueError(
                f"Invalid attribution_method: {self.attribution_method}. "
                f"Must be one of {valid_methods}"
            )


@dataclass
class PathsConfig:
    """File paths configuration."""
    
    cifar10n_root: str = "./data/cifar10n"
    feature_cache_dir: str = "./cache/fast_uncertainty_classification/features"
    results_base_dir: str = "./results"
    
    def validate(self) -> None:
        """Validate paths configuration."""
        # Just check they're not empty
        if not self.cifar10n_root:
            raise ValueError("cifar10n_root cannot be empty")
        if not self.feature_cache_dir:
            raise ValueError("feature_cache_dir cannot be empty")
        if not self.results_base_dir:
            raise ValueError("results_base_dir cannot be empty")
    
    def ensure_directories(self) -> None:
        """Create directories if they don't exist."""
        Path(self.feature_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_base_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with validation.
    
    This class provides type-safe configuration with validation,
    while maintaining backward compatibility with dict-based configs.
    
    Example:
        >>> # From dict (backward compatible)
        >>> config = ExperimentConfig.from_dict(config_dict)
        >>> config.validate()
        >>> 
        >>> # To dict (for YAML export)
        >>> config_dict = config.to_dict()
        >>> 
        >>> # From YAML file
        >>> config = ExperimentConfig.from_yaml("config.yaml")
    """
    
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    def validate(self) -> None:
        """Validate entire configuration."""
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(
                f"Invalid device: {self.device}. "
                f"Must be one of {valid_devices} or 'cuda:N'"
            )
        
        # Validate sub-configs
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        self.paths.validate()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Create config from dictionary (backward compatible).
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig instance
            
        Example:
            >>> config_dict = {
            ...     "seed": 42,
            ...     "data": {"noise_type": "worse_label"},
            ...     "model": {"dinov2_model": "small"}
            ... }
            >>> config = ExperimentConfig.from_dict(config_dict)
        """
        # Extract top-level fields
        seed = config_dict.get("seed", 42)
        device = config_dict.get("device", "auto")
        
        # Extract nested configs
        data = DataConfig(**config_dict.get("data", {}))
        model = ModelConfig(**config_dict.get("model", {}))
        training = TrainingConfig(**config_dict.get("training", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        paths = PathsConfig(**config_dict.get("paths", {}))
        
        return cls(
            seed=seed,
            device=device,
            data=data,
            model=model,
            training=training,
            evaluation=evaluation,
            paths=paths,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Configuration as nested dictionary
        """
        return {
            "seed": self.seed,
            "device": self.device,
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "paths": asdict(self.paths),
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> ExperimentConfig:
        """Load config from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save config to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Backward compatibility: Allow using dict directly
def validate_config_dict(config_dict: Dict[str, Any]) -> None:
    """Validate a configuration dictionary without converting to dataclass.
    
    This function provides validation for existing dict-based configs
    without requiring code changes.
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
        
    Example:
        >>> config_dict = {...}
        >>> validate_config_dict(config_dict)  # Raises if invalid
        >>> # Use config_dict as before
    """
    config = ExperimentConfig.from_dict(config_dict)
    config.validate()


# Made with Bob