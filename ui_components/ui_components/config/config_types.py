"""
Configuration Type Definitions

This module provides type-safe dataclass definitions for experiment configuration
objects, replacing dictionary-based configurations throughout the validation
framework and unified builder.

Benefits:
- Type safety: IDE catches typos at write-time
- Autocomplete: IDE suggests available fields
- Documentation: Dataclass fields are self-documenting
- Validation: Can add __post_init__ validation
- Cleaner code: No more config.get('key', default)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ValidationConfig:
    """Configuration for experiment validation framework."""
    validation_enabled: bool = False
    is_epistemic_sweep: bool = False
    is_aleatoric_sweep: bool = False
    epistemic_parameter: Optional[str] = None
    aleatoric_parameter: Optional[str] = None
    
    def get_badge(self) -> str:
        """Get badge string for experiment type."""
        if not self.validation_enabled:
            return ""
        if self.is_epistemic_sweep and self.is_aleatoric_sweep:
            return "📊 2D Grid"
        elif self.is_epistemic_sweep:
            return "🔬 Epistemic"
        elif self.is_aleatoric_sweep:
            return "🎲 Aleatoric"
        return ""


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    enabled: bool = False
    values: List[float] = field(default_factory=list)
    
    @property
    def num_experiments(self) -> int:
        """Number of experiments this sweep will create."""
        return len(self.values) if self.enabled else 1


@dataclass
class EpistemicConfig:
    """Epistemic uncertainty configuration."""
    under_supported_classes: str = "random:2"
    samples_per_class: int = 50
    regular_samples_per_class: int = 300
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class AleatoricConfig:
    """Aleatoric uncertainty configuration."""
    noise_source: str = "cifar10n"  # "cifar10n" or "random"
    noise_level: float = 40.0  # percentage
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class UnifiedBuilderConfig:
    """Complete configuration from unified builder."""
    epistemic: EpistemicConfig
    aleatoric: AleatoricConfig
    validation: ValidationConfig
    
    # Model config
    dinov2_model: str = "small"
    hidden_dim: int = 256
    use_untrained_resnet: bool = False
    
    # Training config
    epochs: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    train_batch_size: int = 256
    training_dropout: Optional[float] = 0.2  # Regularization during training
    
    # Evaluation/UQ config
    eval_per_group: int = 100
    mc_dropout_enabled: bool = True  # Enable MC dropout for UQ
    mc_dropout_rate: Optional[float] = 0.2  # Dropout rate for MC sampling
    mc_passes: int = 20  # Number of MC forward passes
    
    # Noise type for API
    noise_type: str = "worse_label"
    
    @property
    def experiment_type(self) -> str:
        """Auto-detect experiment type."""
        if self.epistemic.sweep.enabled and self.aleatoric.sweep.enabled:
            return "2D Grid"
        elif self.epistemic.sweep.enabled:
            return "1D Epistemic"
        elif self.aleatoric.sweep.enabled:
            return "1D Aleatoric"
        return "Single"
    
    @property
    def total_experiments(self) -> int:
        """Total number of experiments to create."""
        return (self.epistemic.sweep.num_experiments * 
                self.aleatoric.sweep.num_experiments)


# Made with Bob