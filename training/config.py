"""
Training Configuration - Training-specific configuration and hyperparameters.

This module provides configuration for the training pipeline:
- Training hyperparameters
- Optimizer settings
- Learning rate schedules
- Regularization options
- Checkpoint settings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from shared.types import PathLike


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    
    # Optimizer type
    optimizer_type: Literal["adam", "adamw", "sgd", "rmsprop"] = "adamw"
    
    # Learning rate
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # SGD-specific
    momentum: float = 0.9
    nesterov: bool = True
    
    # Adam/AdamW-specific
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    
    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    
    # Scheduler type
    scheduler_type: Literal["cosine", "step", "exponential", "plateau", "none"] = "cosine"
    
    # Cosine annealing
    t_max: Optional[int] = None  # If None, uses total epochs
    eta_min: float = 1e-6
    
    # Step scheduler
    step_size: int = 10
    gamma: float = 0.1
    milestones: Optional[List[int]] = None
    
    # Exponential scheduler
    decay_rate: float = 0.95
    
    # Plateau scheduler
    mode: Literal["min", "max"] = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    
    # Warmup
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-6


@dataclass
class RegularizationConfig:
    """Regularization configuration."""
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Mixup
    mixup_alpha: float = 0.0
    mixup_prob: float = 0.5
    
    # CutMix
    cutmix_alpha: float = 0.0
    cutmix_prob: float = 0.5
    
    # Dropout
    dropout: float = 0.2
    droppath: float = 0.0
    
    # Stochastic depth
    stochastic_depth: float = 0.0


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""
    
    # Save settings
    save_dir: Path = field(default_factory=lambda: Path("./results/checkpoints"))
    save_every_n_epochs: int = 5
    save_last: bool = True
    save_best: bool = True
    
    # Keep settings
    keep_last_n: int = 3
    keep_best_n: int = 1
    
    # Best model criteria
    monitor_metric: str = "val_accuracy"
    monitor_mode: Literal["min", "max"] = "max"
    
    # Resume settings
    resume_from: Optional[PathLike] = None
    load_optimizer: bool = True
    load_scheduler: bool = True


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    
    # Enable/disable
    enabled: bool = False
    
    # Monitoring
    monitor_metric: str = "val_loss"
    monitor_mode: Literal["min", "max"] = "min"
    
    # Patience
    patience: int = 10
    min_delta: float = 1e-4
    
    # Restore best
    restore_best_weights: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    
    # Basic settings
    epochs: int = 12
    train_batch_size: int = 256
    eval_batch_size: int = 512
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Mixed precision
    use_amp: bool = False
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    
    # Gradient accumulation
    accumulation_steps: int = 1
    
    # Validation
    val_every_n_epochs: int = 1
    val_before_training: bool = True
    
    # Logging
    log_every_n_steps: int = 10
    log_gradients: bool = False
    log_weights: bool = False
    
    # Sub-configurations
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    
    def __post_init__(self):
        """Post-initialization validation."""
        # Ensure checkpoint directory exists
        self.checkpoint.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set scheduler t_max if not provided
        if self.scheduler.scheduler_type == "cosine" and self.scheduler.t_max is None:
            self.scheduler.t_max = self.epochs
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> TrainingConfig:
        """Create configuration from dictionary."""
        # Extract sub-configs
        optimizer_dict = config_dict.pop("optimizer", {})
        scheduler_dict = config_dict.pop("scheduler", {})
        regularization_dict = config_dict.pop("regularization", {})
        checkpoint_dict = config_dict.pop("checkpoint", {})
        early_stopping_dict = config_dict.pop("early_stopping", {})
        
        return cls(
            **config_dict,
            optimizer=OptimizerConfig(**optimizer_dict),
            scheduler=SchedulerConfig(**scheduler_dict),
            regularization=RegularizationConfig(**regularization_dict),
            checkpoint=CheckpointConfig(**checkpoint_dict),
            early_stopping=EarlyStoppingConfig(**early_stopping_dict),
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "accumulation_steps": self.accumulation_steps,
            "val_every_n_epochs": self.val_every_n_epochs,
            "val_before_training": self.val_before_training,
            "log_every_n_steps": self.log_every_n_steps,
            "log_gradients": self.log_gradients,
            "log_weights": self.log_weights,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "regularization": self.regularization.__dict__,
            "checkpoint": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.checkpoint.__dict__.items()
            },
            "early_stopping": self.early_stopping.__dict__,
        }


# Default configurations for common scenarios
DEFAULT_FAST_CONFIG = TrainingConfig(
    epochs=5,
    train_batch_size=512,
    eval_batch_size=1024,
    optimizer=OptimizerConfig(learning_rate=1e-3),
    scheduler=SchedulerConfig(scheduler_type="none"),
)

DEFAULT_STANDARD_CONFIG = TrainingConfig(
    epochs=12,
    train_batch_size=256,
    eval_batch_size=512,
    optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=1e-4),
    scheduler=SchedulerConfig(scheduler_type="cosine", warmup_epochs=2),
)

DEFAULT_ROBUST_CONFIG = TrainingConfig(
    epochs=20,
    train_batch_size=128,
    eval_batch_size=256,
    optimizer=OptimizerConfig(learning_rate=5e-4, weight_decay=1e-3),
    scheduler=SchedulerConfig(scheduler_type="cosine", warmup_epochs=3),
    regularization=RegularizationConfig(
        label_smoothing=0.1,
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
    ),
    early_stopping=EarlyStoppingConfig(enabled=True, patience=5),
)

# Made with Bob