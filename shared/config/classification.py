"""Configuration management for uncertainty classification experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union, Dict, Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

from uqlab.shared.config.signals import DEFAULT_SIGNALS, normalize_evaluation_signals


def parse_under_supported_classes(
    value: Union[str, Sequence[int], None],
    *,
    seed: int = 42,
) -> List[int]:
    """
    Parse under-supported class spec from UI/API into concrete class IDs (0–9).

    Supports:
    - ``random:N`` — pick N distinct classes deterministically from ``seed``
    - ``3,5`` — explicit comma-separated IDs
    - ``[3, 5]`` — already-resolved list
    """
    if value is None:
        return [3, 5]
    if isinstance(value, (list, tuple)):
        return [int(c) for c in value]

    spec = str(value).strip()
    if not spec:
        return [3, 5]

    if spec.startswith("random:"):
        try:
            num_classes = int(spec.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Invalid under_supported_classes '{spec}'. Expected 'random:N' (N=1..9)."
            ) from exc
        if not 0 < num_classes < 10:
            raise ValueError(
                f"Random class count must be between 1 and 9, got {num_classes}"
            )
        import random

        rng = random.Random(seed)
        chosen = list(range(10))
        rng.shuffle(chosen)
        return sorted(chosen[:num_classes])

    try:
        class_ids = [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid under_supported_classes '{spec}'. "
            "Use comma-separated integers (0-9) or 'random:N'."
        ) from exc

    if not class_ids:
        return [3, 5]
    if not all(0 <= cid <= 9 for cid in class_ids):
        raise ValueError(f"Class IDs must be 0-9, got {class_ids}")
    if len(class_ids) != len(set(class_ids)):
        raise ValueError(f"Duplicate class IDs in under_supported_classes: {class_ids}")
    return class_ids


@dataclass
class PerClassConfig:
    """Per-class configuration for training samples and noise.
    
    This allows explicit control over each class's training data and noise level,
    replacing the legacy global under_supported/regular split.
    
    Attributes:
        train_samples: Number of training samples for this class
        label_noise_pct: Label noise percentage (0-100) for this class
        sweep_epistemic: If True, this class participates in epistemic sweeps
        sweep_aleatoric: If True, this class participates in aleatoric sweeps
    """
    train_samples: int = 300
    label_noise_pct: float = 0.0  # 0-100
    sweep_epistemic: bool = False
    sweep_aleatoric: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.train_samples < 0:
            raise ValueError(f"train_samples must be >= 0, got {self.train_samples}")
        if not 0 <= self.label_noise_pct <= 100:
            raise ValueError(f"label_noise_pct must be 0-100, got {self.label_noise_pct}")


@dataclass
class DataConfig:
    """Data loading and splitting configuration.
    
    Supports two modes:
    1. Legacy mode (default): Uses under_supported_classes + global train counts
    2. Per-class mode: Uses per_class_config dict for explicit per-class control
    
    When per_class_config is provided, it takes precedence over legacy fields.
    """
    dataset_name: str = "cifar10"
    noise_type: str = "worse_label"
    
    # Partition mode: "legacy" (under_supported split) or "four_region"
    partition_mode: str = "legacy"
    # Four-region spec (region -> {classes, train_fraction, label_flip_pct})
    class_regions: Optional[Dict[str, Any]] = None
    
    # Legacy mode fields (backward compatible)
    under_supported_classes: Optional[List[int]] = None
    under_train_per_class: int = 10
    regular_train_per_class: int = 500
    aleatoric_noise_percentage: float = 0.0  # 0-100, custom noise injection
    
    # Per-class mode (new, takes precedence if provided)
    per_class_config: Optional[Dict[int, PerClassConfig]] = None
    
    # Common fields
    eval_per_group: int = 600
    
    def __post_init__(self):
        if self.under_supported_classes is None:
            self.under_supported_classes = [3, 5]
    
    def to_per_class_config(self) -> Dict[int, PerClassConfig]:
        """Convert legacy config to per-class format.
        
        Returns:
            Dict mapping class ID (0-9) to PerClassConfig
        """
        if self.per_class_config is not None:
            return self.per_class_config
        
        # Convert legacy mode to per-class format
        per_class = {}
        under_classes = set(self.under_supported_classes or [])
        
        for class_id in range(10):
            if class_id in under_classes:
                # Under-supported class: sparse samples, no noise
                per_class[class_id] = PerClassConfig(
                    train_samples=self.under_train_per_class,
                    label_noise_pct=0.0,
                    sweep_epistemic=False,
                    sweep_aleatoric=False,
                )
            else:
                # Regular class: full samples, global noise
                per_class[class_id] = PerClassConfig(
                    train_samples=self.regular_train_per_class,
                    label_noise_pct=self.aleatoric_noise_percentage,
                    sweep_epistemic=False,
                    sweep_aleatoric=False,
                )
        
        return per_class
    
    @classmethod
    def from_per_class_config(
        cls,
        per_class_config: Dict[int, PerClassConfig],
        dataset_name: str = "cifar10",
        noise_type: str = "worse_label",
        eval_per_group: int = 600,
    ) -> "DataConfig":
        """Create DataConfig from per-class configuration.
        
        Args:
            per_class_config: Dict mapping class ID to PerClassConfig
            dataset_name: Dataset name
            noise_type: Noise type for CIFAR-10N
            eval_per_group: Evaluation samples per group
            
        Returns:
            DataConfig with per_class_config set
        """
        return cls(
            dataset_name=dataset_name,
            noise_type=noise_type,
            per_class_config=per_class_config,
            eval_per_group=eval_per_group,
            # Legacy fields set to None/defaults (not used when per_class_config is set)
            under_supported_classes=None,
            under_train_per_class=10,
            regular_train_per_class=500,
            aleatoric_noise_percentage=0.0,
        )


class ModelConfig(BaseModel):
    """Model architecture configuration with support for multiple architectures."""
    
    # Architecture selection (canonical or legacy alias)
    architecture: str = "dinov2_mlp"
    training_scope: Literal["full", "head_only", "feature_space"] = "feature_space"
    training_mode: Literal["feature_space", "end_to_end"] = "feature_space"
    
    # DINOv2-specific (only used when architecture="dinov2_mlp")
    dinov2_model: str = "dinov2_vitb14"
    
    # Common parameters
    hidden_dim: int = 256
    dropout: float = 0.2
    use_untrained_resnet: bool = False  # If True, use untrained ResNet-50 instead of DINOv2
    checkpoint_path: Optional[str] = None
    
    # CNN-specific (only used when architecture="cnn_mcdropout")
    num_conv_layers: int = 3
    conv_channels: List[int] = [32, 64, 64]
    
    @model_validator(mode="after")
    def sync_scope_and_mode(self) -> "ModelConfig":
        from uqlab.models.architecture import normalize_architecture, scope_to_training_mode

        canonical = normalize_architecture(self.architecture)
        object.__setattr__(self, "architecture", canonical)
        mode = scope_to_training_mode(canonical, self.training_scope)
        object.__setattr__(self, "training_mode", mode)
        return self

    @field_validator("training_mode")
    @classmethod
    def validate_training_mode(cls, v: str, info) -> str:
        """Validate that training_mode is compatible with architecture."""
        arch = info.data.get("architecture")
        if arch == "dinov2_mlp" and v != "feature_space":
            raise ValueError("dinov2_mlp only supports feature_space mode")
        return v
    
    @field_validator("conv_channels")
    @classmethod
    def validate_conv_channels(cls, v: List[int], info) -> List[int]:
        """Validate that conv_channels length matches num_conv_layers."""
        num_layers = info.data.get("num_conv_layers", 3)
        if len(v) != num_layers:
            raise ValueError(
                f"conv_channels length ({len(v)}) must match num_conv_layers ({num_layers})"
            )
        return v
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 12
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_batch_size: int = 256
    feature_batch_size: int = 64


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    mc_passes: int = 20
    top_k: int = 10
    attribution_method: str = "dualxda"
    attribution_backends: Optional[list[str]] = None
    signals: Optional[dict] = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = {k: list(v) for k, v in DEFAULT_SIGNALS.items()}
        else:
            self.signals = normalize_evaluation_signals(self.signals)


@dataclass
class PathConfig:
    """File paths."""
    data_root: Path = Path("./data/cifar10n")
    cifar10n_root: Path = Path("./data/cifar10n")
    results_base_dir: Path = Path("./results")
    feature_cache_dir: Path = Path("./cache/fast_uncertainty_classification/features")

    def __post_init__(self):
        if self.data_root == Path("./data/cifar10n") and self.cifar10n_root != Path("./data/cifar10n"):
            self.data_root = self.cifar10n_root
        elif self.cifar10n_root == Path("./data/cifar10n") and self.data_root != Path("./data/cifar10n"):
            self.cifar10n_root = self.data_root


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    seed: int = 42
    device: str = "auto"
    data: Optional[DataConfig] = None
    model: Optional[ModelConfig] = None
    training: Optional[TrainingConfig] = None
    evaluation: Optional[EvaluationConfig] = None
    paths: Optional[PathConfig] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.paths is None:
            self.paths = PathConfig()
    
    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        seed = int(config_dict.get("seed", 42))

        # Parse data config
        data_dict = config_dict.get("data", {})
        
        # Normalize noise type
        noise_type = data_dict.get("noise_type", "worse_label")
        try:
            from uqlab.data.loaders.cifar10n_loader import normalize_noise_type
            noise_type = normalize_noise_type(noise_type)
        except ImportError:
            # Fallback normalization if import fails
            if noise_type in ("none", "clean", "no_noise"):
                noise_type = "clean_label"
            elif noise_type == "worst":
                noise_type = "worse_label"
        
        # Check if per-class config is provided in YAML
        per_class_dict = data_dict.get("per_class_config")
        per_class_config = None
        
        if per_class_dict is not None:
            # Parse per-class config from YAML
            # Expected format:
            # per_class_config:
            #   0: {train_samples: 300, label_noise_pct: 0, sweep_epistemic: false, sweep_aleatoric: false}
            #   1: {train_samples: 300, label_noise_pct: 0, sweep_epistemic: false, sweep_aleatoric: false}
            #   ...
            per_class_config = {}
            for class_id_str, class_cfg in per_class_dict.items():
                class_id = int(class_id_str)
                per_class_config[class_id] = PerClassConfig(
                    train_samples=class_cfg.get("train_samples", 300),
                    label_noise_pct=class_cfg.get("label_noise_pct", 0.0),
                    sweep_epistemic=class_cfg.get("sweep_epistemic", False),
                    sweep_aleatoric=class_cfg.get("sweep_aleatoric", False),
                )
        
        # Parse legacy fields (used if per_class_config not provided)
        under_classes = parse_under_supported_classes(
            data_dict.get("under_supported_classes", "3,5"),
            seed=seed,
        )

        partition_mode = str(data_dict.get("partition_mode", "legacy"))
        class_regions_raw = data_dict.get("class_regions")
        class_regions = None
        if partition_mode == "four_region" and class_regions_raw is not None:
            from uqlab.data.class_regions import normalize_class_regions

            class_regions = normalize_class_regions(class_regions_raw)
            sparse = class_regions.get("sparse", {}).get("classes") or []
            if sparse:
                under_classes = [int(c) for c in sparse]
        
        data_config = DataConfig(
            dataset_name=data_dict.get("dataset_name", "cifar10"),
            noise_type=noise_type,
            partition_mode=partition_mode,
            class_regions=class_regions,
            under_supported_classes=under_classes,
            under_train_per_class=data_dict.get("under_train_per_class", 10),
            regular_train_per_class=data_dict.get("regular_train_per_class", 500),
            eval_per_group=data_dict.get("eval_per_group", 600),
            aleatoric_noise_percentage=data_dict.get("aleatoric_noise_percentage", 0.0),
            per_class_config=per_class_config,
        )
        
        # Parse model config
        model_dict = config_dict.get("model", {})
        
        # Handle conv_channels as a list
        conv_channels = model_dict.get("conv_channels", [32, 64, 64])
        if isinstance(conv_channels, str):
            conv_channels = [int(x.strip()) for x in conv_channels.split(",")]
        
        scope = model_dict.get("training_scope")
        if scope is None:
            tm = model_dict.get("training_mode", "feature_space")
            scope = "full" if tm == "end_to_end" else "head_only"
            arch_raw = model_dict.get("architecture", "dinov2_mlp")
            if arch_raw == "dinov2_mlp":
                scope = "feature_space"

        model_config = ModelConfig(
            architecture=model_dict.get("architecture", "dinov2_mlp"),
            training_scope=scope,
            training_mode=model_dict.get("training_mode", "feature_space"),
            dinov2_model=model_dict.get("dinov2_model", "dinov2_vitb14"),
            hidden_dim=model_dict.get("hidden_dim", 256),
            dropout=model_dict.get("dropout", 0.2),
            use_untrained_resnet=model_dict.get("use_untrained_resnet", False),
            checkpoint_path=model_dict.get("checkpoint_path"),
            num_conv_layers=model_dict.get("num_conv_layers", 3),
            conv_channels=conv_channels,
        )
        
        # Parse training config
        training_dict = config_dict.get("training", {})
        training_config = TrainingConfig(
            epochs=training_dict.get("epochs", 12),
            learning_rate=training_dict.get("learning_rate", 1e-3),
            weight_decay=training_dict.get("weight_decay", 1e-4),
            train_batch_size=training_dict.get("train_batch_size", 256),
            feature_batch_size=training_dict.get("feature_batch_size", 64),
        )
        
        # Parse evaluation config
        eval_dict = config_dict.get("evaluation", {})
        eval_config = EvaluationConfig(
            mc_passes=eval_dict.get("mc_passes", 20),
            top_k=eval_dict.get("top_k", 10),
            attribution_method=eval_dict.get("attribution_method", "dualxda"),
            attribution_backends=eval_dict.get("attribution_backends"),
            signals=eval_dict.get("signals"),
        )
        
        # Parse paths config
        paths_dict = config_dict.get("paths", {})
        data_root = paths_dict.get("data_root") or paths_dict.get("cifar10n_root", "./data/cifar10n")
        paths_config = PathConfig(
            data_root=Path(data_root),
            cifar10n_root=Path(paths_dict.get("cifar10n_root", data_root)),
            results_base_dir=Path(paths_dict.get("results_base_dir", "./results")),
            feature_cache_dir=Path(paths_dict.get("feature_cache_dir", "./cache/fast_uncertainty_classification/features")),
        )
        
        return cls(
            seed=seed,
            device=config_dict.get("device", "auto"),
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=eval_config,
            paths=paths_config,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fast uncertainty classification with DualXDA attribution signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fast_uq_classification.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    return parser.parse_args()

# Made with Bob
