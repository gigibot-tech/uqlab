"""Configuration management for uncertainty classification experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, field_validator


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
class DataConfig:
    """Data loading and splitting configuration."""
    noise_type: str = "worse_label"
    under_supported_classes: Optional[List[int]] = None
    under_train_per_class: int = 10
    regular_train_per_class: int = 500
    eval_per_group: int = 600
    aleatoric_noise_percentage: float = 0.0  # 0-100, custom noise injection
    
    def __post_init__(self):
        if self.under_supported_classes is None:
            self.under_supported_classes = [3, 5]


class ModelConfig(BaseModel):
    """Model architecture configuration with support for multiple architectures."""
    
    # Architecture selection
    architecture: Literal["dinov2_mlp", "cnn_mcdropout", "resnet18_mcdropout"] = "dinov2_mlp"
    training_mode: Literal["feature_space", "end_to_end"] = "feature_space"
    
    # DINOv2-specific (only used when architecture="dinov2_mlp")
    dinov2_model: str = "dinov2_vitb14"
    
    # Common parameters
    hidden_dim: int = 256
    dropout: float = 0.2
    use_untrained_resnet: bool = False  # If True, use untrained ResNet-50 instead of DINOv2
    
    # CNN-specific (only used when architecture="cnn_mcdropout")
    num_conv_layers: int = 3
    conv_channels: List[int] = [32, 64, 64]
    
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


@dataclass
class PathConfig:
    """File paths."""
    cifar10n_root: Path = Path("./data/cifar10n")
    results_base_dir: Path = Path("./results")
    feature_cache_dir: Path = Path("./cache/fast_uncertainty_classification/features")


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
        under_classes = parse_under_supported_classes(
            data_dict.get("under_supported_classes", "3,5"),
            seed=seed,
        )
        noise_type = data_dict.get("noise_type", "worse_label")
        try:
            # Import from renamed data loaders module
            import importlib
            loaders_module = importlib.import_module("uqlab.1_data.loaders.cifar10n_loader")
            normalize_noise_type = getattr(loaders_module, "normalize_noise_type")
            noise_type = normalize_noise_type(noise_type)
        except (ImportError, AttributeError):
            # Fallback normalization if import fails
            if noise_type in ("none", "clean", "no_noise"):
                noise_type = "clean_label"
            elif noise_type == "worst":
                noise_type = "worse_label"

        data_config = DataConfig(
            noise_type=noise_type,
            under_supported_classes=under_classes,
            under_train_per_class=data_dict.get("under_train_per_class", 10),
            regular_train_per_class=data_dict.get("regular_train_per_class", 500),
            eval_per_group=data_dict.get("eval_per_group", 600),
            aleatoric_noise_percentage=data_dict.get("aleatoric_noise_percentage", 0.0),
        )
        
        # Parse model config
        model_dict = config_dict.get("model", {})
        
        # Handle conv_channels as a list
        conv_channels = model_dict.get("conv_channels", [32, 64, 64])
        if isinstance(conv_channels, str):
            conv_channels = [int(x.strip()) for x in conv_channels.split(",")]
        
        model_config = ModelConfig(
            architecture=model_dict.get("architecture", "dinov2_mlp"),
            training_mode=model_dict.get("training_mode", "feature_space"),
            dinov2_model=model_dict.get("dinov2_model", "dinov2_vitb14"),
            hidden_dim=model_dict.get("hidden_dim", 256),
            dropout=model_dict.get("dropout", 0.2),
            use_untrained_resnet=model_dict.get("use_untrained_resnet", False),
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
        )
        
        # Parse paths config
        paths_dict = config_dict.get("paths", {})
        paths_config = PathConfig(
            cifar10n_root=Path(paths_dict.get("cifar10n_root", "./data/cifar10n")),
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
