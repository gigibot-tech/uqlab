"""Shim: ``uq_classification.config`` → ``uqlab.shared.config.classification``."""

from uqlab.shared.config.classification import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    parse_args,
    parse_under_supported_classes,
)

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathConfig",
    "TrainingConfig",
    "parse_args",
    "parse_under_supported_classes",
]
