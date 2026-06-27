"""Canonical model architecture names and legacy aliases."""

from __future__ import annotations

from typing import Literal, Tuple

CanonicalArchitecture = Literal["resnet18", "cnn_small", "dinov2_mlp", "pixel_mlp"]
TrainingScope = Literal["full", "head_only", "feature_space"]

_ALIASES = {
    "resnet18": "resnet18",
    "resnet18_mcdropout": "resnet18",
    "cnn_small": "cnn_small",
    "cnn_mcdropout": "cnn_small",
    "dinov2_mlp": "dinov2_mlp",
    "pixel_mlp": "pixel_mlp",
    "pixel-mlp": "pixel_mlp",
    "mlp": "pixel_mlp",
}


def normalize_architecture(name: str) -> CanonicalArchitecture:
    key = (name or "dinov2_mlp").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unsupported architecture {name!r}. "
            f"Use one of: {sorted(set(_ALIASES.values()))} (aliases: {sorted(_ALIASES)})"
        )
    return _ALIASES[key]  # type: ignore[return-value]


def normalize_dinov2_model(model_name: str) -> str:
    """Map torch.hub / legacy names to ``DINOv2Backbone`` keys (``small``, ``base``, …)."""
    from uqlab.models.backbones.dinov2_backbone import DINOv2Backbone

    return DINOv2Backbone.normalize_model_name(model_name)


def scope_to_training_mode(
    architecture: CanonicalArchitecture,
    training_scope: TrainingScope,
) -> str:
    """Map training_scope to legacy training_mode for existing pipeline code."""
    if architecture == "dinov2_mlp":
        if training_scope != "feature_space":
            raise ValueError("dinov2_mlp only supports training_scope=feature_space")
        return "feature_space"
    if architecture == "cnn_small":
        if training_scope != "full":
            raise ValueError("cnn_small only supports training_scope=full")
        return "end_to_end"
    if architecture == "pixel_mlp":
        if training_scope != "full":
            raise ValueError("pixel_mlp only supports training_scope=full")
        return "end_to_end"
    # resnet18
    if training_scope == "full":
        return "end_to_end"
    if training_scope in ("head_only", "feature_space"):
        return "feature_space"
    raise ValueError(f"Invalid training_scope {training_scope!r}")


def validate_scope_for_architecture(
    architecture: CanonicalArchitecture,
    training_scope: TrainingScope,
) -> None:
    scope_to_training_mode(architecture, training_scope)


def legacy_architecture_name(canonical: CanonicalArchitecture) -> str:
    """Backward-compatible YAML name for older configs."""
    return {
        "resnet18": "resnet18_mcdropout",
        "cnn_small": "cnn_mcdropout",
        "dinov2_mlp": "dinov2_mlp",
        "pixel_mlp": "pixel_mlp",
    }[canonical]
