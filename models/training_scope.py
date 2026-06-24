"""Resolve and validate trainable scope after model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

from uqlab.evaluation.legacy.triage.dualxda_axioms import infer_classifier_layer_name
from uqlab.models.architecture import CanonicalArchitecture, TrainingScope, normalize_architecture

DataMode = Literal["images", "embeddings"]


@dataclass(frozen=True)
class ResolvedTrainingScope:
    scope: TrainingScope
    architecture: CanonicalArchitecture
    classifier_layer: str
    data_mode: DataMode
    training_mode: str  # legacy: feature_space | end_to_end


def _trainable_prefixes(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, p in model.named_parameters() if p.requires_grad)


def resolve_training_scope(
    model: nn.Module,
    *,
    architecture: str,
    training_scope: TrainingScope,
) -> ResolvedTrainingScope:
    canonical = normalize_architecture(architecture)
    classifier_layer = infer_classifier_layer_name(model)

    if canonical == "dinov2_mlp":
        data_mode: DataMode = "embeddings"
        training_mode = "feature_space"
    else:
        data_mode = "images"
        training_mode = "end_to_end" if training_scope == "full" else "feature_space"

    return ResolvedTrainingScope(
        scope=training_scope,
        architecture=canonical,
        classifier_layer=classifier_layer,
        data_mode=data_mode,
        training_mode=training_mode,
    )


def validate_training_scope(model: nn.Module, resolved: ResolvedTrainingScope) -> None:
    """Ensure frozen/trainable params match declared scope."""
    trainable = _trainable_prefixes(model)
    if not trainable:
        raise ValueError("Model has no trainable parameters")

    layer = resolved.classifier_layer
    head_prefix = layer.split(".")[0]

    if resolved.scope == "full":
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        if frozen and resolved.architecture != "dinov2_mlp":
            raise ValueError(
                f"training_scope=full but parameters are frozen: {frozen[:5]}"
            )
        return

    if resolved.scope in ("head_only", "feature_space"):
        non_head = [n for n in trainable if not n.startswith(head_prefix) and head_prefix not in n]
        if resolved.architecture == "resnet18" and non_head:
            raise ValueError(
                f"training_scope={resolved.scope} but non-head params trainable: {non_head[:5]}"
            )
