"""Backward-compatible re-export; specs live in ``notebook_utils.metrics``."""

from ..metrics import (
    ACCURACY_COLOR,
    ALEATORIC_COLOR,
    ARCHITECTURE_ROW,
    AUROC_ONLY,
    EPISTEMIC_COLOR,
    MetricSpec,
    ORTHOGONAL_AUROC_COLOR,
    PRIMARY_AUROC_COLOR,
    UNCERTAINTY_DECOMPOSITION,
    resolve_columns,
)

__all__ = [
    "ACCURACY_COLOR",
    "ALEATORIC_COLOR",
    "ARCHITECTURE_ROW",
    "AUROC_ONLY",
    "EPISTEMIC_COLOR",
    "MetricSpec",
    "ORTHOGONAL_AUROC_COLOR",
    "PRIMARY_AUROC_COLOR",
    "UNCERTAINTY_DECOMPOSITION",
    "resolve_columns",
]
