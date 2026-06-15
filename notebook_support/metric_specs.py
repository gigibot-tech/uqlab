"""Shim: ``uqlab.notebook_support.metric_specs`` → ``shared.notebook_utils.metrics``."""

from ..shared.notebook_utils.metrics import (
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
