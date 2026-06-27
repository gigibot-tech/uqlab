"""Pure scoring math and the ``results.pt`` read contract."""

from uqlab.evaluation.metrics.artifacts import EvalRunArtifacts, uncertainty_vectors_from_results_pt
from uqlab.evaluation.metrics.scoring import (
    auroc_skip_reason,
    binary_auroc,
    binary_auroc_or_none,
    binary_auroc_vs_group,
    predict_eval_groups_single_signal,
    train_signal_classifier,
)

__all__ = [
    "EvalRunArtifacts",
    "auroc_skip_reason",
    "binary_auroc",
    "binary_auroc_or_none",
    "binary_auroc_vs_group",
    "predict_eval_groups_single_signal",
    "train_signal_classifier",
    "uncertainty_vectors_from_results_pt",
]
