"""Paper benchmark integrations (disentanglement_error bridge)."""

from uqlab.evaluation.benchmarks.disentangling import (
    ExperimentDisentanglingModel,
    UQLabDisentanglingBridge,
    calculate_disentanglement_error,
    collect_cifar10_arrays,
)

__all__ = [
    "ExperimentDisentanglingModel",
    "UQLabDisentanglingBridge",
    "calculate_disentanglement_error",
    "collect_cifar10_arrays",
]
