"""Helpers for which eval groups a run should populate."""

from __future__ import annotations


def expects_aleatoric_eval(aleatoric_noise_percentage: float | None) -> bool:
    """True when label noise is injected (Fig. 4 / aleatoric benchmark)."""
    return float(aleatoric_noise_percentage or 0.0) > 0.0


def expects_epistemic_eval(
    under_supported_classes: list[int] | tuple[int, ...],
    *,
    under_train_per_class: int,
    regular_train_per_class: int | None,
) -> bool:
    """True when under-training creates a distinct epistemic eval pool (Fig. 3)."""
    if not under_supported_classes:
        return False
    if regular_train_per_class is None:
        return True
    return int(under_train_per_class) < int(regular_train_per_class)
