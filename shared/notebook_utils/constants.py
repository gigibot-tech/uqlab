"""Shared sweep-axis constants for notebook / validation plotting."""

from __future__ import annotations

# sweep_type -> preferred x-axis column in unified metrics frames
SWEEP_TO_X: dict[str, str] = {
    "dataset_size": "under_train_per_class",
    "label_noise": "noise_percent",
}
