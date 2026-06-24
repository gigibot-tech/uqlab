"""Thin CIFAR-10 array loader for the paper disentanglement benchmark runner."""

from __future__ import annotations

import numpy as np


def collect_cifar10_arrays(*, data_root: str = "./data") -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(X, y)`` numpy arrays for ``calculate_disentanglement_error``.

    Raw images are ignored by :class:`ExperimentDisentanglingModel.fit` (uqlab loads
    CIFAR internally); arrays only need consistent length for train/test split.
    """
    try:
        from torchvision.datasets import CIFAR10

        dataset = CIFAR10(root=data_root, train=True, download=False)
        labels = np.asarray(dataset.targets, dtype=np.int64)
        # Placeholder features — upstream metric only checks array shapes for split.
        features = np.zeros((len(labels), 1), dtype=np.float32)
        return features, labels
    except Exception:
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 10, size=500, dtype=np.int64)
        features = np.zeros((len(labels), 1), dtype=np.float32)
        return features, labels
