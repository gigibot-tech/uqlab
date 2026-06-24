"""Protocol for noise-aware classification datasets used by the UQ pilot."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClassificationDatasetProtocol(Protocol):
    """Minimal interface for train/eval split and image-mode training."""

    @property
    def num_classes(self) -> int: ...

    @property
    def clean_labels(self) -> np.ndarray: ...

    @property
    def targets(self) -> list | np.ndarray: ...

    @property
    def noisy_labels(self) -> Optional[np.ndarray]: ...

    @property
    def noise_mask(self) -> Optional[np.ndarray]: ...

    @property
    def class_names(self) -> list[str]: ...

    def __len__(self) -> int: ...

    def get_image(self, index: int) -> Any: ...

    def inject_custom_noise(self, noise_percentage: float, seed: int = 42) -> None: ...


def dataset_clean_labels(dataset: Any) -> np.ndarray:
    """Resolve clean labels from protocol or legacy CIFAR wrappers."""
    if hasattr(dataset, "clean_labels"):
        return np.asarray(dataset.clean_labels)
    if hasattr(dataset, "cifar10"):
        return np.asarray(dataset.cifar10.targets)
    if hasattr(dataset, "mnist"):
        return np.asarray(dataset.mnist.targets)
    if hasattr(dataset, "fashion_mnist"):
        return np.asarray(dataset.fashion_mnist.targets)
    return np.asarray(dataset.targets)


def dataset_num_classes(dataset: Any) -> int:
    if hasattr(dataset, "num_classes"):
        return int(dataset.num_classes)
    return int(len(np.unique(dataset_clean_labels(dataset))))
