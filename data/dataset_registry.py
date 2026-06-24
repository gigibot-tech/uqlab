"""Dataset registry, factory, and local stats for classification experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from uqlab.data.classification_dataset import ClassificationDatasetProtocol, dataset_clean_labels

CIFAR10N_NOISE_OPTIONS = (
    "clean_label",
    "worse_label",
    "aggre_label",
    "random_label1",
    "random_label2",
    "random_label3",
)

SYNTHETIC_ONLY_NOISE = ("clean_label",)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label: str
    short: str
    description: str
    num_classes: int
    default_root: str
    total_samples: int
    supports_human_noise: bool
    supports_synthetic_noise: bool
    image_shape: tuple[int, ...]
    noise_options: tuple[str, ...]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        label="CIFAR-10 (original)",
        short="CIFAR-10",
        description=(
            "Standard torchvision CIFAR-10 — **clean, verified labels**. "
            "Paper sweeps (Fig. 3/4) inject synthetic label noise at train time."
        ),
        num_classes=10,
        default_root="./data/cifar10",
        total_samples=50_000,
        supports_human_noise=False,
        supports_synthetic_noise=True,
        image_shape=(3, 32, 32),
        noise_options=SYNTHETIC_ONLY_NOISE,
    ),
    "cifar10n": DatasetSpec(
        name="cifar10n",
        label="CIFAR-10N (human noisy labels)",
        short="CIFAR-10N",
        description=(
            "Same 50k images as CIFAR-10, but labels come from **human annotators** "
            "(Wei et al.). Choose a noise split below — not the same as synthetic sweep noise."
        ),
        num_classes=10,
        default_root="./data/cifar10n",
        total_samples=50_000,
        supports_human_noise=True,
        supports_synthetic_noise=True,
        image_shape=(3, 32, 32),
        noise_options=CIFAR10N_NOISE_OPTIONS,
    ),
    "mnist": DatasetSpec(
        name="mnist",
        label="MNIST (grayscale digits)",
        short="MNIST",
        description=(
            "Classic 28×28 grayscale digits (60k train). "
            "Fig. 4 label-noise sweeps use **synthetic** uniform noise only."
        ),
        num_classes=10,
        default_root="./data/mnist",
        total_samples=60_000,
        supports_human_noise=False,
        supports_synthetic_noise=True,
        image_shape=(1, 28, 28),
        noise_options=SYNTHETIC_ONLY_NOISE,
    ),
    "fashion_mnist": DatasetSpec(
        name="fashion_mnist",
        label="Fashion-MNIST (grayscale apparel)",
        short="Fashion-MNIST",
        description=(
            "28×28 grayscale clothing images (60k train), 10 classes. "
            "Drop-in MNIST replacement with **synthetic** uniform label noise."
        ),
        num_classes=10,
        default_root="./data/fashion_mnist",
        total_samples=60_000,
        supports_human_noise=False,
        supports_synthetic_noise=True,
        image_shape=(1, 28, 28),
        noise_options=SYNTHETIC_ONLY_NOISE,
    ),
}


def list_dataset_names() -> List[str]:
    return list(DATASET_SPECS.keys())


def get_dataset_spec(name: str) -> DatasetSpec:
    key = (name or "cifar10").lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {name!r}. Available: {list_dataset_names()}")
    return DATASET_SPECS[key]


def resolve_data_root(
    dataset_name: str,
    *,
    paths: Optional[Dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Path:
    spec = get_dataset_spec(dataset_name)
    paths = paths or {}
    raw = paths.get("data_root") or paths.get("cifar10n_root") or spec.default_root
    root = Path(raw)
    if not root.is_absolute() and project_root is not None:
        root = project_root / root

    if spec.name in ("cifar10", "cifar10n") and project_root is not None:
        if not (root / "cifar-10-batches-py").exists():
            for alt in (
                project_root / "data" / "cifar10n",
                project_root / "data" / "cifar10",
            ):
                if (alt / "cifar-10-batches-py").exists():
                    return alt
    return root


def load_classification_dataset(
    name: str,
    *,
    root: Path | str,
    noise_type: str = "clean_label",
    aleatoric_noise_percentage: Optional[float] = None,
    train: bool = True,
    download: bool = True,
    transform=None,
) -> ClassificationDatasetProtocol:
    """Load a registered dataset with optional synthetic noise."""
    spec = get_dataset_spec(name)
    root_str = str(root)
    alea = float(aleatoric_noise_percentage) if aleatoric_noise_percentage is not None else None

    if spec.name == "cifar10":
        from uqlab.data.loaders.cifar10_loader import CIFAR10ClassificationDataset

        dataset = CIFAR10ClassificationDataset(
            root=root_str,
            train=train,
            transform=transform,
            download=download,
        )
        if alea is not None and alea > 0.0:
            dataset.inject_custom_noise(noise_percentage=alea, seed=42)
        return dataset

    if spec.name == "cifar10n":
        from uqlab.data.loaders.cifar10n_loader import (
            CIFAR10NDataset,
            apply_clean_training_labels,
            is_clean_training_noise_type,
            normalize_noise_type,
        )

        if alea is not None and alea == 0.0:
            dataset = CIFAR10NDataset(
                root=root_str,
                noise_type="clean_label",
                train=train,
                transform=transform,
                download=download,
            )
            apply_clean_training_labels(dataset)
        elif alea is not None and alea > 0.0:
            dataset = CIFAR10NDataset(
                root=root_str,
                noise_type="clean_label",
                train=train,
                transform=transform,
                download=download,
            )
            apply_clean_training_labels(dataset)
            dataset.inject_custom_noise(noise_percentage=alea, seed=42)
        elif is_clean_training_noise_type(noise_type):
            dataset = CIFAR10NDataset(
                root=root_str,
                noise_type=normalize_noise_type(noise_type),
                train=train,
                transform=transform,
                download=download,
            )
            apply_clean_training_labels(dataset)
        else:
            dataset = CIFAR10NDataset(
                root=root_str,
                noise_type=normalize_noise_type(noise_type),
                train=train,
                transform=transform,
                download=download,
            )
        return dataset

    if spec.name == "mnist":
        from uqlab.data.loaders.mnist_loader import MNISTDataset

        dataset = MNISTDataset(
            root=root_str,
            train=train,
            transform=transform,
            download=download,
        )
        if alea is not None and alea > 0.0:
            dataset.inject_custom_noise(noise_percentage=alea, seed=42)
        return dataset

    if spec.name == "fashion_mnist":
        from uqlab.data.loaders.fashion_mnist_loader import FashionMNISTDataset

        dataset = FashionMNISTDataset(
            root=root_str,
            train=train,
            transform=transform,
            download=download,
        )
        if alea is not None and alea > 0.0:
            dataset.inject_custom_noise(noise_percentage=alea, seed=42)
        return dataset

    raise ValueError(f"No loader registered for dataset {name!r}")


def compute_dataset_stats(
    dataset_name: str,
    noise_type: str = "clean_label",
    *,
    root: Optional[Path | str] = None,
    download: bool = False,
) -> Dict[str, Any]:
    """Compute stats locally (no backend required)."""
    spec = get_dataset_spec(dataset_name)
    root_path = Path(root) if root is not None else Path(spec.default_root)
    dataset = load_classification_dataset(
        dataset_name,
        root=root_path,
        noise_type=noise_type,
        train=True,
        download=download,
    )
    clean = dataset_clean_labels(dataset)
    total = len(dataset)
    if dataset.noise_mask is not None:
        noisy_samples = int(np.sum(dataset.noise_mask))
        noise_rate = float(getattr(dataset, "noise_rate", noisy_samples / max(total, 1)))
    else:
        noisy_samples = 0
        noise_rate = 0.0

    class_counts = {int(i): int(np.sum(clean == i)) for i in range(spec.num_classes)}
    noise_per_class: Dict[int, Dict[str, Any]] = {}
    for i in range(spec.num_classes):
        class_mask = clean == i
        if dataset.noise_mask is not None:
            class_noisy = int(np.sum(dataset.noise_mask[class_mask]))
        else:
            class_noisy = 0
        total_in_class = int(np.sum(class_mask))
        noise_per_class[i] = {
            "total": total_in_class,
            "noisy": class_noisy,
            "rate": float(class_noisy / total_in_class) if total_in_class > 0 else 0.0,
        }

    class_names = getattr(dataset, "class_names", None)
    if class_names is None and hasattr(dataset, "cifar10"):
        class_names = list(dataset.cifar10.classes)
    elif class_names is None:
        class_names = [str(i) for i in range(spec.num_classes)]
    else:
        class_names = list(class_names)

    return {
        "dataset_name": dataset_name,
        "total_samples": total,
        "num_classes": spec.num_classes,
        "noisy_samples": noisy_samples,
        "clean_samples": total - noisy_samples,
        "noise_rate": noise_rate,
        "noise_type": noise_type,
        "resolved_noise_type": getattr(dataset, "noise_type", noise_type),
        "class_distribution": class_counts,
        "noise_per_class": noise_per_class,
        "class_names": class_names,
        "source": "local",
    }
