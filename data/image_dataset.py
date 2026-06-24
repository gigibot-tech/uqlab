"""Image subset datasets for end-to-end ResNet / CNN training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from uqlab.data.classification_dataset import dataset_clean_labels
from uqlab.data.preprocessing import get_dataset_image_transform
from uqlab.data.experiment_loader import SplitSpec


class ClassificationImageDataset(Dataset):
    """Subset wrapper returning image tensors with labels/metadata."""

    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

        clean_labels = dataset_clean_labels(base_dataset)
        if base_dataset.noisy_labels is not None and base_dataset.noise_mask is not None:
            noisy_labels = np.asarray(base_dataset.noisy_labels)
            is_noisy = np.asarray(base_dataset.noise_mask, dtype=bool)
        else:
            noisy_labels = clean_labels.copy()
            is_noisy = np.zeros(len(base_dataset), dtype=bool)

        self.targets = torch.as_tensor(noisy_labels[self.indices], dtype=torch.long)
        self.clean_labels = torch.as_tensor(clean_labels[self.indices], dtype=torch.long)
        self.is_noisy = torch.as_tensor(is_noisy[self.indices], dtype=torch.bool)
        self.original_indices = torch.as_tensor(self.indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        dataset_index = int(self.indices[item])
        image = self.base_dataset.get_image(dataset_index)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[item]


CIFAR10NImageDataset = ClassificationImageDataset


def load_image_datasets(
    dataset,
    split_spec: SplitSpec,
    *,
    dataset_name: str = "cifar10",
) -> tuple[ClassificationImageDataset, dict[str, dict[str, torch.Tensor]]]:
    """Build train subset and eval packs for image-mode training."""
    transform = get_dataset_image_transform(dataset_name)
    train_dataset = ClassificationImageDataset(
        dataset, split_spec.train_indices, transform=transform
    )

    def build_eval_pack(indices: np.ndarray) -> dict[str, torch.Tensor]:
        subset = ClassificationImageDataset(dataset, indices, transform=transform)
        images = (
            torch.stack([subset[i][0] for i in range(len(subset))], dim=0)
            if len(subset) > 0
            else torch.empty((0, 3, 32, 32), dtype=torch.float32)
        )
        return {
            "inputs": images,
            "features": images,
            "noisy_labels": subset.targets,
            "clean_labels": subset.clean_labels,
            "is_noisy": subset.is_noisy,
            "original_indices": subset.original_indices,
        }

    eval_packs = {
        "clean": build_eval_pack(split_spec.clean_eval_indices),
        "aleatoric": build_eval_pack(split_spec.aleatoric_eval_indices),
        "epistemic": build_eval_pack(split_spec.epistemic_eval_indices),
        "ood": build_eval_pack(split_spec.ood_eval_indices),
    }
    return train_dataset, eval_packs
