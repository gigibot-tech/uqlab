"""Fashion-MNIST loader with synthetic label noise."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

FASHION_MNIST_CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST with optional uniform label noise injection."""

    def __init__(
        self,
        root: str = "./data/fashion_mnist",
        train: bool = True,
        transform=None,
        download: bool = True,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.fashion_mnist = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=None,
        )
        clean = np.array(self.fashion_mnist.targets, dtype=np.int64)
        self.noisy_labels: Optional[np.ndarray] = clean.copy()
        self.noise_mask: Optional[np.ndarray] = np.zeros(len(clean), dtype=bool)
        self.noise_rate = 0.0

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def clean_labels(self) -> np.ndarray:
        return np.array(self.fashion_mnist.targets, dtype=np.int64)

    @property
    def targets(self) -> np.ndarray:
        if self.noisy_labels is not None:
            return self.noisy_labels
        return self.clean_labels

    @property
    def class_names(self) -> list[str]:
        return list(FASHION_MNIST_CLASS_NAMES)

    def get_image(self, index: int) -> Image.Image:
        img, _ = self.fashion_mnist[index]
        return img

    def inject_custom_noise(self, noise_percentage: float, seed: int = 42) -> None:
        if noise_percentage <= 0:
            clean = self.clean_labels
            self.noisy_labels = clean.copy()
            self.noise_mask = np.zeros(len(clean), dtype=bool)
            self.noise_rate = 0.0
            return

        rng = np.random.default_rng(seed)
        clean = self.clean_labels
        n = len(clean)
        n_flip = int(round(n * float(noise_percentage) / 100.0))
        noisy = clean.copy()
        if n_flip > 0:
            flip_idx = rng.choice(n, size=min(n_flip, n), replace=False)
            for idx in flip_idx:
                original = int(clean[idx])
                wrong = [c for c in range(self.num_classes) if c != original]
                noisy[idx] = int(rng.choice(wrong))
        self.noisy_labels = noisy
        self.noise_mask = noisy != clean
        self.noise_rate = float(self.noise_mask.mean())

    def __len__(self) -> int:
        return len(self.fashion_mnist)

    def __getitem__(self, index: int):
        image = self.get_image(index)
        label = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
