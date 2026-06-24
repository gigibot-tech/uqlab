"""
CIFAR-10 Data Loader
Provides clean CIFAR-10 dataset loading utilities and classification protocol.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CIFAR10ClassificationDataset(Dataset):
    """Clean CIFAR-10 with optional synthetic label noise (Fig. 4 sweeps)."""

    def __init__(
        self,
        root: str = "./data/cifar10",
        train: bool = True,
        transform=None,
        download: bool = True,
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.noise_type = "clean_label"
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None,
        )
        clean = np.array(self.cifar10.targets, dtype=np.int64)
        self.noisy_labels: Optional[np.ndarray] = clean.copy()
        self.noise_mask: Optional[np.ndarray] = np.zeros(len(clean), dtype=bool)
        self.noise_rate = 0.0

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def clean_labels(self) -> np.ndarray:
        return np.array(self.cifar10.targets, dtype=np.int64)

    @property
    def targets(self) -> np.ndarray:
        if self.noisy_labels is not None:
            return self.noisy_labels
        return self.clean_labels

    @property
    def class_names(self) -> list[str]:
        return list(self.cifar10.classes)

    def get_image(self, index: int) -> Image.Image:
        img, _ = self.cifar10[index]
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
        return len(self.cifar10)

    def __getitem__(self, index: int):
        image = self.get_image(index)
        label = int(self.targets[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CIFAR10Dataset:
    """Wrapper for CIFAR-10 dataset with standard transforms."""
    
    def __init__(self, root='./data/cifar10', download=True):
        """
        Initialize CIFAR-10 dataset.
        
        Args:
            root: Root directory for dataset
            download: Whether to download if not present
        """
        self.root = root
        self.download = download
        
    def get_train_transform(self, augment=True):
        """Get training transforms."""
        if augment:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
    
    def get_test_transform(self):
        """Get test transforms."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    def get_train_loader(self, batch_size=128, augment=True, num_workers=4, shuffle=True):
        """
        Get training data loader.
        
        Args:
            batch_size: Batch size
            augment: Whether to use data augmentation
            num_workers: Number of data loading workers
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for training data
        """
        transform = self.get_train_transform(augment)
        train_dataset = datasets.CIFAR10(
            root=self.root,
            train=True,
            download=self.download,
            transform=transform
        )
        
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self, batch_size=128, num_workers=4):
        """
        Get test data loader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            
        Returns:
            DataLoader for test data
        """
        transform = self.get_test_transform()
        test_dataset = datasets.CIFAR10(
            root=self.root,
            train=False,
            download=self.download,
            transform=transform
        )
        
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


def get_cifar10_loaders(root='./data/cifar10', batch_size=128, download=True, num_workers=4):
    """
    Convenience function to get both train and test loaders.
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size
        download: Whether to download if not present
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, test_loader
    """
    dataset = CIFAR10Dataset(root=root, download=download)
    train_loader = dataset.get_train_loader(batch_size=batch_size, num_workers=num_workers)
    test_loader = dataset.get_test_loader(batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader

# Made with Bob
