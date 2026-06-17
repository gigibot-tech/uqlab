"""
CIFAR-10 Data Loader
Provides clean CIFAR-10 dataset loading utilities.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
