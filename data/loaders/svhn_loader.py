"""
SVHN Data Loader
Provides SVHN (Street View House Numbers) dataset for OOD detection experiments.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


class SVHNDataset:
    """Wrapper for SVHN dataset with standard transforms."""
    
    def __init__(self, root='./data/svhn', download=True):
        """
        Initialize SVHN dataset.
        
        Args:
            root: Root directory for dataset
            download: Whether to download if not present
        """
        self.root = root
        self.download = download
        
    def get_transform(self):
        """
        Get SVHN transforms.
        Note: SVHN images are 32x32, same as CIFAR-10.
        Using CIFAR-10 normalization for consistency.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])
    
    def get_full_loader(self, batch_size=128, num_workers=4, split='test'):
        """
        Get full SVHN data loader.
        
        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            split: 'train' or 'test'
            
        Returns:
            DataLoader for SVHN data
        """
        transform = self.get_transform()
        svhn_dataset = datasets.SVHN(
            root=self.root,
            split=split,
            download=self.download,
            transform=transform
        )
        
        return DataLoader(
            svhn_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_subset_loader(self, n_samples=1000, batch_size=128, num_workers=4, split='test'):
        """
        Get a subset of SVHN data for faster experiments.
        
        Args:
            n_samples: Number of samples to use
            batch_size: Batch size
            num_workers: Number of data loading workers
            split: 'train' or 'test'
            
        Returns:
            DataLoader for SVHN subset
        """
        transform = self.get_transform()
        svhn_dataset = datasets.SVHN(
            root=self.root,
            split=split,
            download=self.download,
            transform=transform
        )
        
        # Create random subset
        n_samples = min(n_samples, len(svhn_dataset))
        indices = np.random.choice(len(svhn_dataset), n_samples, replace=False)
        subset = Subset(svhn_dataset, indices.tolist())
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


def get_svhn_loader(root='./data/svhn', batch_size=128, download=True, 
                    num_workers=4, n_samples=None, split='test'):
    """
    Convenience function to get SVHN loader.
    
    Args:
        root: Root directory for dataset
        batch_size: Batch size
        download: Whether to download if not present
        num_workers: Number of data loading workers
        n_samples: If specified, return subset with this many samples
        split: 'train' or 'test'
        
    Returns:
        DataLoader for SVHN data
    """
    dataset = SVHNDataset(root=root, download=download)
    
    if n_samples is not None:
        return dataset.get_subset_loader(
            n_samples=n_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            split=split
        )
    else:
        return dataset.get_full_loader(
            batch_size=batch_size,
            num_workers=num_workers,
            split=split
        )

# Made with Bob
