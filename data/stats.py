"""
Dataset Statistics and Analysis - Unified statistics computation.

This module provides functions for analyzing datasets:
- Dataset statistics (mean, std, class distribution)
- Label noise analysis
- Data quality metrics
- Split analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .loaders import CIFAR10NDataset


def compute_dataset_statistics(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    batch_size: int = 128,
) -> Dict[str, any]:
    """
    Compute dataset statistics (mean, std, etc.).
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to use (None = all)
        batch_size: Batch size for computation
        
    Returns:
        Dictionary with statistics
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Compute mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
        
        if num_samples is not None and total_samples >= num_samples:
            break
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'num_samples': total_samples,
    }


def analyze_label_distribution(
    dataset: Dataset,
    num_classes: int = 10,
) -> Dict[str, any]:
    """
    Analyze label distribution in dataset.
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
        
    Returns:
        Dictionary with label distribution statistics
    """
    # Extract labels
    if isinstance(dataset, CIFAR10NDataset):
        if dataset.noisy_labels is not None:
            labels = dataset.noisy_labels
        else:
            labels = np.array(dataset.cifar10.targets)
    else:
        labels = np.array(dataset.targets)
    
    # Compute class counts
    class_counts = np.bincount(labels, minlength=num_classes)
    class_frequencies = class_counts / len(labels)
    
    # Compute balance metrics
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / max_count if min_count > 0 else float('inf')
    
    # Compute entropy (measure of uniformity)
    entropy = -np.sum(class_frequencies * np.log(class_frequencies + 1e-10))
    max_entropy = np.log(num_classes)
    normalized_entropy = entropy / max_entropy
    
    return {
        'class_counts': class_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'total_samples': len(labels),
        'num_classes': num_classes,
        'max_count': int(max_count),
        'min_count': int(min_count),
        'imbalance_ratio': float(imbalance_ratio),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
    }


def compute_noise_statistics(
    dataset: CIFAR10NDataset,
) -> Dict[str, any]:
    """
    Compute noise statistics for CIFAR-10N dataset.
    
    Args:
        dataset: CIFAR10NDataset instance
        
    Returns:
        Dictionary with noise statistics
    """
    if dataset.noise_mask is None or dataset.noisy_labels is None:
        return {
            'has_noise': False,
            'noise_rate': 0.0,
            'num_noisy': 0,
            'num_clean': len(dataset),
        }
    
    clean_labels = np.array(dataset.cifar10.targets)
    noisy_labels = dataset.noisy_labels
    noise_mask = dataset.noise_mask
    
    # Overall noise statistics
    num_noisy = noise_mask.sum()
    num_clean = (~noise_mask).sum()
    noise_rate = noise_mask.mean()
    
    # Per-class noise statistics
    per_class_noise = {}
    for cls in range(10):
        cls_mask = (clean_labels == cls)
        cls_noise_mask = noise_mask[cls_mask]
        
        per_class_noise[cls] = {
            'total': cls_mask.sum(),
            'noisy': cls_noise_mask.sum(),
            'clean': (~cls_noise_mask).sum(),
            'noise_rate': cls_noise_mask.mean() if cls_mask.sum() > 0 else 0.0,
        }
    
    # Transition matrix (clean -> noisy)
    transition_matrix = np.zeros((10, 10))
    for i in range(len(dataset)):
        if noise_mask[i]:
            clean_cls = clean_labels[i]
            noisy_cls = noisy_labels[i]
            transition_matrix[clean_cls, noisy_cls] += 1
    
    # Normalize by row sums
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    transition_matrix = transition_matrix / row_sums
    
    return {
        'has_noise': True,
        'noise_rate': float(noise_rate),
        'num_noisy': int(num_noisy),
        'num_clean': int(num_clean),
        'per_class_noise': per_class_noise,
        'transition_matrix': transition_matrix.tolist(),
    }


def analyze_split_statistics(
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    dataset: Dataset,
    num_classes: int = 10,
) -> Dict[str, any]:
    """
    Analyze statistics of train/eval split.
    
    Args:
        train_indices: Training indices
        eval_indices: Evaluation indices
        dataset: Dataset
        num_classes: Number of classes
        
    Returns:
        Dictionary with split statistics
    """
    # Extract labels
    if isinstance(dataset, CIFAR10NDataset):
        all_labels = np.array(dataset.cifar10.targets)
    else:
        all_labels = np.array(dataset.targets)
    
    train_labels = all_labels[train_indices]
    eval_labels = all_labels[eval_indices]
    
    # Compute class distributions
    train_counts = np.bincount(train_labels, minlength=num_classes)
    eval_counts = np.bincount(eval_labels, minlength=num_classes)
    
    # Compute overlap
    train_set = set(train_indices)
    eval_set = set(eval_indices)
    overlap = len(train_set & eval_set)
    
    return {
        'train_size': len(train_indices),
        'eval_size': len(eval_indices),
        'overlap': overlap,
        'train_class_counts': train_counts.tolist(),
        'eval_class_counts': eval_counts.tolist(),
        'train_class_frequencies': (train_counts / len(train_indices)).tolist(),
        'eval_class_frequencies': (eval_counts / len(eval_indices)).tolist(),
    }


def compute_feature_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
) -> Dict[str, any]:
    """
    Compute statistics of extracted features.
    
    Args:
        features: Feature tensor [N, D]
        labels: Label tensor [N]
        num_classes: Number of classes
        
    Returns:
        Dictionary with feature statistics
    """
    # Overall statistics
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0)
    feature_norm = torch.norm(features, dim=1)
    
    # Per-class statistics
    per_class_stats = {}
    for cls in range(num_classes):
        cls_mask = (labels == cls)
        if cls_mask.sum() == 0:
            continue
        
        cls_features = features[cls_mask]
        cls_mean = cls_features.mean(dim=0)
        cls_std = cls_features.std(dim=0)
        cls_norm = torch.norm(cls_features, dim=1)
        
        per_class_stats[cls] = {
            'count': int(cls_mask.sum()),
            'mean_norm': float(cls_norm.mean()),
            'std_norm': float(cls_norm.std()),
            'mean_feature_std': float(cls_std.mean()),
        }
    
    # Compute inter-class distances (centroids)
    centroids = []
    for cls in range(num_classes):
        cls_mask = (labels == cls)
        if cls_mask.sum() > 0:
            centroids.append(features[cls_mask].mean(dim=0))
        else:
            centroids.append(torch.zeros_like(feature_mean))
    
    centroids = torch.stack(centroids)
    
    # Pairwise centroid distances
    centroid_distances = torch.cdist(centroids, centroids)
    
    return {
        'feature_dim': features.shape[1],
        'num_samples': features.shape[0],
        'mean_norm': float(feature_norm.mean()),
        'std_norm': float(feature_norm.std()),
        'mean_feature_std': float(feature_std.mean()),
        'per_class_stats': per_class_stats,
        'mean_centroid_distance': float(centroid_distances[centroid_distances > 0].mean()),
        'min_centroid_distance': float(centroid_distances[centroid_distances > 0].min()),
        'max_centroid_distance': float(centroid_distances.max()),
    }


def print_dataset_summary(
    dataset: Dataset,
    name: str = "Dataset",
) -> None:
    """
    Print a summary of dataset statistics.
    
    Args:
        dataset: Dataset to analyze
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Total samples: {len(dataset)}")
    
    # Label distribution
    label_stats = analyze_label_distribution(dataset)
    print(f"\nLabel Distribution:")
    print(f"  Classes: {label_stats['num_classes']}")
    print(f"  Balance (entropy): {label_stats['normalized_entropy']:.3f}")
    print(f"  Imbalance ratio: {label_stats['imbalance_ratio']:.2f}")
    
    for cls, count in enumerate(label_stats['class_counts']):
        freq = label_stats['class_frequencies'][cls]
        print(f"    Class {cls}: {count:5d} ({freq*100:5.2f}%)")
    
    # Noise statistics (if applicable)
    if isinstance(dataset, CIFAR10NDataset):
        noise_stats = compute_noise_statistics(dataset)
        if noise_stats['has_noise']:
            print(f"\nNoise Statistics:")
            print(f"  Noise rate: {noise_stats['noise_rate']*100:.2f}%")
            print(f"  Noisy samples: {noise_stats['num_noisy']}")
            print(f"  Clean samples: {noise_stats['num_clean']}")
            
            print(f"\n  Per-class noise rates:")
            for cls, stats in noise_stats['per_class_noise'].items():
                print(f"    Class {cls}: {stats['noise_rate']*100:5.2f}% "
                      f"({stats['noisy']}/{stats['total']})")
    
    print(f"{'='*60}\n")


def print_split_summary(
    train_indices: np.ndarray,
    clean_eval_indices: np.ndarray,
    aleatoric_eval_indices: np.ndarray,
    epistemic_eval_indices: np.ndarray,
    dataset: Dataset,
) -> None:
    """
    Print a summary of data split statistics.
    
    Args:
        train_indices: Training indices
        clean_eval_indices: Clean evaluation indices
        aleatoric_eval_indices: Aleatoric evaluation indices
        epistemic_eval_indices: Epistemic evaluation indices
        dataset: Dataset
    """
    print(f"\n{'='*60}")
    print(f"Data Split Summary")
    print(f"{'='*60}")
    
    print(f"Training set: {len(train_indices)} samples")
    print(f"Clean eval set: {len(clean_eval_indices)} samples")
    print(f"Aleatoric eval set: {len(aleatoric_eval_indices)} samples")
    print(f"Epistemic eval set: {len(epistemic_eval_indices)} samples")
    
    # Analyze each split
    for name, indices in [
        ("Training", train_indices),
        ("Clean Eval", clean_eval_indices),
        ("Aleatoric Eval", aleatoric_eval_indices),
        ("Epistemic Eval", epistemic_eval_indices),
    ]:
        if len(indices) == 0:
            continue
        
        # Get labels
        if isinstance(dataset, CIFAR10NDataset):
            labels = np.array(dataset.cifar10.targets)[indices]
        else:
            labels = np.array(dataset.targets)[indices]
        
        class_counts = np.bincount(labels, minlength=10)
        
        print(f"\n{name} class distribution:")
        for cls, count in enumerate(class_counts):
            if count > 0:
                print(f"  Class {cls}: {count:4d} ({count/len(indices)*100:5.2f}%)")
    
    print(f"{'='*60}\n")

# Made with Bob
