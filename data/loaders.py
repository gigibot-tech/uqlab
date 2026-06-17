"""
Unified Data Loaders - Consolidates CIFAR-10N and feature extraction.

This module combines functionality from:
- src/data/cifar10n_loader.py (CIFAR-10N dataset with noisy labels)
- src/uqlab/classification/data_loader.py (feature extraction and split management)

Key consolidations:
- Single source of truth for CIFAR-10N loading
- Unified split specification and sampling
- Integrated feature extraction with caching
- Simplified embedding organization
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class SplitSpec:
    """
    Specification of train/eval data splits for uncertainty classification.
    
    Attributes:
        train_indices: Indices for training set
        clean_eval_indices: Indices for clean evaluation samples
        aleatoric_eval_indices: Indices for aleatoric-like samples (noisy labels)
        epistemic_eval_indices: Indices for epistemic-like samples (under-supported classes)
        under_supported_classes: List of intentionally under-supported class IDs
    """
    train_indices: np.ndarray
    clean_eval_indices: np.ndarray
    aleatoric_eval_indices: np.ndarray
    epistemic_eval_indices: np.ndarray
    under_supported_classes: List[int]


class CIFAR10NDataset(Dataset):
    """
    CIFAR-10 with noisy labels for noise filtering experiments.
    
    This dataset wraps CIFAR-10 and provides access to both clean and noisy labels.
    Supports both CIFAR-10N human-annotated noise and custom synthetic noise injection.
    
    Reference: https://github.com/UCSC-REAL/cifar-10-100n
    """
    
    def __init__(
        self,
        root: str = './data/cifar10',
        noise_type: str = 'worse_label',
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ):
        """
        Initialize CIFAR-10N dataset.
        
        Args:
            root: Root directory for CIFAR-10 data
            noise_type: Type of noise ('worse_label', 'aggre_label', 'random_label1', etc.)
            train: Whether to use training set
            transform: Transforms to apply
            download: Whether to download CIFAR-10 if not present
        """
        self.root = root
        self.noise_type = noise_type
        self.train = train
        self.transform = transform
        
        # Load base CIFAR-10 dataset
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll apply transform in __getitem__
        )
        
        # Load noisy labels if available
        self.noisy_labels: Optional[np.ndarray] = None
        self.noise_mask: Optional[np.ndarray] = None
        self.noise_rate: float = 0.0
        
        if train:
            self._load_noisy_labels()
    
    def _load_noisy_labels(self) -> None:
        """Load noisy labels from CIFAR-10N dataset."""
        noise_file = os.path.join(self.root, 'cifar-10-batches-py', 'CIFAR-10_human.pt')
        
        if not os.path.exists(noise_file):
            print(f"Warning: CIFAR-10N noise file not found at {noise_file}")
            print("Using clean labels. To use noisy labels, download CIFAR-10N from:")
            print("https://github.com/UCSC-REAL/cifar-10-100n")
            return
        
        try:
            # Load noise data (disable weights_only for trusted data files)
            noise_data = torch.load(noise_file, map_location='cpu', weights_only=False)
            
            # Get noisy labels for specified noise type
            if self.noise_type in noise_data:
                self.noisy_labels = np.array(noise_data[self.noise_type])
                
                # Calculate noise mask (where noisy != clean)
                clean_labels = np.array(self.cifar10.targets)
                self.noise_mask = (self.noisy_labels != clean_labels)
                self.noise_rate = self.noise_mask.mean()
                
                print(f"Loaded CIFAR-10N with {self.noise_type}: {self.noise_rate*100:.2f}% noise")
            else:
                print(f"Warning: Noise type '{self.noise_type}' not found in CIFAR-10N data")
                print(f"Available types: {list(noise_data.keys())}")
        except Exception as e:
            print(f"Error loading CIFAR-10N: {e}")
    
    def inject_custom_noise(self, noise_percentage: float, seed: int = 42) -> None:
        """
        Inject uniform random label noise.
        
        This performs synthetic label-noise flipping:
        - Start from clean CIFAR-10 labels
        - Randomly choose `noise_percentage` of dataset indices
        - For each chosen index, replace with a randomly selected wrong class
        - Store result in `self.noisy_labels` and mark in `self.noise_mask`
        
        Args:
            noise_percentage: Percentage of labels to corrupt (0-100)
            seed: Random seed for reproducibility
        """
        if noise_percentage <= 0:
            print("Custom noise percentage is 0, skipping noise injection")
            return
        
        if noise_percentage > 100:
            raise ValueError(f"noise_percentage must be between 0 and 100, got {noise_percentage}")
        
        # Get clean labels
        clean_labels = np.array(self.cifar10.targets)
        num_samples = len(clean_labels)
        
        # Number of labels to flip
        num_noisy = int(num_samples * (noise_percentage / 100.0))
        
        # Reproducible random selection
        rng = np.random.default_rng(seed)
        noisy_indices = rng.choice(num_samples, size=num_noisy, replace=False)
        
        # Start from clean labels
        noisy_labels = clean_labels.copy()
        
        # Flip selected labels to random wrong classes
        for idx in noisy_indices:
            original_class = clean_labels[idx]
            wrong_classes = [c for c in range(10) if c != original_class]
            noisy_labels[idx] = rng.choice(wrong_classes)
        
        # Store results
        self.noisy_labels = noisy_labels
        self.noise_mask = (noisy_labels != clean_labels)
        self.noise_rate = self.noise_mask.mean()
        
        print(f"Injected custom noise: {self.noise_rate*100:.2f}% ({num_noisy}/{num_samples} samples)")
        print(f"  - Noise distribution: uniform across all samples")
        print(f"  - Label flipping: random wrong class selection")
        print(f"  - Seed: {seed} (reproducible)")
    
    def __len__(self) -> int:
        return len(self.cifar10)
    
    def __getitem__(self, index: int):
        """
        Get item with both clean and noisy labels.
        
        Returns:
            image: Transformed image
            noisy_label: Noisy label (or clean if noise not available)
            clean_label: Clean label
            is_noisy: Boolean indicating if label is noisy
        """
        image, clean_label = self.cifar10[index]
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Use noisy label if available, otherwise use clean
        if self.noisy_labels is not None and self.noise_mask is not None:
            noisy_label = int(self.noisy_labels[index])
            is_noisy = bool(self.noise_mask[index])
        else:
            noisy_label = clean_label
            is_noisy = False
        
        return image, noisy_label, clean_label, is_noisy
    
    def get_noisy_samples(self) -> np.ndarray:
        """Get indices of noisy samples."""
        if self.noise_mask is not None:
            return np.where(self.noise_mask)[0]
        return np.array([])
    
    def get_clean_samples(self) -> np.ndarray:
        """Get indices of clean samples."""
        if self.noise_mask is not None:
            return np.where(~self.noise_mask)[0]
        return np.arange(len(self))


class CIFAR10NLabelView(Dataset):
    """
    Wrapper to present CIFAR10NDataset as (image, label) for compatibility.
    """
    
    def __init__(self, cifar10n_dataset: CIFAR10NDataset, label_mode: str = 'noisy'):
        """
        Args:
            cifar10n_dataset: CIFAR10NDataset instance
            label_mode: 'noisy' or 'clean' - which label to return
        """
        self.dataset = cifar10n_dataset
        self.label_mode = label_mode
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        image, noisy_label, clean_label, is_noisy = self.dataset[index]
        label = noisy_label if self.label_mode == 'noisy' else clean_label
        return image, label


def get_cifar10n_loaders(
    root: str = './data/cifar10',
    noise_type: str = 'worse_label',
    batch_size: int = 128,
    download: bool = True,
    num_workers: int = 4,
    augment_train: bool = True,
):
    """
    Convenience function to get both train and test loaders.
    
    Args:
        root: Root directory for dataset
        noise_type: Type of noise to use
        batch_size: Batch size
        download: Whether to download if not present
        num_workers: Number of data loading workers
        augment_train: Whether to use data augmentation for training
        
    Returns:
        train_loader, train_dataset, test_loader, test_dataset
    """
    from .preprocessing import get_cifar10_transforms
    
    train_transform, test_transform = get_cifar10_transforms(augment=augment_train)
    
    train_dataset = CIFAR10NDataset(
        root=root,
        noise_type=noise_type,
        train=True,
        transform=train_transform,
        download=download
    )
    
    test_dataset = CIFAR10NDataset(
        root=root,
        noise_type=noise_type,
        train=False,
        transform=test_transform,
        download=download
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, train_dataset, test_loader, test_dataset


def sample_indices_for_fast_pilot(
    dataset: Union[datasets.CIFAR10, CIFAR10NDataset],
    *,
    under_supported_classes: Sequence[int],
    under_train_per_class: int,
    regular_train_per_class: int,
    eval_per_group: int,
    seed: int,
    aleatoric_noise_percentage: float = 0.0,
) -> SplitSpec:
    """
    Sample train/eval indices with controlled class support for uncertainty experiments.
    
    Creates three evaluation groups:
    - Clean: Clean samples from well-supported classes
    - Aleatoric-like: Noisy samples (noisy_label != clean_label) from well-supported classes
    - Epistemic-like: Clean samples from intentionally under-supported classes
    
    Args:
        dataset: CIFAR-10 or CIFAR-10N dataset
        under_supported_classes: Classes to intentionally under-support
        under_train_per_class: Number of training samples per under-supported class
        regular_train_per_class: Number of training samples per regular class
        eval_per_group: Number of evaluation samples per group
        seed: Random seed for reproducibility
        aleatoric_noise_percentage: Custom noise percentage (0-100). If > 0, injects
            uniform random label noise instead of using CIFAR-10N noise.
        
    Returns:
        SplitSpec with train and evaluation indices
    """
    rng = np.random.default_rng(seed)
    
    # Get clean labels - handle both CIFAR10 and CIFAR10NDataset
    if isinstance(dataset, CIFAR10NDataset):
        clean_labels = np.asarray(dataset.cifar10.targets)
    else:
        clean_labels = np.asarray(dataset.targets)
    
    # Get noise mask - handle both CIFAR10 (no noise) and CIFAR10NDataset
    if isinstance(dataset, CIFAR10NDataset):
        noise_mask = (
            np.asarray(dataset.noise_mask)
            if dataset.noise_mask is not None
            else np.zeros(len(dataset), dtype=bool)
        )
    else:
        # Raw CIFAR10 has no noise
        noise_mask = np.zeros(len(dataset), dtype=bool)
    
    under_supported_classes = [int(c) for c in under_supported_classes]
    train_indices: List[int] = []
    
    # Sample training data with controlled class support
    for cls in range(10):
        cls_all = np.where(clean_labels == cls)[0]
        rng.shuffle(cls_all)
        
        if cls in under_supported_classes:
            # Under-supported classes use only clean samples
            cls_clean = cls_all[~noise_mask[cls_all]]
            selected = cls_clean[:under_train_per_class]
        else:
            # Regular classes use the normal training budget
            if regular_train_per_class is None:
                selected = np.array([], dtype=np.int64)
            else:
                selected = cls_all[:regular_train_per_class]
        
        train_indices.extend(selected.tolist())
    
    train_indices = np.array(sorted(set(train_indices)), dtype=np.int64)
    train_mask = np.zeros(len(dataset), dtype=bool)
    train_mask[train_indices] = True
    
    # Define evaluation pools
    under_mask = np.isin(clean_labels, np.asarray(under_supported_classes))
    non_under_mask = ~under_mask
    clean_mask = ~noise_mask
    
    clean_eval_pool = np.where(non_under_mask & clean_mask & ~train_mask)[0]
    aleatoric_eval_pool = np.where(non_under_mask & noise_mask & ~train_mask)[0]
    epistemic_eval_pool = np.where(under_mask & clean_mask & ~train_mask)[0]
    
    # Sample evaluation sets
    rng.shuffle(clean_eval_pool)
    rng.shuffle(aleatoric_eval_pool)
    rng.shuffle(epistemic_eval_pool)
    
    clean_eval_indices = clean_eval_pool[:min(eval_per_group, len(clean_eval_pool))]
    aleatoric_eval_indices = aleatoric_eval_pool[:min(eval_per_group, len(aleatoric_eval_pool))]
    epistemic_eval_indices = epistemic_eval_pool[:min(eval_per_group, len(epistemic_eval_pool))]
    
    # Log warnings if we got fewer samples than requested
    import logging
    logger = logging.getLogger(__name__)
    
    if len(clean_eval_indices) < eval_per_group:
        logger.warning(
            f"⚠️  Clean eval pool: requested {eval_per_group}, got {len(clean_eval_indices)} "
            f"(pool size: {len(clean_eval_pool)})"
        )
    if len(aleatoric_eval_indices) < eval_per_group:
        logger.warning(
            f"⚠️  Aleatoric eval pool: requested {eval_per_group}, got {len(aleatoric_eval_indices)} "
            f"(pool size: {len(aleatoric_eval_pool)})"
        )
    if len(epistemic_eval_indices) < eval_per_group:
        logger.warning(
            f"⚠️  Epistemic eval pool: requested {eval_per_group}, got {len(epistemic_eval_indices)} "
            f"(pool size: {len(epistemic_eval_pool)})"
        )
    
    return SplitSpec(
        train_indices=train_indices,
        clean_eval_indices=clean_eval_indices,
        aleatoric_eval_indices=aleatoric_eval_indices,
        epistemic_eval_indices=epistemic_eval_indices,
        under_supported_classes=under_supported_classes,
    )


@torch.no_grad()
def extract_features_for_indices(
    dataset: CIFAR10NDataset,
    indices: Sequence[int],
    *,
    feature_extractor: nn.Module,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Extract features for specified dataset indices using a feature extractor.
    
    Args:
        dataset: CIFAR-10N dataset
        indices: Indices to extract features for
        feature_extractor: Feature extraction model (DINOv2, ResNet, etc.)
        batch_size: Batch size for feature extraction
        device: Device to run on
        
    Returns:
        Dictionary containing:
            - features: Extracted features [N, feature_dim]
            - noisy_labels: Training labels [N]
            - clean_labels: Ground truth labels [N]
            - is_noisy: Boolean mask [N]
            - original_indices: Original dataset indices [N]
    """
    subset = Subset(dataset, list(indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    all_features: List[torch.Tensor] = []
    all_noisy_labels: List[torch.Tensor] = []
    all_clean_labels: List[torch.Tensor] = []
    all_is_noisy: List[torch.Tensor] = []
    
    for batch in loader:
        # Handle both CIFAR10 (2 values) and CIFAR10NDataset (4 values)
        if len(batch) == 2:
            images, labels = batch
            noisy_labels = labels
            clean_labels = labels
            is_noisy = torch.zeros(len(labels), dtype=torch.bool)
        else:
            images, noisy_labels, clean_labels, is_noisy = batch
        
        # Extract features
        images = images.to(device)
        if hasattr(feature_extractor, 'extract_features'):
            features = feature_extractor.extract_features(images)
        else:
            features = feature_extractor(images)
        
        all_features.append(features.cpu())
        all_noisy_labels.append(noisy_labels.cpu())
        all_clean_labels.append(clean_labels.cpu())
        all_is_noisy.append(is_noisy.cpu())
    
    return {
        "features": torch.cat(all_features, dim=0),
        "noisy_labels": torch.cat(all_noisy_labels, dim=0),
        "clean_labels": torch.cat(all_clean_labels, dim=0),
        "is_noisy": torch.cat(all_is_noisy, dim=0).bool(),
        "original_indices": torch.as_tensor(indices, dtype=torch.long),
    }


def build_feature_cache_path(
    cache_dir: Path,
    indices: Sequence[int],
    *,
    noise_type: str,
    model_name: str,
) -> Path:
    """
    Build a stable feature-cache path from the selected data indices.
    
    The hash keeps the filename short while making the cache reusable across
    result directories whenever the same split and backbone are used again.
    """
    index_bytes = np.asarray(indices, dtype=np.int64).tobytes()
    index_hash = hashlib.sha1(index_bytes).hexdigest()[:12]
    
    return cache_dir / f"features_{noise_type}_{model_name}_n{len(indices)}_{index_hash}.pt"


def maybe_load_or_compute_feature_cache(
    dataset: CIFAR10NDataset,
    indices: Sequence[int],
    *,
    cache_file: Path,
    feature_extractor: nn.Module,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Load features from cache or compute and cache them.
    
    Args:
        dataset: CIFAR-10N dataset
        indices: Indices to extract features for
        cache_file: Path to cache file
        feature_extractor: Feature extraction model
        batch_size: Batch size for feature extraction
        device: Device to run on
        
    Returns:
        Dictionary with features and labels
    """
    if cache_file.exists():
        print(f"Loading cached features from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)
    
    print(f"Computing features (will cache to {cache_file})")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    payload = extract_features_for_indices(
        dataset,
        indices,
        feature_extractor=feature_extractor,
        batch_size=batch_size,
        device=device,
    )
    
    torch.save(payload, cache_file)
    return payload


class EmbeddingOrganizer:
    """
    Organizes feature extraction and split management for uncertainty classification.
    
    This class encapsulates the complex logic of:
    1. Loading/computing embeddings for all samples (train + eval)
    2. Building an index mapping from dataset indices to embedding positions
    3. Extracting embeddings for specific splits
    
    Example Usage:
    --------------
    ```python
    organizer = EmbeddingOrganizer(
        dataset=cifar10n_dataset,
        split_spec=split_spec,
        feature_cache_dir=Path("./cache"),
        noise_type="worse_label",
        model_name="dinov2_large",
        feature_extractor=model,
        batch_size=256,
        device=torch.device("cuda"),
    )
    
    # Load embeddings once
    organizer.load_or_compute_features()
    
    # Extract organized packs
    train_pack = organizer.get_train_pack()
    clean_eval_pack = organizer.get_clean_eval_pack()
    ```
    """
    
    def __init__(
        self,
        dataset: CIFAR10NDataset,
        split_spec: SplitSpec,
        feature_cache_dir: Path,
        noise_type: str,
        model_name: str,
        feature_extractor: nn.Module,
        batch_size: int,
        device: torch.device,
    ):
        """
        Initialize the embedding organizer.
        
        Args:
            dataset: CIFAR-10N dataset
            split_spec: Train/eval split specification
            feature_cache_dir: Directory for embedding caching
            noise_type: CIFAR-10N noise type
            model_name: Feature extractor model name
            feature_extractor: Feature extraction model
            batch_size: Batch size for embedding extraction
            device: Device to run on
        """
        self.dataset = dataset
        self.split_spec = split_spec
        self.feature_cache_dir = feature_cache_dir
        self.noise_type = noise_type
        self.model_name = model_name
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.device = device
        
        # Will be populated by load_or_compute_features()
        self._payload: Optional[Dict[str, torch.Tensor]] = None
        self._index_to_pos: Optional[Dict[int, int]] = None
    
    def load_or_compute_features(self) -> None:
        """
        Load embeddings from cache or compute them.
        
        This method:
        1. Concatenates all indices (train + all eval groups)
        2. Builds cache path based on indices hash
        3. Loads from cache or computes embeddings
        4. Creates index-to-position mapping for fast lookup
        """
        # Step 1: Combine all indices we need features for
        union_indices = np.concatenate([
            self.split_spec.train_indices,
            self.split_spec.clean_eval_indices,
            self.split_spec.aleatoric_eval_indices,
            self.split_spec.epistemic_eval_indices,
        ])
        
        # Step 2: Build cache path
        cache_file = build_feature_cache_path(
            self.feature_cache_dir,
            union_indices.tolist(),
            noise_type=self.noise_type,
            model_name=self.model_name,
        )
        
        # Step 3: Load or compute features
        self._payload = maybe_load_or_compute_feature_cache(
            self.dataset,
            union_indices.tolist(),
            cache_file=cache_file,
            feature_extractor=self.feature_extractor,
            batch_size=self.batch_size,
            device=self.device,
        )
        
        # Step 4: Build index mapping for fast lookup
        original_indices = self._payload["original_indices"].numpy()
        self._index_to_pos = {
            int(idx): pos
            for pos, idx in enumerate(original_indices.tolist())
        }
    
    def _gather(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for specific indices from the loaded payload.
        
        Args:
            indices: Dataset indices to extract
            
        Returns:
            Dictionary with embeddings, labels, and metadata for the indices
            
        Raises:
            RuntimeError: If embeddings haven't been loaded yet
        """
        if self._payload is None or self._index_to_pos is None:
            raise RuntimeError(
                "Embeddings not loaded. Call load_or_compute_features() first."
            )
        
        # Map dataset indices to payload positions
        positions = torch.as_tensor(
            [self._index_to_pos[int(idx)] for idx in indices],
            dtype=torch.long
        )
        
        # Extract data at those positions
        return {
            "features": self._payload["features"][positions],
            "noisy_labels": self._payload["noisy_labels"][positions],
            "clean_labels": self._payload["clean_labels"][positions],
            "is_noisy": self._payload["is_noisy"][positions],
            "original_indices": self._payload["original_indices"][positions],
        }
    
    def get_train_pack(self) -> Dict[str, torch.Tensor]:
        """Extract training data pack."""
        return self._gather(self.split_spec.train_indices)
    
    def get_clean_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract clean evaluation data pack."""
        return self._gather(self.split_spec.clean_eval_indices)
    
    def get_aleatoric_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract aleatoric evaluation data pack."""
        return self._gather(self.split_spec.aleatoric_eval_indices)
    
    def get_epistemic_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Extract epistemic evaluation data pack."""
        return self._gather(self.split_spec.epistemic_eval_indices)

# Made with Bob
