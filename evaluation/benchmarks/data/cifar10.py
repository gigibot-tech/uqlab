"""
CIFAR-10 dataset loader.
Adapted from uq_disentanglement_comparison package.
"""

import numpy as np
from functools import lru_cache
from typing import Optional

try:
    import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("Warning: Keras not available. CIFAR-10 loader will not work.")

from ..datatypes import Dataset


@lru_cache(maxsize=None)
def get_cifar10_dataset(
    test_mode: bool = False,
    noise_rate: float = 0.0,
    seed: int = 42
) -> Dataset:
    """
    Load CIFAR-10 dataset with optional label noise injection.
    
    Args:
        test_mode: If True, use only 100 test samples for quick testing
        noise_rate: Percentage of labels to corrupt (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Dataset object with train/test splits
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras is required for CIFAR-10 loading. Install with: pip install keras")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Test mode: reduce dataset size
    if test_mode:
        n_test_samples = 100
        x_test = x_test[:n_test_samples]
        y_test = y_test[:n_test_samples]
        
        n_train_samples = 500
        x_train = x_train[:n_train_samples]
        y_train = y_train[:n_train_samples]
    
    # Inject label noise if requested
    noise_mask = None
    clean_labels = None
    if noise_rate > 0:
        np.random.seed(seed)
        n_samples = len(y_train)
        n_noisy = int(n_samples * noise_rate)
        
        # Create noise mask
        noise_mask = np.zeros(n_samples, dtype=bool)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        noise_mask[noisy_indices] = True
        
        # Store clean labels
        clean_labels = y_train.copy()
        
        # Corrupt labels (random shuffle within noisy samples)
        y_train_noisy = y_train.copy()
        np.random.shuffle(y_train_noisy[noise_mask])
        y_train = y_train_noisy
    
    return Dataset(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        is_regression=False,
        noise_mask=noise_mask,
        clean_labels=clean_labels,
        noise_rate=noise_rate
    )


def get_cifar10_with_epistemic_manipulation(
    under_supported_classes: list[int],
    under_train_per_class: int = 50,
    regular_train_per_class: int = 300,
    eval_per_class: int = 100,
    noise_rate: float = 0.0,
    seed: int = 42
) -> Dataset:
    """
    Create CIFAR-10 dataset with epistemic uncertainty manipulation.
    
    Some classes are under-supported (fewer training samples) to create
    epistemic uncertainty, while others have regular support.
    
    Args:
        under_supported_classes: List of class indices to under-support (0-9)
        under_train_per_class: Training samples per under-supported class
        regular_train_per_class: Training samples per regular class
        eval_per_class: Test samples per class
        noise_rate: Label noise rate for aleatoric uncertainty
        seed: Random seed
        
    Returns:
        Dataset with manipulated epistemic uncertainty
    """
    if not KERAS_AVAILABLE:
        raise ImportError("Keras is required. Install with: pip install keras")
    
    # Load full dataset
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()
    
    # Normalize
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test_full = x_test_full.astype('float32') / 255.0
    y_train_full = y_train_full.flatten()
    y_test_full = y_test_full.flatten()
    
    np.random.seed(seed)
    
    # Build training set with class imbalance
    x_train_list = []
    y_train_list = []
    
    for class_idx in range(10):
        # Get all samples for this class
        class_mask = y_train_full == class_idx
        x_class = x_train_full[class_mask]
        y_class = y_train_full[class_mask]
        
        # Determine how many samples to use
        if class_idx in under_supported_classes:
            n_samples = under_train_per_class
        else:
            n_samples = regular_train_per_class
        
        # Randomly sample
        indices = np.random.choice(len(x_class), min(n_samples, len(x_class)), replace=False)
        x_train_list.append(x_class[indices])
        y_train_list.append(y_class[indices])
    
    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Build balanced test set
    x_test_list = []
    y_test_list = []
    
    for class_idx in range(10):
        class_mask = y_test_full == class_idx
        x_class = x_test_full[class_mask]
        y_class = y_test_full[class_mask]
        
        indices = np.random.choice(len(x_class), min(eval_per_class, len(x_class)), replace=False)
        x_test_list.append(x_class[indices])
        y_test_list.append(y_class[indices])
    
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    # Shuffle training data
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Inject label noise if requested
    noise_mask = None
    clean_labels = None
    if noise_rate > 0:
        n_samples = len(y_train)
        n_noisy = int(n_samples * noise_rate)
        
        noise_mask = np.zeros(n_samples, dtype=bool)
        noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
        noise_mask[noisy_indices] = True
        
        clean_labels = y_train.copy()
        y_train_noisy = y_train.copy()
        np.random.shuffle(y_train_noisy[noise_mask])
        y_train = y_train_noisy
    
    return Dataset(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
        is_regression=False,
        noise_mask=noise_mask,
        clean_labels=clean_labels,
        noise_rate=noise_rate
    )

# Made with Bob
