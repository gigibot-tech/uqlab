"""
Dataset Statistics and Exploration API Endpoints

Consolidated dataset endpoints for CIFAR-10N statistics and analysis.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Configuration
# ============================================================================

def _setup_paths():
    """Setup DTAG_ROOT and data directory paths."""
    DTAG_ROOT_ENV = os.getenv("DTAG_ROOT")
    DATA_DIR_ENV = os.getenv("CIFAR10N_DATA_DIR")
    
    # Determine DTAG_ROOT
    if DTAG_ROOT_ENV:
        DTAG_ROOT = Path(DTAG_ROOT_ENV)
        if not DTAG_ROOT.exists():
            raise RuntimeError(f"DTAG_ROOT path does not exist: {DTAG_ROOT}")
    else:
        CURRENT_FILE = Path(__file__).resolve()
        POSSIBLE_PATHS = [
            Path.home() / "Desktop" / "GigiApps" / "dtag",
            Path("/dtag"),
            CURRENT_FILE.parents[5] / "dtag" if len(CURRENT_FILE.parents) > 5 else None,
        ]
        
        DTAG_ROOT = None
        for path in POSSIBLE_PATHS:
            if path and path.exists():
                DTAG_ROOT = path
                logger.info(f"Found DTAG_ROOT at: {path}")
                break
        
        if not DTAG_ROOT:
            raise RuntimeError(
                "DTAG_ROOT not found. Set DTAG_ROOT environment variable or ensure "
                "dtag directory exists in one of: " + ", ".join(str(p) for p in POSSIBLE_PATHS if p)
            )
    
    # Add to Python path
    if str(DTAG_ROOT) not in sys.path:
        sys.path.insert(0, str(DTAG_ROOT))
    
    # Set data directory
    if DATA_DIR_ENV:
        DATA_DIR = Path(DATA_DIR_ENV)
    else:
        DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "cifar10n"
    
    if not DATA_DIR.exists():
        logger.warning(f"Data directory does not exist: {DATA_DIR}")
    
    os.environ["CIFAR10N_DATA_DIR"] = str(DATA_DIR)
    
    return DTAG_ROOT, DATA_DIR


# Initialize paths at module load
DTAG_ROOT, DATA_DIR = _setup_paths()


# ============================================================================
# Helper Functions
# ============================================================================

def _get_cifar10n_dataset():
    """Lazy import of CIFAR10NDataset to avoid startup failures."""
    try:
        from uqlab.data_loaders.cifar10n_loader import CIFAR10NDataset
        import numpy as np
        return CIFAR10NDataset, np
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"CIFAR10N dependencies not available. Error: {str(e)}"
        )


def _calculate_dataset_stats(dataset, np) -> dict[str, Any]:
    """Calculate comprehensive dataset statistics."""
    total_samples = len(dataset)
    
    # Get noise info
    if dataset.noise_mask is not None:
        noisy_samples = int(np.sum(dataset.noise_mask))
        noise_rate = float(dataset.noise_rate)
    else:
        noisy_samples = 0
        noise_rate = 0.0
    
    # Get clean labels
    clean_labels = np.array(dataset.cifar10.targets)
    class_counts = {
        int(i): int(np.sum(clean_labels == i)) for i in range(10)
    }
    
    # Noise per class
    noise_per_class = {}
    for i in range(10):
        class_mask = clean_labels == i
        if dataset.noise_mask is not None:
            class_noisy = int(np.sum(dataset.noise_mask[class_mask]))
        else:
            class_noisy = 0
        
        total_in_class = int(np.sum(class_mask))
        noise_per_class[int(i)] = {
            "total": total_in_class,
            "noisy": class_noisy,
            "rate": float(class_noisy / total_in_class) if total_in_class > 0 else 0.0,
        }
    
    return {
        "total_samples": total_samples,
        "num_classes": 10,
        "noisy_samples": noisy_samples,
        "clean_samples": total_samples - noisy_samples,
        "noise_rate": noise_rate,
        "class_distribution": class_counts,
        "noise_per_class": noise_per_class,
        "class_names": dataset.cifar10.classes,
    }


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/{dataset_name}/stats")
async def get_dataset_stats_by_name(
    dataset_name: str,
    noise_type: str = Query("worse_label", description="Noise type to analyze")
) -> dict[str, Any]:
    """
    Get dataset statistics including noise distribution.
    
    Args:
        dataset_name: Name of the dataset (currently only 'cifar10' supported)
        noise_type: Type of noise to analyze (worse_label, random_label1, etc.)
    
    Returns:
        Dictionary with dataset statistics including:
        - total_samples: Total number of samples
        - num_classes: Number of classes
        - noise_rate: Percentage of noisy labels
        - class_distribution: Samples per class
        - noise_per_class: Noise statistics per class
    """
    if dataset_name.lower() != "cifar10":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported dataset: {dataset_name}. Currently only 'cifar10' is supported."
        )
    
    CIFAR10NDataset, np = _get_cifar10n_dataset()
    
    try:
        dataset = CIFAR10NDataset(
            root=str(DATA_DIR),
            noise_type=noise_type,
            download=False,
            train=True
        )
        return _calculate_dataset_stats(dataset, np)
    except Exception as e:
        logger.error(f"Error loading dataset stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dataset error: {str(e)}")


@router.get("/cifar10n/stats")
async def get_cifar10n_stats(
    noise_type: str = Query("worse_label", description="Noise type to analyze")
) -> dict[str, Any]:
    """Get CIFAR-10N dataset statistics (legacy endpoint)."""
    CIFAR10NDataset, np = _get_cifar10n_dataset()
    
    try:
        dataset = CIFAR10NDataset(
            root=str(DATA_DIR),
            noise_type=noise_type,
            download=False,
            train=True
        )
        return _calculate_dataset_stats(dataset, np)
    except Exception as e:
        logger.error(f"Error loading dataset stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dataset error: {str(e)}")


@router.get("/cifar10n/confusion-matrix")
async def get_confusion_matrix(
    noise_type: str = Query("worse_label", description="Noise type to analyze")
) -> dict[str, Any]:
    """Get confusion matrix showing clean vs noisy label relationships."""
    CIFAR10NDataset, np = _get_cifar10n_dataset()
    
    try:
        dataset = CIFAR10NDataset(
            root=str(DATA_DIR),
            noise_type=noise_type,
            download=False,
            train=True
        )
        
        # Get clean labels
        clean_labels = np.array(dataset.cifar10.targets)
        
        # Get noisy labels
        if dataset.noisy_labels is not None:
            noisy_labels = dataset.noisy_labels
        else:
            noisy_labels = clean_labels
        
        # Build confusion matrix
        confusion = np.zeros((10, 10), dtype=int)
        for clean, noisy in zip(clean_labels, noisy_labels):
            confusion[clean, noisy] += 1
        
        return {
            "matrix": confusion.tolist(),
            "class_names": dataset.cifar10.classes,
        }
    except Exception as e:
        logger.error(f"Error loading confusion matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dataset error: {str(e)}")

# Made with Bob
