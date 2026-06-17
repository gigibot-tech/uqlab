"""
Result Storage

Utilities for storing and retrieving experiment results.
Handles result persistence, cleanup, and retrieval.
"""

import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pickle

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Storage Manager
# ============================================================================

class ResultStorage:
    """
    Manages experiment result storage.
    
    Features:
    - Save/load results
    - Metadata management
    - Cleanup utilities
    - Result querying
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize result storage.
        
        Args:
            base_dir: Base directory for all results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.base_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata index."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "experiments": {},
                "batches": {},
                "created_at": datetime.utcnow().isoformat(),
            }
    
    def _save_metadata(self):
        """Save metadata index."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_experiment_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        metrics: Optional[pd.DataFrame] = None,
        signals: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        Save experiment results.
        
        Args:
            experiment_id: Experiment ID
            results: Results dictionary
            metrics: Optional metrics DataFrame
            signals: Optional signals DataFrame
        
        Returns:
            Path to saved results
        """
        # Create experiment directory
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results JSON
        results_file = exp_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        if metrics is not None:
            metrics_file = exp_dir / "metrics.csv"
            metrics.to_csv(metrics_file, index=False)
        
        # Save signals
        if signals is not None:
            signals_file = exp_dir / "signals.csv"
            signals.to_csv(signals_file, index=False)
        
        # Update metadata
        self.metadata["experiments"][experiment_id] = {
            "path": str(exp_dir),
            "saved_at": datetime.utcnow().isoformat(),
            "has_metrics": metrics is not None,
            "has_signals": signals is not None,
        }
        self._save_metadata()
        
        logger.info(f"Saved results for experiment {experiment_id} to {exp_dir}")
        
        return exp_dir
    
    def load_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load experiment results.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            Results dictionary or None if not found
        """
        if experiment_id not in self.metadata["experiments"]:
            logger.warning(f"Experiment {experiment_id} not found in metadata")
            return None
        
        exp_dir = Path(self.metadata["experiments"][experiment_id]["path"])
        
        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {exp_dir}")
            return None
        
        # Load results
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Load metrics if available
        metrics_file = exp_dir / "metrics.csv"
        if metrics_file.exists():
            results["metrics"] = pd.read_csv(metrics_file)
        
        # Load signals if available
        signals_file = exp_dir / "signals.csv"
        if signals_file.exists():
            results["signals"] = pd.read_csv(signals_file)
        
        logger.info(f"Loaded results for experiment {experiment_id}")
        
        return results
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.
        
        Returns:
            List of experiment metadata
        """
        experiments = []
        
        for exp_id, exp_meta in self.metadata["experiments"].items():
            experiments.append({
                "id": exp_id,
                "path": exp_meta["path"],
                "saved_at": exp_meta["saved_at"],
                "has_metrics": exp_meta.get("has_metrics", False),
                "has_signals": exp_meta.get("has_signals", False),
            })
        
        # Sort by saved_at descending
        experiments.sort(key=lambda x: x["saved_at"], reverse=True)
        
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete experiment results.
        
        Args:
            experiment_id: Experiment ID
        
        Returns:
            True if deleted, False otherwise
        """
        if experiment_id not in self.metadata["experiments"]:
            logger.warning(f"Experiment {experiment_id} not found")
            return False
        
        exp_dir = Path(self.metadata["experiments"][experiment_id]["path"])
        
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            logger.info(f"Deleted experiment directory: {exp_dir}")
        
        # Remove from metadata
        del self.metadata["experiments"][experiment_id]
        self._save_metadata()
        
        logger.info(f"Deleted experiment {experiment_id}")
        
        return True
    
    def cleanup_old_results(self, days: int = 30) -> int:
        """
        Cleanup results older than specified days.
        
        Args:
            days: Number of days to keep
        
        Returns:
            Number of experiments deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0
        
        experiments_to_delete = []
        
        for exp_id, exp_meta in self.metadata["experiments"].items():
            saved_at = datetime.fromisoformat(exp_meta["saved_at"])
            
            if saved_at < cutoff_date:
                experiments_to_delete.append(exp_id)
        
        for exp_id in experiments_to_delete:
            if self.delete_experiment(exp_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old experiments")
        
        return deleted_count
    
    def get_storage_size(self) -> Dict[str, float]:
        """
        Get storage size statistics.
        
        Returns:
            Dictionary with size information in GB
        """
        total_size = 0
        
        for exp_id, exp_meta in self.metadata["experiments"].items():
            exp_dir = Path(exp_meta["path"])
            
            if exp_dir.exists():
                for file in exp_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
        
        return {
            "total_gb": total_size / (1024 ** 3),
            "total_experiments": len(self.metadata["experiments"]),
            "avg_size_mb": (total_size / len(self.metadata["experiments"]) / (1024 ** 2))
                if self.metadata["experiments"] else 0,
        }


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """Manages model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        experiment_id: str,
        epoch: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            experiment_id: Experiment ID
            epoch: Epoch number
            model_state: Model state dict
            optimizer_state: Optional optimizer state
            metrics: Optional metrics
        
        Returns:
            Path to checkpoint file
        """
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}_epoch_{epoch}.pt"
        
        checkpoint = {
            "experiment_id": experiment_id,
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "metrics": metrics,
            "saved_at": datetime.utcnow().isoformat(),
        }
        
        import torch
        torch.save(checkpoint, checkpoint_file)
        
        logger.info(f"Saved checkpoint: {checkpoint_file}")
        
        return checkpoint_file
    
    def load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_file: Path to checkpoint
        
        Returns:
            Checkpoint dictionary
        """
        import torch
        
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        
        logger.info(f"Loaded checkpoint: {checkpoint_file}")
        
        return checkpoint
    
    def list_checkpoints(self, experiment_id: Optional[str] = None) -> List[Path]:
        """
        List available checkpoints.
        
        Args:
            experiment_id: Optional experiment ID to filter
        
        Returns:
            List of checkpoint paths
        """
        if experiment_id:
            pattern = f"{experiment_id}_epoch_*.pt"
        else:
            pattern = "*_epoch_*.pt"
        
        checkpoints = sorted(self.checkpoint_dir.glob(pattern))
        
        return checkpoints


# ============================================================================
# Convenience Functions
# ============================================================================

def create_storage(base_dir: Path) -> ResultStorage:
    """Create result storage instance."""
    return ResultStorage(base_dir)


def save_results(
    storage: ResultStorage,
    experiment_id: str,
    results: Dict[str, Any],
    metrics: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
) -> Path:
    """Save experiment results."""
    return storage.save_experiment_results(experiment_id, results, metrics, signals)


def load_results(storage: ResultStorage, experiment_id: str) -> Optional[Dict[str, Any]]:
    """Load experiment results."""
    return storage.load_experiment_results(experiment_id)


# Made with Bob