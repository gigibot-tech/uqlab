"""
Result Coordinator for the Experiment Facade pattern.

Handles result collection, storage, and retrieval.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from .base import BaseCoordinator


class ResultCoordinator(BaseCoordinator):
    """
    Coordinates all result-related operations for experiments.
    
    Responsibilities:
    - Collect and organize experiment results
    - Save results to disk (JSON, CSV, etc.)
    - Load and retrieve previous results
    - Generate result summaries and reports
    - Manage result versioning and metadata
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Result Coordinator.
        
        Args:
            config: Configuration dictionary containing:
                - results_dir: Directory to save results
                - experiment_name: Name of the experiment
                - save_format: Format to save results ("json", "csv", etc.)
                - save_checkpoints: Whether to save model checkpoints
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.results_dir: Optional[Path] = None
        self.experiment_results: Dict[str, Any] = {}
        
    def setup(self) -> None:
        """
        Setup result storage resources.
        
        Creates result directories and initializes result tracking.
        """
        self.logger.info("Setting up Result Coordinator...")
        
        # Setup results directory
        results_dir = self.config.get("results_dir", "results")
        experiment_name = self.config.get("experiment_name", "experiment")
        
        self.results_dir = Path(results_dir) / experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.experiment_results = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "config": self.config.copy(),
            "training_history": [],
            "evaluation_results": {},
            "metadata": {},
        }
        
        # Store in state
        self._state["results_dir"] = str(self.results_dir)
        self._state["experiment_name"] = experiment_name
        
        self.logger.info(f"Result Coordinator setup complete. Results dir: {self.results_dir}")
    
    def teardown(self) -> None:
        """
        Cleanup result resources.
        
        Saves final results and releases resources.
        """
        self.logger.info("Tearing down Result Coordinator...")
        
        # Save final results
        if self.experiment_results:
            self.save_results()
        
        self.experiment_results = {}
        self._state.clear()
        self.logger.info("Result Coordinator teardown complete")
    
    def add_training_epoch_result(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Add training results for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary containing training metrics
        """
        epoch_result = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.experiment_results["training_history"].append(epoch_result)
        self.logger.debug(f"Added training result for epoch {epoch}")
    
    def add_evaluation_result(self, split: str, metrics: Dict[str, Any]) -> None:
        """
        Add evaluation results for a data split.
        
        Args:
            split: Data split name ("train", "val", "test")
            metrics: Dictionary containing evaluation metrics
        """
        self.experiment_results["evaluation_results"][split] = {
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self.logger.debug(f"Added evaluation result for {split} split")
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the experiment results.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.experiment_results["metadata"][key] = value
        self.logger.debug(f"Added metadata: {key}")
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save experiment results to disk.
        
        Args:
            filename: Optional custom filename. If None, uses default naming.
        
        Returns:
            Path to the saved results file
        """
        if self.results_dir is None:
            raise ValueError("Results directory not initialized. Call setup() first.")
        
        # Add end time
        self.experiment_results["end_time"] = datetime.now().isoformat()
        
        # Determine filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"
        
        results_path = self.results_dir / filename
        
        # Save as JSON
        with open(results_path, 'w') as f:
            json.dump(self.experiment_results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_path}")
        
        return results_path
    
    def load_results(self, results_path: str) -> Dict[str, Any]:
        """
        Load experiment results from disk.
        
        Args:
            results_path: Path to the results file
        
        Returns:
            Dictionary containing the loaded results
        """
        self.logger.info(f"Loading results from: {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get the training history.
        
        Returns:
            List of training epoch results
        """
        return self.experiment_results.get("training_history", [])
    
    def get_evaluation_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all evaluation results.
        
        Returns:
            Dictionary mapping split names to evaluation metrics
        """
        return self.experiment_results.get("evaluation_results", {})
    
    def get_best_epoch(self, metric: str = "val_loss", mode: str = "min") -> Optional[Dict[str, Any]]:
        """
        Get the best epoch based on a specific metric.
        
        Args:
            metric: Metric name to optimize
            mode: "min" or "max" for optimization direction
        
        Returns:
            Dictionary containing the best epoch's results, or None if no history
        """
        history = self.get_training_history()
        
        if not history:
            return None
        
        if mode == "min":
            best_epoch = min(history, key=lambda x: x.get(metric, float('inf')))
        else:
            best_epoch = max(history, key=lambda x: x.get(metric, float('-inf')))
        
        return best_epoch
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the experiment results.
        
        Returns:
            Dictionary containing result summary
        """
        history = self.get_training_history()
        eval_results = self.get_evaluation_results()
        
        summary = {
            "experiment_name": self.experiment_results.get("experiment_name"),
            "start_time": self.experiment_results.get("start_time"),
            "end_time": self.experiment_results.get("end_time"),
            "num_epochs": len(history),
            "final_metrics": history[-1] if history else {},
            "evaluation_metrics": eval_results,
        }
        
        # Add best epoch info
        if history:
            best_val = self.get_best_epoch("val_loss", "min")
            if best_val:
                summary["best_epoch"] = best_val["epoch"]
                summary["best_val_loss"] = best_val.get("val_loss")
        
        return summary
    
    def save_checkpoint_metadata(self, checkpoint_path: str, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Save metadata about a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            epoch: Epoch number
            metrics: Metrics at the time of checkpointing
        """
        checkpoint_metadata = {
            "checkpoint_path": checkpoint_path,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        
        if "checkpoints" not in self.experiment_results["metadata"]:
            self.experiment_results["metadata"]["checkpoints"] = []
        
        self.experiment_results["metadata"]["checkpoints"].append(checkpoint_metadata)
        self.logger.info(f"Saved checkpoint metadata for epoch {epoch}")
    
    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export training history to CSV format.
        
        Args:
            filename: Optional custom filename
        
        Returns:
            Path to the exported CSV file
        """
        import pandas as pd
        
        if self.results_dir is None:
            raise ValueError("Results directory not initialized. Call setup() first.")
        
        history = self.get_training_history()
        
        if not history:
            raise ValueError("No training history to export")
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Determine filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}.csv"
        
        csv_path = self.results_dir / filename
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Training history exported to: {csv_path}")
        
        return csv_path
    
    def get_results_directory(self) -> Optional[Path]:
        """
        Get the results directory path.
        
        Returns:
            Path to the results directory
        """
        return self.results_dir

# Made with Bob
