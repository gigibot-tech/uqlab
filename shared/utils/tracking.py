"""Unified experiment tracking interface for MLflow and JSON.

This module provides a single interface that automatically switches between
MLflow and JSON-based tracking based on configuration.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Unified interface for MLflow and JSON tracking.
    
    Automatically uses MLflow if URI provided, otherwise falls back to JSON.
    
    Usage:
        # JSON mode
        tracker = ExperimentTracker(mlflow_uri=None)
        
        # MLflow mode
        tracker = ExperimentTracker(mlflow_uri="http://localhost:5000")
        
        # Same interface for both
        tracker.log_param("learning_rate", 0.001)
        tracker.log_metric("accuracy", 0.95, step=10)
        tracker.log_artifact("model.pt")
        
        # Load results
        results = tracker.load_results(run_id="abc123")
    """
    
    def __init__(
        self,
        experiment_name: str = "default",
        mlflow_uri: Optional[str] = None,
        json_dir: str = "experiments"
    ):
        """Initialize tracker with automatic backend selection.
        
        Args:
            experiment_name: Name of the experiment
            mlflow_uri: MLflow tracking URI (None = use JSON)
            json_dir: Directory for JSON files (when not using MLflow)
        """
        self.experiment_name = experiment_name
        self.json_dir = Path(json_dir)
        self.use_mlflow = False
        self.run_id = None
        self._json_data = None
        
        # Try MLflow if URI provided
        if mlflow_uri and MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(experiment_name)
                self.use_mlflow = True
            except Exception as e:
                print(f"MLflow unavailable ({e}), falling back to JSON")
                self._init_json()
        else:
            self._init_json()
    
    def _init_json(self):
        """Initialize JSON-based tracking."""
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._json_data = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "params": {},
            "metrics": {},
            "artifacts": []
        }
    
    def start_run(self, run_name: Optional[str] = None) -> 'ExperimentTracker':
        """Start a new run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Self for chaining
        """
        if self.use_mlflow:
            mlflow.start_run(run_name=run_name)
            self.run_id = mlflow.active_run().info.run_id
        else:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            if run_name:
                self._json_data["run_name"] = run_name
        return self
    
    def end_run(self):
        """End the current run."""
        if self.use_mlflow:
            mlflow.end_run()
        else:
            self._save_json()
    
    def log_param(self, key: str, value: Any):
        """Log a parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        if self.use_mlflow:
            mlflow.log_param(key, value)
        else:
            self._json_data["params"][key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self._json_data["params"].update(params)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        if self.use_mlflow:
            mlflow.log_metric(key, value, step=step)
        else:
            if key not in self._json_data["metrics"]:
                self._json_data["metrics"][key] = []
            self._json_data["metrics"][key].append({
                "step": step if step is not None else len(self._json_data["metrics"][key]),
                "value": value
            })
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)
    
    def log_artifact(self, path: str):
        """Log an artifact file.
        
        Args:
            path: Path to the artifact file
        """
        if self.use_mlflow:
            mlflow.log_artifact(path)
        else:
            self._json_data["artifacts"].append(path)
    
    def _save_json(self):
        """Save JSON data to file."""
        if not self._json_data:
            return
        
        exp_dir = self.json_dir / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = exp_dir / f"{self.run_id}.json"
        with open(filepath, 'w') as f:
            json.dump(self._json_data, f, indent=2)
    
    def load_results(self, run_id: str) -> Dict[str, Any]:
        """Load results from a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with params, metrics, and artifacts
        """
        if self.use_mlflow:
            return self._load_mlflow_results(run_id)
        else:
            return self._load_json_results(run_id)
    
    def _load_mlflow_results(self, run_id: str) -> Dict[str, Any]:
        """Load results from MLflow."""
        run = mlflow.get_run(run_id)
        return {
            "run_id": run_id,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "artifacts": [a.path for a in mlflow.MlflowClient().list_artifacts(run_id)]
        }
    
    def _load_json_results(self, run_id: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        # Search in all experiment directories
        for exp_dir in self.json_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            filepath = exp_dir / f"{run_id}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
        raise FileNotFoundError(f"Run {run_id} not found")
    
    def list_experiments(self) -> List[str]:
        """List all experiments.
        
        Returns:
            List of experiment names
        """
        if self.use_mlflow:
            return [exp.name for exp in mlflow.search_experiments()]
        else:
            return [d.name for d in self.json_dir.iterdir() if d.is_dir()]
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters from current run."""
        if self.use_mlflow:
            return mlflow.active_run().data.params
        else:
            return self._json_data["params"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from current run."""
        if self.use_mlflow:
            return mlflow.active_run().data.metrics
        else:
            return self._json_data["metrics"]
    
    @classmethod
    def from_run_id(
        cls,
        run_id: str,
        mlflow_uri: Optional[str] = None,
        json_dir: str = "experiments"
    ) -> 'ExperimentTracker':
        """Create tracker from existing run.
        
        Args:
            run_id: Run identifier
            mlflow_uri: MLflow tracking URI (None = use JSON)
            json_dir: Directory for JSON files
            
        Returns:
            ExperimentTracker instance
        """
        tracker = cls(mlflow_uri=mlflow_uri, json_dir=json_dir)
        results = tracker.load_results(run_id)
        tracker.run_id = run_id
        if not tracker.use_mlflow:
            tracker._json_data = results
        return tracker
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()

# Made with Bob
