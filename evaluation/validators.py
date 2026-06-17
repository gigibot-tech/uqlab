"""
Validation Utilities - Validation for experiments, configs, and results.

This module provides:
- Experiment configuration validation
- Result validation
- Data quality checks
- Consistency checks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shared.types import MetricsDict, SignalDict
from shared.utils import get_logger


class ConfigValidator:
    """Validator for experiment configurations."""
    
    def __init__(self):
        self.logger = get_logger()
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_data_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate data configuration.
        
        Args:
            config: Data configuration dictionary
        
        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Check required fields
        required_fields = ["noise_type", "num_classes"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        # Validate noise type
        if "noise_type" in config:
            valid_noise_types = ["clean", "worst", "aggre", "random1", "random2", "random3"]
            if config["noise_type"] not in valid_noise_types:
                self.errors.append(
                    f"Invalid noise_type: {config['noise_type']}. "
                    f"Must be one of {valid_noise_types}"
                )
        
        # Validate num_classes
        if "num_classes" in config:
            if not isinstance(config["num_classes"], int) or config["num_classes"] <= 0:
                self.errors.append("num_classes must be a positive integer")
        
        # Validate split sizes
        if "train_per_class" in config:
            if config["train_per_class"] <= 0:
                self.errors.append("train_per_class must be positive")
        
        if "eval_per_group" in config:
            if config["eval_per_group"] <= 0:
                self.errors.append("eval_per_group must be positive")
        
        return len(self.errors) == 0
    
    def validate_model_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
        
        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Check required fields
        required_fields = ["architecture"]
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")
        
        # Validate architecture
        if "architecture" in config:
            valid_architectures = [
                "dinov2_mlp",
                "resnet18_mcdropout",
                "resnet50_mcdropout",
                "cnn_mcdropout",
            ]
            if config["architecture"] not in valid_architectures:
                self.errors.append(
                    f"Invalid architecture: {config['architecture']}. "
                    f"Must be one of {valid_architectures}"
                )
        
        # Validate hidden_dim
        if "hidden_dim" in config:
            if not isinstance(config["hidden_dim"], int) or config["hidden_dim"] <= 0:
                self.errors.append("hidden_dim must be a positive integer")
        
        # Validate dropout
        if "dropout" in config:
            if not 0 <= config["dropout"] < 1:
                self.errors.append("dropout must be in range [0, 1)")
        
        return len(self.errors) == 0
    
    def validate_training_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration dictionary
        
        Returns:
            True if valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate epochs
        if "epochs" in config:
            if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
                self.errors.append("epochs must be a positive integer")
        
        # Validate learning rate
        if "learning_rate" in config:
            if config["learning_rate"] <= 0:
                self.errors.append("learning_rate must be positive")
            if config["learning_rate"] > 1.0:
                self.warnings.append("learning_rate > 1.0 is unusually high")
        
        # Validate batch size
        if "train_batch_size" in config:
            if config["train_batch_size"] <= 0:
                self.errors.append("train_batch_size must be positive")
            if config["train_batch_size"] > 1024:
                self.warnings.append("train_batch_size > 1024 may cause memory issues")
        
        # Validate weight decay
        if "weight_decay" in config:
            if config["weight_decay"] < 0:
                self.errors.append("weight_decay must be non-negative")
        
        return len(self.errors) == 0
    
    def validate_full_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate complete experiment configuration.
        
        Args:
            config: Full configuration dictionary
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        all_errors = []
        all_warnings = []
        
        # Validate data config
        if "data" in config:
            if not self.validate_data_config(config["data"]):
                all_errors.extend(self.errors)
            all_warnings.extend(self.warnings)
        
        # Validate model config
        if "model" in config:
            if not self.validate_model_config(config["model"]):
                all_errors.extend(self.errors)
            all_warnings.extend(self.warnings)
        
        # Validate training config
        if "training" in config:
            if not self.validate_training_config(config["training"]):
                all_errors.extend(self.errors)
            all_warnings.extend(self.warnings)
        
        is_valid = len(all_errors) == 0
        return is_valid, all_errors, all_warnings


class ResultValidator:
    """Validator for experiment results."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def validate_metrics(self, metrics: MetricsDict) -> Tuple[bool, List[str]]:
        """
        Validate metrics dictionary.
        
        Args:
            metrics: Metrics dictionary
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for NaN or Inf values
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    errors.append(f"Metric '{key}' is NaN")
                if np.isinf(value):
                    errors.append(f"Metric '{key}' is Inf")
        
        # Validate specific metrics
        if "accuracy" in metrics:
            if not 0 <= metrics["accuracy"] <= 1:
                errors.append(f"Accuracy {metrics['accuracy']} not in [0, 1]")
        
        if "auroc" in metrics:
            if not 0 <= metrics["auroc"] <= 1:
                errors.append(f"AUROC {metrics['auroc']} not in [0, 1]")
        
        if "ece" in metrics:
            if not 0 <= metrics["ece"] <= 1:
                errors.append(f"ECE {metrics['ece']} not in [0, 1]")
        
        return len(errors) == 0, errors
    
    def validate_signals(self, signals: SignalDict) -> Tuple[bool, List[str]]:
        """
        Validate signals dictionary.
        
        Args:
            signals: Signals dictionary
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for empty signals
        if not signals:
            errors.append("No signals provided")
            return False, errors
        
        # Check signal shapes
        shapes = [arr.shape for arr in signals.values()]
        if len(set(shapes)) > 1:
            errors.append(f"Inconsistent signal shapes: {shapes}")
        
        # Check for NaN or Inf values
        for name, values in signals.items():
            if np.any(np.isnan(values)):
                errors.append(f"Signal '{name}' contains NaN values")
            if np.any(np.isinf(values)):
                errors.append(f"Signal '{name}' contains Inf values")
        
        return len(errors) == 0, errors
    
    def validate_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_classes: int,
    ) -> Tuple[bool, List[str]]:
        """
        Validate predictions and targets.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            num_classes: Number of classes
        
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check shapes
        if predictions.shape != targets.shape:
            errors.append(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        # Check value ranges
        if np.any(predictions < 0) or np.any(predictions >= num_classes):
            errors.append(f"Predictions out of range [0, {num_classes})")
        
        if np.any(targets < 0) or np.any(targets >= num_classes):
            errors.append(f"Targets out of range [0, {num_classes})")
        
        return len(errors) == 0, errors


class DataQualityChecker:
    """Checker for data quality issues."""
    
    def __init__(self):
        self.logger = get_logger()
    
    def check_class_balance(
        self,
        labels: np.ndarray,
        num_classes: int,
        threshold: float = 0.1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for class imbalance.
        
        Args:
            labels: Label array
            num_classes: Number of classes
            threshold: Imbalance threshold (fraction of mean)
        
        Returns:
            Tuple of (is_balanced, info_dict)
        """
        # Count samples per class
        class_counts = np.bincount(labels, minlength=num_classes)
        mean_count = class_counts.mean()
        
        # Check for imbalance
        imbalanced_classes = []
        for class_idx, count in enumerate(class_counts):
            if count < mean_count * threshold:
                imbalanced_classes.append((class_idx, count))
        
        is_balanced = len(imbalanced_classes) == 0
        
        info = {
            "class_counts": class_counts.tolist(),
            "mean_count": float(mean_count),
            "imbalanced_classes": imbalanced_classes,
            "is_balanced": is_balanced,
        }
        
        return is_balanced, info
    
    def check_label_noise(
        self,
        clean_labels: np.ndarray,
        noisy_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze label noise.
        
        Args:
            clean_labels: Clean labels
            noisy_labels: Noisy labels
        
        Returns:
            Dictionary with noise statistics
        """
        # Calculate noise rate
        is_noisy = (clean_labels != noisy_labels)
        noise_rate = is_noisy.mean()
        
        # Per-class noise rates
        num_classes = max(clean_labels.max(), noisy_labels.max()) + 1
        per_class_noise = {}
        
        for class_idx in range(num_classes):
            class_mask = (clean_labels == class_idx)
            if class_mask.sum() > 0:
                class_noise_rate = is_noisy[class_mask].mean()
                per_class_noise[int(class_idx)] = float(class_noise_rate)
        
        return {
            "overall_noise_rate": float(noise_rate),
            "num_noisy_samples": int(is_noisy.sum()),
            "per_class_noise_rate": per_class_noise,
        }
    
    def check_data_consistency(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[bool, List[str]]:
        """
        Check data consistency.
        
        Args:
            data: Data array
            labels: Label array
        
        Returns:
            Tuple of (is_consistent, issues)
        """
        issues = []
        
        # Check for NaN or Inf
        if np.any(np.isnan(data)):
            issues.append("Data contains NaN values")
        if np.any(np.isinf(data)):
            issues.append("Data contains Inf values")
        
        # Check shape consistency
        if len(data) != len(labels):
            issues.append(f"Data length {len(data)} != labels length {len(labels)}")
        
        # Check for duplicates
        unique_samples = np.unique(data, axis=0)
        if len(unique_samples) < len(data):
            duplicate_rate = 1 - (len(unique_samples) / len(data))
            issues.append(f"Found {duplicate_rate:.1%} duplicate samples")
        
        return len(issues) == 0, issues


# Made with Bob