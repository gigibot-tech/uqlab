"""
Evaluation Metrics - Comprehensive metrics for uncertainty quantification.

This module provides:
- Classification metrics (accuracy, precision, recall, F1)
- Uncertainty metrics (AUROC, UDE, ECE, Brier score)
- Calibration metrics
- Confusion matrix utilities
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shared.types import FloatArray, LabelArray, MetricsDict
from shared.utils import to_numpy


class MetricsCalculator:
    """
    Calculator for evaluation metrics.
    
    Provides methods for computing:
    - Classification metrics
    - Uncertainty metrics
    - Calibration metrics
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    def calculate_accuracy(
        self,
        predictions: LabelArray,
        targets: LabelArray,
    ) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            predictions: Predicted labels
            targets: True labels
        
        Returns:
            Accuracy score
        """
        return float(accuracy_score(targets, predictions))
    
    def calculate_precision_recall_f1(
        self,
        predictions: LabelArray,
        targets: LabelArray,
        average: str = "macro",
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            average: Averaging strategy ('micro', 'macro', 'weighted')
        
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        precision = precision_score(targets, predictions, average=average, zero_division=0)
        recall = recall_score(targets, predictions, average=average, zero_division=0)
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }
    
    def calculate_confusion_matrix(
        self,
        predictions: LabelArray,
        targets: LabelArray,
        normalize: Optional[str] = None,
    ) -> NDArray:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            normalize: Normalization mode ('true', 'pred', 'all', or None)
        
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(targets, predictions, labels=range(self.num_classes))
        
        if normalize == "true":
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm.astype('float') / cm.sum()
        
        return cm
    
    def calculate_auroc(
        self,
        uncertainties: FloatArray,
        is_correct: NDArray[np.bool_],
        use_correct_as_positive: bool = False,
    ) -> float:
        """
        Calculate AUROC for uncertainty quantification.
        
        Args:
            uncertainties: Uncertainty scores
            is_correct: Boolean array indicating correct predictions
            use_correct_as_positive: If True, correct predictions are positive class
        
        Returns:
            AUROC score
        """
        if use_correct_as_positive:
            # Correct predictions should have low uncertainty
            labels = is_correct.astype(int)
            scores = -uncertainties  # Negate so low uncertainty = high score
        else:
            # Incorrect predictions should have high uncertainty
            labels = (~is_correct).astype(int)
            scores = uncertainties
        
        # Check if we have both classes
        if len(np.unique(labels)) < 2:
            return 0.5  # Random performance if only one class
        
        try:
            return float(roc_auc_score(labels, scores))
        except ValueError:
            return 0.5
    
    def calculate_epistemic_auroc(
        self,
        uncertainties: FloatArray,
        is_epistemic: NDArray[np.bool_],
    ) -> float:
        """
        Calculate AUROC for epistemic uncertainty detection.
        
        Args:
            uncertainties: Uncertainty scores
            is_epistemic: Boolean array indicating epistemic samples
        
        Returns:
            AUROC score
        """
        labels = is_epistemic.astype(int)
        
        if len(np.unique(labels)) < 2:
            return 0.5
        
        try:
            return float(roc_auc_score(labels, uncertainties))
        except ValueError:
            return 0.5
    
    def calculate_aleatoric_auroc(
        self,
        uncertainties: FloatArray,
        is_noisy: NDArray[np.bool_],
    ) -> float:
        """
        Calculate AUROC for aleatoric uncertainty detection.
        
        Args:
            uncertainties: Uncertainty scores
            is_noisy: Boolean array indicating noisy labels
        
        Returns:
            AUROC score
        """
        labels = is_noisy.astype(int)
        
        if len(np.unique(labels)) < 2:
            return 0.5
        
        try:
            return float(roc_auc_score(labels, uncertainties))
        except ValueError:
            return 0.5
    
    def calculate_ude(
        self,
        uncertainties: FloatArray,
        is_correct: NDArray[np.bool_],
        num_bins: int = 10,
    ) -> float:
        """
        Calculate Uncertainty Disagreement Error (UDE).
        
        UDE measures the correlation between uncertainty and correctness.
        Lower UDE indicates better uncertainty calibration.
        
        Args:
            uncertainties: Uncertainty scores
            is_correct: Boolean array indicating correct predictions
            num_bins: Number of bins for discretization
        
        Returns:
            UDE score
        """
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_correct = is_correct[sorted_indices]
        
        # Split into bins
        bin_size = len(sorted_correct) // num_bins
        ude_sum = 0.0
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_correct)
            
            bin_correct = sorted_correct[start_idx:end_idx]
            bin_accuracy = bin_correct.mean()
            
            # UDE: difference from expected accuracy
            # Low uncertainty bins should have high accuracy
            expected_accuracy = 1.0 - (i / num_bins)
            ude_sum += abs(bin_accuracy - expected_accuracy)
        
        return float(ude_sum / num_bins)
    
    def calculate_ece(
        self,
        probabilities: FloatArray,
        targets: LabelArray,
        num_bins: int = 15,
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            probabilities: Predicted probabilities (N, C)
            targets: True labels (N,)
            num_bins: Number of bins
        
        Returns:
            ECE score
        """
        # Get confidence (max probability) and predictions
        confidences = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        accuracies = (predictions == targets).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def calculate_brier_score(
        self,
        probabilities: FloatArray,
        targets: LabelArray,
    ) -> float:
        """
        Calculate Brier score.
        
        Args:
            probabilities: Predicted probabilities (N, C)
            targets: True labels (N,)
        
        Returns:
            Brier score
        """
        # Convert targets to one-hot
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(targets)), targets] = 1
        
        # Calculate Brier score
        brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
        
        return float(brier)
    
    def calculate_nll(
        self,
        probabilities: FloatArray,
        targets: LabelArray,
    ) -> float:
        """
        Calculate Negative Log-Likelihood.
        
        Args:
            probabilities: Predicted probabilities (N, C)
            targets: True labels (N,)
        
        Returns:
            NLL score
        """
        # Get probabilities of true class
        true_class_probs = probabilities[np.arange(len(targets)), targets]
        
        # Clip to avoid log(0)
        true_class_probs = np.clip(true_class_probs, 1e-10, 1.0)
        
        # Calculate NLL
        nll = -np.mean(np.log(true_class_probs))
        
        return float(nll)
    
    def calculate_all_metrics(
        self,
        probabilities: FloatArray,
        predictions: LabelArray,
        targets: LabelArray,
        uncertainties: Optional[FloatArray] = None,
        is_noisy: Optional[NDArray[np.bool_]] = None,
        is_epistemic: Optional[NDArray[np.bool_]] = None,
    ) -> MetricsDict:
        """
        Calculate all available metrics.
        
        Args:
            probabilities: Predicted probabilities (N, C)
            predictions: Predicted labels (N,)
            targets: True labels (N,)
            uncertainties: Uncertainty scores (N,)
            is_noisy: Boolean array indicating noisy labels
            is_epistemic: Boolean array indicating epistemic samples
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Classification metrics
        metrics["accuracy"] = self.calculate_accuracy(predictions, targets)
        metrics.update(self.calculate_precision_recall_f1(predictions, targets))
        
        # Calibration metrics
        metrics["ece"] = self.calculate_ece(probabilities, targets)
        metrics["brier_score"] = self.calculate_brier_score(probabilities, targets)
        metrics["nll"] = self.calculate_nll(probabilities, targets)
        
        # Uncertainty metrics
        if uncertainties is not None:
            is_correct = (predictions == targets)
            metrics["auroc"] = self.calculate_auroc(uncertainties, is_correct)
            metrics["ude"] = self.calculate_ude(uncertainties, is_correct)
            
            if is_noisy is not None:
                metrics["auroc_aleatoric"] = self.calculate_aleatoric_auroc(
                    uncertainties, is_noisy
                )
            
            if is_epistemic is not None:
                metrics["auroc_epistemic"] = self.calculate_epistemic_auroc(
                    uncertainties, is_epistemic
                )
        
        return metrics


def calculate_per_class_metrics(
    predictions: LabelArray,
    targets: LabelArray,
    num_classes: int = 10,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate per-class metrics.
    
    Args:
        predictions: Predicted labels
        targets: True labels
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class index to metrics
    """
    per_class = {}
    
    for class_idx in range(num_classes):
        # Binary classification for this class
        class_predictions = (predictions == class_idx).astype(int)
        class_targets = (targets == class_idx).astype(int)
        
        # Calculate metrics
        if class_targets.sum() > 0:  # Only if class exists in targets
            precision = precision_score(class_targets, class_predictions, zero_division=0)
            recall = recall_score(class_targets, class_predictions, zero_division=0)
            f1 = f1_score(class_targets, class_predictions, zero_division=0)
            
            per_class[class_idx] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(class_targets.sum()),
            }
    
    return per_class


# Made with Bob