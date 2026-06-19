"""
Evaluation Coordinator for the Experiment Facade pattern.

Handles model evaluation, uncertainty quantification, and metrics calculation.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging

from .base import BaseCoordinator


class EvaluationCoordinator(BaseCoordinator):
    """
    Coordinates all evaluation-related operations for experiments.
    
    Responsibilities:
    - Perform model evaluation on test/validation sets
    - Calculate uncertainty metrics (epistemic, aleatoric)
    - Compute MC Dropout predictions
    - Calculate AUROC, accuracy, and other metrics
    - Generate uncertainty signals (DualXDA)
    - Perform triage analysis
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Evaluation Coordinator.
        
        Args:
            config: Configuration dictionary containing:
                - mc_passes: Number of MC Dropout forward passes
                - uncertainty_signals: List of uncertainty signals to compute
                - eval_batch_size: Batch size for evaluation
                - compute_auroc: Whether to compute AUROC metrics
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.mc_passes: int = config.get("mc_passes", 20)
        self.predictions: Optional[np.ndarray] = None
        self.uncertainties: Optional[Dict[str, np.ndarray]] = None
        
    def setup(self) -> None:
        """
        Setup evaluation resources.
        
        Initializes evaluation configuration and prepares for metrics calculation.
        """
        self.logger.info("Setting up Evaluation Coordinator...")
        
        self.mc_passes = self.config.get("mc_passes", 20)
        
        # Store config in state
        self._state["mc_passes"] = self.mc_passes
        self._state["uncertainty_signals"] = self.config.get("uncertainty_signals", [])
        
        self.logger.info(f"Evaluation Coordinator setup complete. MC passes: {self.mc_passes}")
    
    def teardown(self) -> None:
        """
        Cleanup evaluation resources.
        
        Releases prediction and uncertainty arrays.
        """
        self.logger.info("Tearing down Evaluation Coordinator...")
        self.predictions = None
        self.uncertainties = None
        self._state.clear()
        self.logger.info("Evaluation Coordinator teardown complete")
    
    def evaluate_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        compute_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the model on the given data loader.
        
        Args:
            model: The model to evaluate
            data_loader: Data loader for evaluation
            device: Device to evaluate on
            compute_uncertainty: Whether to compute uncertainty estimates
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info("Evaluating model...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_logits = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                all_logits.append(output.cpu().numpy())
                all_predictions.append(output.argmax(dim=1).cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        logits = np.concatenate(all_logits)
        
        # Calculate basic metrics
        accuracy = (predictions == targets).mean() * 100.0
        
        results = {
            "accuracy": accuracy,
            "num_samples": len(predictions),
        }
        
        # Compute uncertainty if requested
        if compute_uncertainty:
            uncertainties = self._compute_mc_dropout_uncertainty(
                model, data_loader, device
            )
            results["uncertainties"] = uncertainties
        
        self.logger.info(f"Evaluation complete. Accuracy: {accuracy:.2f}%")
        
        return results
    
    def _compute_mc_dropout_uncertainty(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """
        Compute uncertainty estimates using MC Dropout.
        
        Args:
            model: The model with dropout layers
            data_loader: Data loader for evaluation
            device: Device to evaluate on
        
        Returns:
            Dictionary containing uncertainty estimates
        """
        self.logger.info(f"Computing MC Dropout uncertainty with {self.mc_passes} passes...")
        
        # Enable dropout during inference
        model.train()  # This enables dropout
        
        all_mc_predictions = []
        
        # Perform multiple forward passes
        for pass_idx in range(self.mc_passes):
            pass_predictions = []
            
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(device)
                    output = model(data)
                    probs = torch.softmax(output, dim=1)
                    pass_predictions.append(probs.cpu().numpy())
            
            all_mc_predictions.append(np.concatenate(pass_predictions))
        
        # Stack predictions: (mc_passes, num_samples, num_classes)
        mc_predictions = np.stack(all_mc_predictions)
        
        # Calculate uncertainty metrics
        uncertainties = self._calculate_uncertainty_metrics(mc_predictions)
        
        self.logger.info("MC Dropout uncertainty computation complete")
        
        return uncertainties
    
    def _calculate_uncertainty_metrics(
        self,
        mc_predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate various uncertainty metrics from MC Dropout predictions.
        
        Args:
            mc_predictions: Array of shape (mc_passes, num_samples, num_classes)
        
        Returns:
            Dictionary containing uncertainty metrics
        """
        # Mean prediction across MC passes
        mean_pred = mc_predictions.mean(axis=0)
        
        # Predictive entropy (aleatoric + epistemic)
        predictive_entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
        
        # Mutual information (epistemic uncertainty)
        entropy_per_pass = -np.sum(
            mc_predictions * np.log(mc_predictions + 1e-10),
            axis=2
        )
        expected_entropy = entropy_per_pass.mean(axis=0)
        mutual_information = predictive_entropy - expected_entropy
        
        # Variation ratio
        mode_count = np.apply_along_axis(
            lambda x: np.bincount(x).max(),
            axis=0,
            arr=mc_predictions.argmax(axis=2)
        )
        variation_ratio = 1.0 - (mode_count / mc_predictions.shape[0])
        
        return {
            "predictive_entropy": predictive_entropy,
            "mutual_information": mutual_information,
            "expected_entropy": expected_entropy,
            "variation_ratio": variation_ratio,
            "mean_prediction": mean_pred,
        }
    
    def compute_auroc_metrics(
        self,
        uncertainties: Dict[str, np.ndarray],
        is_correct: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute AUROC metrics for uncertainty estimates.
        
        Args:
            uncertainties: Dictionary of uncertainty estimates
            is_correct: Boolean array indicating correct predictions
        
        Returns:
            Dictionary containing AUROC scores for each uncertainty metric
        """
        from sklearn.metrics import roc_auc_score
        
        auroc_scores = {}
        
        for metric_name, uncertainty_values in uncertainties.items():
            if metric_name == "mean_prediction":
                continue  # Skip prediction probabilities
            
            try:
                # Higher uncertainty should correlate with incorrect predictions
                auroc = roc_auc_score(~is_correct, uncertainty_values)
                auroc_scores[f"auroc_{metric_name}"] = auroc
            except Exception as e:
                self.logger.warning(f"Failed to compute AUROC for {metric_name}: {e}")
        
        return auroc_scores
    
    def compute_dualxda_signals(
        self,
        predictions: np.ndarray,
        uncertainties: Dict[str, np.ndarray],
        targets: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute DualXDA uncertainty signals.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: Ground truth labels
        
        Returns:
            Dictionary containing DualXDA signals
        """
        # TODO: Integrate with existing DualXDATracer
        self.logger.warning("DualXDA signal computation not yet implemented")
        return {}
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.
        
        Returns:
            Dictionary containing evaluation summary
        """
        return {
            "mc_passes": self._state.get("mc_passes", 0),
            "uncertainty_signals": self._state.get("uncertainty_signals", []),
        }

# Made with Bob
