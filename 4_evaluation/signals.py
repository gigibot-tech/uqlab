"""
Uncertainty Signals - Computation of uncertainty signals.

This module provides:
- Probabilistic signals (entropy, mutual information)
- Attribution-based signals (coherence)
- Logit-based signals (dominance, mass, magnitude)
- Signal aggregation and analysis
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from shared.types import FloatArray, SignalDict, SIGNAL_NAMES
from shared.utils import to_numpy


class SignalCalculator:
    """
    Calculator for uncertainty signals.
    
    Provides methods for computing various uncertainty signals:
    - Probabilistic: entropy, mutual information, variance
    - Attribution: coherence
    - Logit-based: dominance, mass, magnitude
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize signal calculator.
        
        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
    
    # ========================================================================
    # Probabilistic Signals
    # ========================================================================
    
    def calculate_msp_uncertainty(self, probabilities: FloatArray) -> FloatArray:
        """
        Calculate Maximum Softmax Probability (MSP) uncertainty.
        
        MSP uncertainty = 1 - max(p)
        
        Args:
            probabilities: Predicted probabilities (N, C)
        
        Returns:
            MSP uncertainty scores (N,)
        """
        max_probs = probabilities.max(axis=1)
        return 1.0 - max_probs
    
    def calculate_predictive_entropy(self, probabilities: FloatArray) -> FloatArray:
        """
        Calculate predictive entropy.
        
        H[y|x] = -sum(p * log(p))
        
        Args:
            probabilities: Predicted probabilities (N, C)
        
        Returns:
            Entropy scores (N,)
        """
        # Clip to avoid log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return entropy
    
    def calculate_mutual_information(
        self,
        mc_probabilities: FloatArray,
    ) -> FloatArray:
        """
        Calculate mutual information from MC dropout samples.
        
        I[y;θ|x] = H[E[p]] - E[H[p]]
        
        Args:
            mc_probabilities: MC dropout probabilities (N, MC, C)
        
        Returns:
            Mutual information scores (N,)
        """
        # Mean probability across MC samples
        mean_probs = mc_probabilities.mean(axis=1)  # (N, C)
        
        # Entropy of mean
        entropy_of_mean = self.calculate_predictive_entropy(mean_probs)
        
        # Mean of entropies
        entropies = np.array([
            self.calculate_predictive_entropy(mc_probabilities[:, i, :])
            for i in range(mc_probabilities.shape[1])
        ])
        mean_of_entropies = entropies.mean(axis=0)
        
        # Mutual information
        mi = entropy_of_mean - mean_of_entropies
        return np.maximum(mi, 0.0)  # Ensure non-negative
    
    def calculate_predictive_variance(
        self,
        mc_probabilities: FloatArray,
    ) -> FloatArray:
        """
        Calculate predictive variance from MC dropout samples.
        
        Args:
            mc_probabilities: MC dropout probabilities (N, MC, C)
        
        Returns:
            Variance scores (N,)
        """
        # Variance across MC samples for predicted class
        predicted_class = mc_probabilities.mean(axis=1).argmax(axis=1)
        
        variances = []
        for i, pred_class in enumerate(predicted_class):
            class_probs = mc_probabilities[i, :, pred_class]
            variances.append(class_probs.var())
        
        return np.array(variances)
    
    # ========================================================================
    # Attribution-based Signals
    # ========================================================================
    
    def calculate_coherence(
        self,
        attributions: FloatArray,
        predictions: NDArray[np.int64],
    ) -> FloatArray:
        """
        Calculate coherence from attribution maps.
        
        Coherence measures how much attribution is concentrated on the
        predicted class vs. other classes.
        
        Args:
            attributions: Attribution scores (N, C)
            predictions: Predicted class indices (N,)
        
        Returns:
            Coherence scores (N,)
        """
        n_samples = len(predictions)
        coherence = np.zeros(n_samples)
        
        for i in range(n_samples):
            pred_class = predictions[i]
            pred_attr = attributions[i, pred_class]
            
            # Sum of attributions for other classes
            other_attr = np.sum(attributions[i]) - pred_attr
            
            # Coherence: ratio of predicted class attribution to total
            total_attr = np.sum(np.abs(attributions[i]))
            if total_attr > 0:
                coherence[i] = np.abs(pred_attr) / total_attr
            else:
                coherence[i] = 0.0
        
        return coherence
    
    def calculate_inverse_coherence(
        self,
        attributions: FloatArray,
        predictions: NDArray[np.int64],
    ) -> FloatArray:
        """
        Calculate inverse coherence (uncertainty signal).
        
        Args:
            attributions: Attribution scores (N, C)
            predictions: Predicted class indices (N,)
        
        Returns:
            Inverse coherence scores (N,)
        """
        coherence = self.calculate_coherence(attributions, predictions)
        return 1.0 - coherence
    
    # ========================================================================
    # Logit-based Signals
    # ========================================================================
    
    def calculate_dominance(self, logits: FloatArray) -> FloatArray:
        """
        Calculate dominance signal.
        
        Dominance = max(logit) - second_max(logit)
        Higher dominance = lower uncertainty
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Dominance scores (N,)
        """
        # Sort logits in descending order
        sorted_logits = np.sort(logits, axis=1)[:, ::-1]
        
        # Difference between top two logits
        dominance = sorted_logits[:, 0] - sorted_logits[:, 1]
        
        return dominance
    
    def calculate_inverse_dominance(self, logits: FloatArray) -> FloatArray:
        """
        Calculate inverse dominance (uncertainty signal).
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Inverse dominance scores (N,)
        """
        dominance = self.calculate_dominance(logits)
        # Normalize to [0, 1] range
        max_dom = dominance.max() if dominance.max() > 0 else 1.0
        return 1.0 - (dominance / max_dom)
    
    def calculate_mass(self, logits: FloatArray) -> FloatArray:
        """
        Calculate mass signal.
        
        Mass = sum(exp(logit)) = sum of unnormalized probabilities
        Higher mass = lower uncertainty
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Mass scores (N,)
        """
        # Sum of exponentials (unnormalized probabilities)
        mass = np.sum(np.exp(logits), axis=1)
        return mass
    
    def calculate_inverse_mass(self, logits: FloatArray) -> FloatArray:
        """
        Calculate inverse mass (uncertainty signal).
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Inverse mass scores (N,)
        """
        mass = self.calculate_mass(logits)
        # Normalize to [0, 1] range
        max_mass = mass.max() if mass.max() > 0 else 1.0
        return 1.0 - (mass / max_mass)
    
    def calculate_logit_magnitude(self, logits: FloatArray) -> FloatArray:
        """
        Calculate logit magnitude.
        
        Magnitude = ||logits||_2 = L2 norm of logit vector
        Higher magnitude = lower uncertainty
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Magnitude scores (N,)
        """
        magnitude = np.linalg.norm(logits, axis=1)
        return magnitude
    
    def calculate_inverse_logit_magnitude(self, logits: FloatArray) -> FloatArray:
        """
        Calculate inverse logit magnitude (uncertainty signal).
        
        Args:
            logits: Model logits (N, C)
        
        Returns:
            Inverse magnitude scores (N,)
        """
        magnitude = self.calculate_logit_magnitude(logits)
        # Normalize to [0, 1] range
        max_mag = magnitude.max() if magnitude.max() > 0 else 1.0
        return 1.0 - (magnitude / max_mag)
    
    # ========================================================================
    # Aggregation Methods
    # ========================================================================
    
    def calculate_all_signals(
        self,
        logits: Optional[FloatArray] = None,
        probabilities: Optional[FloatArray] = None,
        mc_probabilities: Optional[FloatArray] = None,
        attributions: Optional[FloatArray] = None,
        predictions: Optional[NDArray[np.int64]] = None,
    ) -> SignalDict:
        """
        Calculate all available uncertainty signals.
        
        Args:
            logits: Model logits (N, C)
            probabilities: Predicted probabilities (N, C)
            mc_probabilities: MC dropout probabilities (N, MC, C)
            attributions: Attribution scores (N, C)
            predictions: Predicted class indices (N,)
        
        Returns:
            Dictionary of all computed signals
        """
        signals = {}
        
        # Probabilistic signals
        if probabilities is not None:
            signals["msp_uncertainty"] = self.calculate_msp_uncertainty(probabilities)
            signals["predictive_entropy"] = self.calculate_predictive_entropy(probabilities)
        
        if mc_probabilities is not None:
            signals["mutual_info"] = self.calculate_mutual_information(mc_probabilities)
            signals["predictive_variance"] = self.calculate_predictive_variance(mc_probabilities)
        
        # Attribution-based signals
        if attributions is not None and predictions is not None:
            signals["coherence"] = self.calculate_coherence(attributions, predictions)
            signals["inverse_coherence"] = self.calculate_inverse_coherence(attributions, predictions)
        
        # Logit-based signals
        if logits is not None:
            signals["dominance"] = self.calculate_dominance(logits)
            signals["inverse_mass"] = self.calculate_inverse_mass(logits)
            signals["inverse_logit_magnitude"] = self.calculate_inverse_logit_magnitude(logits)
        
        return signals
    
    def normalize_signals(self, signals: SignalDict) -> SignalDict:
        """
        Normalize signals to [0, 1] range.
        
        Args:
            signals: Dictionary of signals
        
        Returns:
            Dictionary of normalized signals
        """
        normalized = {}
        
        for name, values in signals.items():
            min_val = values.min()
            max_val = values.max()
            
            if max_val > min_val:
                normalized[name] = (values - min_val) / (max_val - min_val)
            else:
                normalized[name] = np.zeros_like(values)
        
        return normalized


def get_top_signals(
    signals: SignalDict,
    auroc_scores: Dict[str, float],
    n: int = 4,
) -> List[str]:
    """
    Get top N signals by AUROC score.
    
    Args:
        signals: Dictionary of signals
        auroc_scores: Dictionary of AUROC scores for each signal
        n: Number of top signals to return
    
    Returns:
        List of top signal names
    """
    # Sort signals by AUROC
    sorted_signals = sorted(
        auroc_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    
    # Return top N signal names
    return [name for name, _ in sorted_signals[:n]]


def aggregate_signals(
    signals: SignalDict,
    weights: Optional[Dict[str, float]] = None,
) -> FloatArray:
    """
    Aggregate multiple signals into a single uncertainty score.
    
    Args:
        signals: Dictionary of signals
        weights: Optional weights for each signal (default: equal weights)
    
    Returns:
        Aggregated uncertainty scores
    """
    if not signals:
        raise ValueError("No signals provided")
    
    # Get signal names and values
    signal_names = list(signals.keys())
    signal_values = np.stack([signals[name] for name in signal_names], axis=1)
    
    # Set weights
    if weights is None:
        weights_array = np.ones(len(signal_names)) / len(signal_names)
    else:
        weights_array = np.array([weights.get(name, 1.0) for name in signal_names])
        weights_array = weights_array / weights_array.sum()
    
    # Weighted average
    aggregated = np.dot(signal_values, weights_array)
    
    return aggregated


# Made with Bob