"""
DualXDA Integrity Score Implementation
Calculates the Integrity Score (I) that measures data provenance quality.

The Integrity Score is designed to detect "Self-Saboteurs" - training samples
that create conflicting gradients due to label noise or ambiguous data.

Mathematical Foundation:
    I(x) = 1 - \\bar{H}(x) / \\log(K)

Where \\bar{H}(x) is the expected predictive entropy across MC dropout passes
and K is the number of classes.

This is a lightweight proxy for "evidence integrity" using only predictive
distributions (no explicit influence decomposition yet).
"""

import torch
import numpy as np
from typing import Dict, Tuple


class IntegrityScoreCalculator:
    """
    Calculate the Integrity Score for uncertainty quantification.
    
    The Integrity Score measures how "clean" the data provenance is for a sample.
    High integrity = Low aleatoric noise (clean training data)
    Low integrity = High aleatoric noise (conflicting training samples)
    """
    
    def __init__(self, epsilon=1e-6, off_diagonal_threshold=0.1):
        """
        Args:
            epsilon: Stability constant to prevent division by zero
            off_diagonal_threshold: Threshold for detecting significant confusion
        """
        self.epsilon = epsilon
        self.off_diagonal_threshold = off_diagonal_threshold
    
    def calculate_confusion_mass(self, predictions: torch.Tensor, 
                                 labels: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate off-diagonal mass from prediction distribution.
        
        For a sample, high off-diagonal mass indicates the model is uncertain
        between multiple classes, suggesting conflicting training data.
        
        Args:
            predictions: Prediction probabilities [batch_size, num_classes]
            labels: Optional ground truth labels [batch_size]
            
        Returns:
            off_diagonal_mass: Confusion measure [batch_size]
        """
        batch_size, num_classes = predictions.shape
        
        # Get top-2 predictions
        top2_probs, top2_indices = torch.topk(predictions, k=2, dim=1)
        
        # Off-diagonal mass: probability mass NOT on the top prediction
        # High value = model is confused between multiple classes
        off_diagonal_mass = 1.0 - top2_probs[:, 0]
        
        # Alternative: Use entropy as confusion measure
        # entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)
        # off_diagonal_mass = entropy / np.log(num_classes)  # Normalized
        
        return off_diagonal_mass
    
    def calculate_mc_confusion(self, mc_predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate confusion from MC Dropout predictions.
        
        This measures how much the predictions vary across different dropout masks,
        indicating conflicting information in the training data.
        
        Args:
            mc_predictions: MC predictions [n_passes, batch_size, num_classes]
            
        Returns:
            confusion: Confusion measure [batch_size]
        """
        # Calculate variance in predicted class across MC samples
        predicted_classes = mc_predictions.argmax(dim=2)  # [n_passes, batch_size]
        
        # Count unique predicted classes for each sample
        batch_size = mc_predictions.shape[1]
        confusion = torch.zeros(batch_size)
        
        for i in range(batch_size):
            unique_classes = torch.unique(predicted_classes[:, i])
            # Normalize by number of classes
            confusion[i] = len(unique_classes) / mc_predictions.shape[2]
        
        return confusion
    
    def calculate_integrity_score(self, mc_predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Integrity Score from MC Dropout predictions.
        
        Practical formulation (bounded, stable):
            I(x) = 1 - \\bar{H}(x) / \\log(K)

        where \bar{H}(x) is the expected predictive entropy across MC passes and
        K is the number of classes.

        Intuition (aligned with the thesis story):
        - High expected entropy means each dropout sample is itself "confused"
          (ambiguity/noise-like regime), so integrity should be low.
        - Low expected entropy means each dropout sample is sharp (even if they
          disagree), so integrity stays high (epistemic-like regime).
        
        This keeps I in [0, 1] and avoids the previous degenerate case where
        M_diag + M_off == 1 forced I to collapse to a rescaled confidence.
        
        High I → low per-pass confusion (cleaner evidence)
        Low I → high per-pass confusion (noisy/ambiguous evidence)
        
        Args:
            mc_predictions: MC predictions [n_passes, batch_size, num_classes]
            
        Returns:
            integrity_score: Integrity scores [batch_size] in range [0, 1]
        """
        if mc_predictions.dim() != 3:
            raise ValueError(
                f"mc_predictions must have shape [n_passes, batch_size, num_classes], got {tuple(mc_predictions.shape)}"
            )

        n_classes = mc_predictions.shape[2]
        if n_classes <= 1:
            # Degenerate single-class case: treat as fully "integrity" since no ambiguity exists.
            return torch.ones(mc_predictions.shape[1], device=mc_predictions.device)

        # Expected entropy across MC passes: E_t[ H(p_t(y|x)) ].
        # This is a standard proxy for aleatoric-like uncertainty under MC dropout.
        expected_entropy = -torch.sum(
            mc_predictions * torch.log(mc_predictions + 1e-10), dim=2
        ).mean(dim=0)  # [batch_size]

        # Normalize to [0, 1] using the maximum entropy log(K).
        max_entropy = torch.log(torch.tensor(float(n_classes), device=mc_predictions.device))
        expected_entropy_norm = expected_entropy / (max_entropy + self.epsilon)

        integrity_score = 1.0 - expected_entropy_norm
        return integrity_score.clamp(0.0, 1.0)
    
    def detect_self_saboteurs(self, integrity_scores: torch.Tensor) -> torch.Tensor:
        """
        Detect samples with low integrity (self-saboteurs).
        
        These are samples where the training data contains conflicting information,
        making them unreliable for uncertainty quantification.
        
        Args:
            integrity_scores: Integrity scores [batch_size]
            
        Returns:
            is_saboteur: Boolean mask [batch_size]
        """
        # Samples with integrity below threshold are self-saboteurs
        is_saboteur = integrity_scores < self.off_diagonal_threshold
        return is_saboteur


def calculate_integrity_batch(model, dataloader, n_passes=50, device='cuda'):
    """
    Calculate integrity scores for entire dataset.
    
    Args:
        model: MC Dropout model
        dataloader: Data loader
        n_passes: Number of MC forward passes
        device: Device to run on
        
    Returns:
        all_integrity_scores: Integrity scores for all samples
        all_labels: Ground truth labels
    """
    calculator = IntegrityScoreCalculator()
    model.eval()
    
    all_integrity_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            
            # MC Dropout forward passes
            mc_predictions = model.mc_forward(data, n_passes=n_passes)
            
            # Calculate integrity scores
            integrity_scores = calculator.calculate_integrity_score(mc_predictions)
            
            all_integrity_scores.append(integrity_scores.cpu())
            all_labels.append(target)
    
    return {
        'integrity_scores': torch.cat(all_integrity_scores),
        'labels': torch.cat(all_labels)
    }


if __name__ == "__main__":
    # Test integrity score calculation
    n_passes = 50
    batch_size = 32
    num_classes = 10
    
    calculator = IntegrityScoreCalculator()
    
    # Simulate clean predictions (high integrity)
    clean_predictions = torch.zeros(n_passes, batch_size, num_classes)
    clean_predictions[:, :, 0] = 0.9  # Confident in class 0
    clean_predictions[:, :, 1:] = 0.1 / (num_classes - 1)
    clean_predictions = torch.softmax(clean_predictions, dim=2)
    
    clean_integrity = calculator.calculate_integrity_score(clean_predictions)
    print(f"Clean predictions - Mean Integrity: {clean_integrity.mean():.4f}")
    
    # Simulate noisy predictions (low integrity)
    noisy_predictions = torch.softmax(torch.randn(n_passes, batch_size, num_classes), dim=2)
    noisy_integrity = calculator.calculate_integrity_score(noisy_predictions)
    print(f"Noisy predictions - Mean Integrity: {noisy_integrity.mean():.4f}")
    
    # Detect self-saboteurs
    saboteurs = calculator.detect_self_saboteurs(noisy_integrity)
    print(f"Self-saboteurs detected: {saboteurs.sum().item()} / {batch_size}")

# Made with Bob
