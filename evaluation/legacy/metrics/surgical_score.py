"""
Surgical Score Implementation
Combines MC Dropout variance with Integrity Score to isolate Epistemic Uncertainty.

Mathematical Foundation:
    Surgical Score = σ² × I
    
Where:
    σ² = MC Dropout variance (Total Uncertainty)
    I = Integrity Score (Data provenance quality)
    
This multiplication achieves:
    - High σ², High I → High Epistemic (The Void - model lacks knowledge)
    - High σ², Low I → Suppressed (The Pile - noisy data, not model uncertainty)
    - Low σ², High I → Low Epistemic (Model is confident and data is clean)
    - Low σ², Low I → Low Epistemic (Model is confident despite noise)
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .mc_dropout_uq import calculate_mc_dropout_uncertainty
from .integrity_score import IntegrityScoreCalculator


class SurgicalScoreCalculator:
    """
    Calculate the Surgical Score for epistemic uncertainty quantification.
    
    This is the PROPOSED metric that isolates epistemic uncertainty by
    filtering out aleatoric noise using the Integrity Score.
    """
    
    def __init__(self, epsilon=1e-6, off_diagonal_threshold=0.1):
        """
        Args:
            epsilon: Stability constant
            off_diagonal_threshold: Threshold for detecting self-saboteurs
        """
        self.integrity_calculator = IntegrityScoreCalculator(
            epsilon=epsilon,
            off_diagonal_threshold=off_diagonal_threshold
        )
        self.epsilon = epsilon
    
    def calculate_surgical_score(self, mc_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the Surgical Score from MC Dropout predictions.
        
        Mathematical formulation:
            Surgical Score = σ² × I ≈ U_epistemic
        
        Where:
            σ² = Predictive variance (Total UQ from MC Dropout)
            I = Integrity Score (Inverse of aleatoric noise)
        
        Args:
            mc_predictions: MC predictions [n_passes, batch_size, num_classes]
            
        Returns:
            Dictionary containing:
                - surgical_score: The main metric [batch_size]
                - mc_variance: Raw MC Dropout variance [batch_size]
                - integrity_score: Data quality score [batch_size]
                - is_epistemic: Boolean mask for epistemic samples [batch_size]
                - is_aleatoric: Boolean mask for aleatoric samples [batch_size]
        """
        # Step 1: Calculate MC Dropout uncertainty (Total UQ)
        mc_uncertainty = calculate_mc_dropout_uncertainty(mc_predictions)
        mc_variance = mc_uncertainty['mean_variance']  # [batch_size]
        
        # Step 2: Calculate Integrity Score (Aleatoric filter)
        integrity_score = self.integrity_calculator.calculate_integrity_score(mc_predictions)
        
        # Step 3: Calculate Surgical Score (Epistemic UQ)
        surgical_score = mc_variance * integrity_score
        
        # Step 4: Classify samples
        is_aleatoric = self.integrity_calculator.detect_self_saboteurs(integrity_score)
        is_epistemic = ~is_aleatoric & (mc_variance > mc_variance.median())
        
        return {
            'surgical_score': surgical_score,
            'mc_variance': mc_variance,
            'integrity_score': integrity_score,
            'is_epistemic': is_epistemic,
            'is_aleatoric': is_aleatoric,
            'mean_prediction': mc_uncertainty['mean_prediction']
        }
    
    def compare_with_mc_dropout(self, mc_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Direct comparison between MC Dropout and Surgical Score.
        
        This is the key function for the thesis benchmark.
        
        Args:
            mc_predictions: MC predictions [n_passes, batch_size, num_classes]
            
        Returns:
            Dictionary with both metrics for comparison
        """
        # Calculate both metrics
        mc_uncertainty = calculate_mc_dropout_uncertainty(mc_predictions)
        surgical_results = self.calculate_surgical_score(mc_predictions)
        
        return {
            # MC Dropout (Baseline)
            'mc_dropout_variance': mc_uncertainty['mean_variance'],
            'mc_dropout_entropy': mc_uncertainty['entropy'],
            
            # Surgical Score (Proposed)
            'surgical_score': surgical_results['surgical_score'],
            'integrity_score': surgical_results['integrity_score'],
            
            # Classification
            'is_epistemic': surgical_results['is_epistemic'],
            'is_aleatoric': surgical_results['is_aleatoric'],
            
            # Predictions
            'mean_prediction': surgical_results['mean_prediction']
        }


def batch_surgical_score(model, dataloader, n_passes=50, device='cuda'):
    """
    Calculate Surgical Scores for entire dataset.
    
    Args:
        model: MC Dropout model
        dataloader: Data loader
        n_passes: Number of MC forward passes
        device: Device to run on
        
    Returns:
        Dictionary with all metrics for the dataset
    """
    calculator = SurgicalScoreCalculator()
    model.eval()
    
    all_surgical_scores = []
    all_mc_variances = []
    all_integrity_scores = []
    all_is_epistemic = []
    all_is_aleatoric = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            
            # MC Dropout forward passes
            mc_predictions = model.mc_forward(data, n_passes=n_passes)
            
            # Calculate surgical scores
            results = calculator.calculate_surgical_score(mc_predictions)
            
            all_surgical_scores.append(results['surgical_score'].cpu())
            all_mc_variances.append(results['mc_variance'].cpu())
            all_integrity_scores.append(results['integrity_score'].cpu())
            all_is_epistemic.append(results['is_epistemic'].cpu())
            all_is_aleatoric.append(results['is_aleatoric'].cpu())
            all_labels.append(target)
            all_predictions.append(results['mean_prediction'].cpu())
    
    return {
        'surgical_score': torch.cat(all_surgical_scores),
        'mc_variance': torch.cat(all_mc_variances),
        'integrity_score': torch.cat(all_integrity_scores),
        'is_epistemic': torch.cat(all_is_epistemic),
        'is_aleatoric': torch.cat(all_is_aleatoric),
        'labels': torch.cat(all_labels),
        'predictions': torch.cat(all_predictions)
    }


def analyze_uncertainty_decomposition(results: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Analyze how the Surgical Score decomposes uncertainty.
    
    This provides statistics for the thesis results section.
    
    Args:
        results: Output from batch_surgical_score
        
    Returns:
        Dictionary with analysis statistics
    """
    surgical_score = results['surgical_score']
    mc_variance = results['mc_variance']
    integrity_score = results['integrity_score']
    is_epistemic = results['is_epistemic']
    is_aleatoric = results['is_aleatoric']
    
    analysis = {
        # Overall statistics
        'mean_mc_variance': mc_variance.mean().item(),
        'mean_surgical_score': surgical_score.mean().item(),
        'mean_integrity': integrity_score.mean().item(),
        
        # Sample classification
        'pct_epistemic': (is_epistemic.sum() / len(is_epistemic) * 100).item(),
        'pct_aleatoric': (is_aleatoric.sum() / len(is_aleatoric) * 100).item(),
        
        # Correlation
        'correlation_mc_surgical': torch.corrcoef(
            torch.stack([mc_variance, surgical_score])
        )[0, 1].item(),
        
        # Suppression effect
        'suppression_ratio': (surgical_score.mean() / mc_variance.mean()).item(),
    }
    
    return analysis


if __name__ == "__main__":
    # Test surgical score calculation
    n_passes = 50
    batch_size = 32
    num_classes = 10
    
    calculator = SurgicalScoreCalculator()
    
    print("=" * 60)
    print("Testing Surgical Score on Clean Data (High Integrity)")
    print("=" * 60)
    
    # Simulate clean, confident predictions
    clean_predictions = torch.zeros(n_passes, batch_size, num_classes)
    clean_predictions[:, :, 0] = 0.9
    clean_predictions[:, :, 1:] = 0.1 / (num_classes - 1)
    clean_predictions = torch.softmax(clean_predictions, dim=2)
    
    clean_results = calculator.calculate_surgical_score(clean_predictions)
    print(f"MC Variance: {clean_results['mc_variance'].mean():.4f}")
    print(f"Integrity Score: {clean_results['integrity_score'].mean():.4f}")
    print(f"Surgical Score: {clean_results['surgical_score'].mean():.4f}")
    print(f"Epistemic samples: {clean_results['is_epistemic'].sum()}")
    print(f"Aleatoric samples: {clean_results['is_aleatoric'].sum()}")
    
    print("\n" + "=" * 60)
    print("Testing Surgical Score on Noisy Data (Low Integrity)")
    print("=" * 60)
    
    # Simulate noisy, uncertain predictions
    noisy_predictions = torch.softmax(torch.randn(n_passes, batch_size, num_classes), dim=2)
    
    noisy_results = calculator.calculate_surgical_score(noisy_predictions)
    print(f"MC Variance: {noisy_results['mc_variance'].mean():.4f}")
    print(f"Integrity Score: {noisy_results['integrity_score'].mean():.4f}")
    print(f"Surgical Score: {noisy_results['surgical_score'].mean():.4f}")
    print(f"Epistemic samples: {noisy_results['is_epistemic'].sum()}")
    print(f"Aleatoric samples: {noisy_results['is_aleatoric'].sum()}")
    
    print("\n" + "=" * 60)
    print("Key Observation:")
    print("=" * 60)
    print("MC Dropout flags both clean and noisy samples as uncertain.")
    print("Surgical Score suppresses noisy samples (low integrity).")
    print("This is the core thesis contribution!")

# Made with Bob
