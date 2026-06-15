"""
MC Dropout Uncertainty Quantification
Implements variance-based uncertainty estimation using Monte Carlo Dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


def calculate_mc_dropout_uncertainty(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate uncertainty metrics from MC Dropout predictions.
    
    This is the BASELINE metric that measures Total Uncertainty (Epistemic + Aleatoric).
    
    Mathematical formulation:
        U_total = Var[p(y|x)] = E[Var[p(y|x,θ)]] + Var[E[p(y|x,θ)]]
                = Aleatoric + Epistemic
    
    Args:
        predictions: Stacked predictions [n_passes, batch_size, num_classes]
        
    Returns:
        Dictionary containing:
            - variance: Predictive variance [batch_size, num_classes]
            - mean_variance: Mean variance across classes [batch_size]
            - entropy: Predictive entropy [batch_size]
            - mutual_info: Mutual information (epistemic proxy) [batch_size]
    """
    # Mean prediction across MC samples
    mean_pred = predictions.mean(dim=0)  # [batch_size, num_classes]
    
    # Predictive variance (Total Uncertainty)
    variance = predictions.var(dim=0)  # [batch_size, num_classes]
    mean_variance = variance.mean(dim=1)  # [batch_size]
    
    # Predictive entropy (Total Uncertainty)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)  # [batch_size]
    
    # Mutual Information (Epistemic Uncertainty proxy)
    # MI = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]
    expected_entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)
    mutual_info = entropy - expected_entropy  # [batch_size]
    
    return {
        'variance': variance,
        'mean_variance': mean_variance,
        'entropy': entropy,
        'mutual_info': mutual_info,
        'mean_prediction': mean_pred
    }


@torch.no_grad()
def _mc_forward_one_chunk(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int,
) -> torch.Tensor:
    """
    MC passes on one sample chunk [B, ...] → [T, B, C].

    CNN/ResNet: backbone features computed once per chunk; only dropout+head repeat.
    MLP: T lightweight forwards (dropout is the main cost).
    """
    if hasattr(model, "enable_dropout"):
        model.eval()
        model.enable_dropout()

    if hasattr(model, "extract_features") and hasattr(model, "classifier"):
        features = model.extract_features(x)
        dropout = getattr(model, "dropout", None)
        if dropout is None:
            raise AttributeError("Model has extract_features but no dropout module")
        preds = []
        for _ in range(n_passes):
            h = dropout(features)
            logits = model.classifier(h)
            preds.append(F.softmax(logits, dim=1))
        return torch.stack(preds, dim=0)

    preds = []
    for _ in range(n_passes):
        logits = model(x)
        preds.append(F.softmax(logits, dim=1))
    return torch.stack(preds, dim=0)


@torch.no_grad()
def mc_forward_efficient(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int,
    *,
    sample_batch_size: int = 256,
) -> torch.Tensor:
    """
    Batched MC Dropout over eval samples.

    Chunks the eval set along batch dimension to limit memory; reuses CNN/ResNet
    backbone features within each chunk (see :func:`_mc_forward_one_chunk`).
    """
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("empty eval tensor")
    chunks: list[torch.Tensor] = []
    for start in range(0, n, sample_batch_size):
        end = min(start + sample_batch_size, n)
        chunks.append(_mc_forward_one_chunk(model, x[start:end], n_passes))
    return torch.cat(chunks, dim=1)


def calculate_sirc_score(predictions: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Calculate SIRC (Softmax Information Retaining Combination) Score.
    
    SIRC is a SOTA uncertainty metric that combines MSP (Maximum Softmax Probability)
    with Entropy to handle both in-distribution errors and OOD samples.
    
    Mathematical formulation:
        SIRC(x) = -log(p_max) × H(p)
        
    Where:
        p_max = max_c p(y=c|x) (confidence in top prediction)
        H(p) = -Σ p(y=c|x) log p(y=c|x) (prediction entropy)
        
    Intuition:
        - Low confidence (low p_max) → high -log(p_max) → high uncertainty
        - High entropy (uniform distribution) → high H(p) → high uncertainty
        - Product captures both: confident but wrong (low H, high -log p_max)
          AND uncertain predictions (high H)
          
    Properties:
        - Handles in-distribution errors (via MSP component)
        - Handles OOD samples (via entropy component)
        - Better than MSP or Entropy alone
        
    Reference:
        "SIRC: Softmax Information Retaining Combination for Uncertainty Estimation"
        Recent SOTA for selective prediction
        
    Args:
        predictions: Mean predictions [batch_size, num_classes] or
                    Stacked predictions [n_passes, batch_size, num_classes]
        epsilon: Small constant for numerical stability
        
    Returns:
        sirc_score: [batch_size] uncertainty scores (higher = more uncertain)
    """
    # Handle both mean predictions and stacked MC predictions
    if predictions.dim() == 3:
        # Take mean across MC passes
        mean_pred = predictions.mean(dim=0)  # [batch_size, num_classes]
    else:
        mean_pred = predictions  # Already [batch_size, num_classes]
    
    # Component 1: MSP (Maximum Softmax Probability)
    # Confidence in the top prediction
    p_max = mean_pred.max(dim=1)[0]  # [batch_size]
    
    # Convert to uncertainty: -log(p_max)
    # High confidence (p_max ≈ 1) → low uncertainty (-log(1) = 0)
    # Low confidence (p_max ≈ 0) → high uncertainty (-log(0) → ∞)
    neg_log_p_max = -torch.log(p_max + epsilon)  # [batch_size]
    
    # Component 2: Entropy
    # Measures spread of probability distribution
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=1)  # [batch_size]
    
    # SIRC: Combine both components via multiplication
    # This captures BOTH:
    # - Confident but potentially wrong predictions (high -log p_max, low entropy)
    # - Uncertain predictions (high entropy)
    sirc_score = neg_log_p_max * entropy  # [batch_size]
    
    return sirc_score



def calculate_aleatoric_epistemic_split(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Attempt to decompose uncertainty into aleatoric and epistemic components.
    
    This is the STANDARD approach but has limitations:
    - Assumes the decomposition is clean (it's not in practice)
    - Cannot filter aleatoric noise from epistemic signal
    
    Args:
        predictions: Stacked predictions [n_passes, batch_size, num_classes]
        
    Returns:
        Dictionary with aleatoric and epistemic estimates
    """
    # Expected data uncertainty (Aleatoric)
    # E[H[p(y|x,θ)]] - entropy of each prediction, then average
    aleatoric = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)
    
    # Model uncertainty (Epistemic)
    # H[E[p(y|x,θ)]] - entropy of mean prediction
    mean_pred = predictions.mean(dim=0)
    epistemic = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
    
    # Total = Aleatoric + Epistemic
    total = aleatoric + epistemic
    
    return {
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': total
    }


def get_top_uncertain_samples(uncertainty_scores: torch.Tensor, 
                              n_samples: int,
                              indices: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get indices of most uncertain samples for active learning.
    
    Args:
        uncertainty_scores: Uncertainty values [batch_size]
        n_samples: Number of samples to select
        indices: Optional original indices [batch_size]
        
    Returns:
        selected_indices: Indices of selected samples
        selected_scores: Uncertainty scores of selected samples
    """
    n_samples = min(n_samples, len(uncertainty_scores))
    
    # Get top-k uncertain samples
    top_scores, top_idx = torch.topk(uncertainty_scores, n_samples)
    
    if indices is not None:
        selected_indices = indices[top_idx]
    else:
        selected_indices = top_idx
    
    return selected_indices, top_scores


def batch_mc_dropout_uncertainty(model, dataloader, n_passes=50, device='cuda'):
    """
    Calculate MC Dropout uncertainty for entire dataset.
    
    Args:
        model: MC Dropout model
        dataloader: Data loader
        n_passes: Number of MC forward passes
        device: Device to run on
        
    Returns:
        all_uncertainties: Dictionary of uncertainty metrics
        all_labels: Ground truth labels
        all_predictions: Mean predictions
    """
    model.eval()
    
    all_mean_variance = []
    all_entropy = []
    all_mutual_info = []
    all_sirc_score = []
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            
            # MC Dropout forward passes
            mc_predictions = model.mc_forward(data, n_passes=n_passes)
            
            # Calculate uncertainties
            uncertainties = calculate_mc_dropout_uncertainty(mc_predictions)
            
            # Calculate SIRC Score (SOTA uncertainty)
            sirc = calculate_sirc_score(mc_predictions)
            
            all_mean_variance.append(uncertainties['mean_variance'].cpu())
            all_entropy.append(uncertainties['entropy'].cpu())
            all_mutual_info.append(uncertainties['mutual_info'].cpu())
            all_sirc_score.append(sirc.cpu())
            all_labels.append(target)
            all_predictions.append(uncertainties['mean_prediction'].cpu())
    
    return {
        'mean_variance': torch.cat(all_mean_variance),
        'entropy': torch.cat(all_entropy),
        'mutual_info': torch.cat(all_mutual_info),
        'sirc_score': torch.cat(all_sirc_score),
        'labels': torch.cat(all_labels),
        'predictions': torch.cat(all_predictions)
    }


if __name__ == "__main__":
    # Test uncertainty calculation
    n_passes = 50
    batch_size = 32
    num_classes = 10
    
    # Simulate MC Dropout predictions
    predictions = torch.softmax(torch.randn(n_passes, batch_size, num_classes), dim=2)
    
    # Calculate uncertainties
    uncertainties = calculate_mc_dropout_uncertainty(predictions)
    
    print("MC Dropout Uncertainty Metrics:")
    print(f"Mean Variance shape: {uncertainties['mean_variance'].shape}")
    print(f"Entropy shape: {uncertainties['entropy'].shape}")
    print(f"Mutual Info shape: {uncertainties['mutual_info'].shape}")
    
    print(f"\nMean Variance: {uncertainties['mean_variance'].mean():.4f}")
    print(f"Mean Entropy: {uncertainties['entropy'].mean():.4f}")
    print(f"Mean Mutual Info: {uncertainties['mutual_info'].mean():.4f}")
    
    # Test decomposition
    decomposition = calculate_aleatoric_epistemic_split(predictions)
    print(f"\nAleatoric: {decomposition['aleatoric'].mean():.4f}")
    print(f"Epistemic: {decomposition['epistemic'].mean():.4f}")
    print(f"Total: {decomposition['total'].mean():.4f}")

# Made with Bob
