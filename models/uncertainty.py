"""
Uncertainty Quantification - MC Dropout, Deep Ensembles, and uncertainty utilities.

This module consolidates uncertainty quantification functionality from:
- src/metrics/mc_dropout_uq.py (MC Dropout implementation)
- Various scattered uncertainty code

Provides:
- MC Dropout utilities
- Deep Ensemble implementation
- Uncertainty metric calculations
- Batch uncertainty estimation
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MC Dropout Utilities
# ============================================================================

def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    
    Args:
        model: PyTorch model with dropout layers
    
    Example:
        >>> model.eval()
        >>> enable_mc_dropout(model)
        >>> # Now dropout is active during inference
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_mc_dropout(model: nn.Module) -> None:
    """
    Disable dropout layers (standard inference mode).
    
    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.eval()


@torch.no_grad()
def mc_forward_pass(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20,
    enable_dropout: bool = True
) -> torch.Tensor:
    """
    Perform multiple forward passes with MC Dropout.
    
    Args:
        model: PyTorch model
        x: Input tensor [batch_size, ...]
        n_passes: Number of MC forward passes
        enable_dropout: Whether to enable dropout (True for MC Dropout)
    
    Returns:
        predictions: Stacked predictions [n_passes, batch_size, num_classes]
    
    Example:
        >>> model.eval()
        >>> predictions = mc_forward_pass(model, images, n_passes=20)
        >>> mean_pred = predictions.mean(dim=0)
        >>> uncertainty = predictions.var(dim=0).mean(dim=1)
    """
    if enable_dropout:
        enable_mc_dropout(model)
    
    predictions = []
    for _ in range(n_passes):
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        predictions.append(probs)
    
    return torch.stack(predictions, dim=0)


@torch.no_grad()
def mc_forward_efficient(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20,
    batch_size: int = 256
) -> torch.Tensor:
    """
    Memory-efficient MC Dropout with batching.
    
    For models with separate feature extraction and classification:
    - Extract features once per batch
    - Only repeat dropout + classifier head
    
    Args:
        model: PyTorch model
        x: Input tensor [N, ...]
        n_passes: Number of MC passes
        batch_size: Batch size for processing
    
    Returns:
        predictions: Stacked predictions [n_passes, N, num_classes]
    """
    enable_mc_dropout(model)
    
    n_samples = x.shape[0]
    all_predictions = []
    
    # Process in batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_x = x[start_idx:end_idx]
        
        # Check if model has feature extraction
        if hasattr(model, 'extract_features') and hasattr(model, 'head'):
            # Extract features once
            features = model.extract_features(batch_x)
            
            # Multiple passes through head only
            batch_preds = []
            for _ in range(n_passes):
                logits = model.head(features)
                probs = F.softmax(logits, dim=-1)
                batch_preds.append(probs)
            
            batch_preds = torch.stack(batch_preds, dim=0)
        else:
            # Full forward passes
            batch_preds = mc_forward_pass(model, batch_x, n_passes, enable_dropout=False)
        
        all_predictions.append(batch_preds)
    
    return torch.cat(all_predictions, dim=1)


# ============================================================================
# Uncertainty Metrics
# ============================================================================

def calculate_predictive_entropy(predictions: torch.Tensor) -> torch.Tensor:
    """
    Calculate predictive entropy (total uncertainty).
    
    Args:
        predictions: Predictions [batch_size, num_classes] or
                    MC predictions [n_passes, batch_size, num_classes]
    
    Returns:
        entropy: Predictive entropy [batch_size]
    """
    if predictions.dim() == 3:
        # Take mean across MC passes
        mean_pred = predictions.mean(dim=0)
    else:
        mean_pred = predictions
    
    entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
    return entropy


def calculate_mutual_information(predictions: torch.Tensor) -> torch.Tensor:
    """
    Calculate mutual information (epistemic uncertainty proxy).
    
    MI = H[E[p(y|x,θ)]] - E[H[p(y|x,θ)]]
    
    Args:
        predictions: MC predictions [n_passes, batch_size, num_classes]
    
    Returns:
        mutual_info: Mutual information [batch_size]
    """
    if predictions.dim() != 3:
        raise ValueError("Mutual information requires MC predictions [n_passes, batch_size, num_classes]")
    
    # Predictive entropy
    mean_pred = predictions.mean(dim=0)
    pred_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
    
    # Expected entropy
    expected_entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1).mean(dim=0)
    
    # Mutual information
    mutual_info = pred_entropy - expected_entropy
    return mutual_info


def calculate_predictive_variance(predictions: torch.Tensor) -> torch.Tensor:
    """
    Calculate predictive variance (total uncertainty).
    
    Args:
        predictions: MC predictions [n_passes, batch_size, num_classes]
    
    Returns:
        variance: Mean variance across classes [batch_size]
    """
    if predictions.dim() != 3:
        raise ValueError("Variance requires MC predictions [n_passes, batch_size, num_classes]")
    
    variance = predictions.var(dim=0).mean(dim=-1)
    return variance


def calculate_mc_uncertainty(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Calculate comprehensive uncertainty metrics from MC Dropout predictions.
    
    Args:
        predictions: MC predictions [n_passes, batch_size, num_classes]
    
    Returns:
        Dictionary containing:
            - mean_prediction: Mean prediction [batch_size, num_classes]
            - variance: Predictive variance [batch_size]
            - entropy: Predictive entropy [batch_size]
            - mutual_info: Mutual information [batch_size]
    
    Example:
        >>> mc_preds = mc_forward_pass(model, images, n_passes=20)
        >>> uncertainty = calculate_mc_uncertainty(mc_preds)
        >>> print(f"Mean entropy: {uncertainty['entropy'].mean():.4f}")
    """
    mean_pred = predictions.mean(dim=0)
    
    return {
        'mean_prediction': mean_pred,
        'variance': calculate_predictive_variance(predictions),
        'entropy': calculate_predictive_entropy(predictions),
        'mutual_info': calculate_mutual_information(predictions),
    }


def calculate_msp_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    """
    Calculate Maximum Softmax Probability (MSP) uncertainty.
    
    MSP uncertainty = 1 - max(softmax(logits))
    
    Args:
        predictions: Predictions [batch_size, num_classes]
    
    Returns:
        uncertainty: MSP uncertainty [batch_size]
    """
    if predictions.dim() == 3:
        predictions = predictions.mean(dim=0)
    
    max_probs = predictions.max(dim=-1)[0]
    uncertainty = 1.0 - max_probs
    return uncertainty


# ============================================================================
# Deep Ensemble
# ============================================================================

class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification.
    
    Trains multiple models with different initializations and aggregates
    their predictions for uncertainty estimation.
    
    Args:
        models: List of trained models
        aggregation: Aggregation method ('mean', 'vote')
    
    Example:
        >>> models = [create_model() for _ in range(5)]
        >>> # Train each model independently
        >>> ensemble = DeepEnsemble(models)
        >>> predictions, uncertainty = ensemble.predict_with_uncertainty(images)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = 'mean'
    ):
        super().__init__()
        
        if len(models) == 0:
            raise ValueError("DeepEnsemble requires at least one model")
        
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.n_models = len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble (mean prediction).
        
        Args:
            x: Input tensor [batch_size, ...]
        
        Returns:
            logits: Mean logits [batch_size, num_classes]
        """
        all_logits = []
        for model in self.models:
            logits = model(x)
            all_logits.append(logits)
        
        # Stack and average
        all_logits = torch.stack(all_logits, dim=0)
        mean_logits = all_logits.mean(dim=0)
        
        return mean_logits
    
    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble uncertainty.
        
        Args:
            x: Input tensor [batch_size, ...]
        
        Returns:
            predictions: Mean softmax probabilities [batch_size, num_classes]
            uncertainty: Predictive entropy [batch_size]
        """
        all_probs = []
        
        for model in self.models:
            model.eval()
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        # Stack predictions: [n_models, batch_size, num_classes]
        all_probs = torch.stack(all_probs, dim=0)
        
        # Mean prediction
        mean_pred = all_probs.mean(dim=0)
        
        # Predictive entropy (uncertainty)
        entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
        
        return mean_pred, entropy
    
    def get_all_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all ensemble members.
        
        Args:
            x: Input tensor [batch_size, ...]
        
        Returns:
            predictions: All predictions [n_models, batch_size, num_classes]
        """
        all_probs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)
        
        return torch.stack(all_probs, dim=0)


# ============================================================================
# Batch Uncertainty Estimation
# ============================================================================

@torch.no_grad()
def batch_uncertainty_estimation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    method: str = 'mc_dropout',
    n_passes: int = 20,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Calculate uncertainty for entire dataset.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        method: Uncertainty method ('mc_dropout', 'msp', 'ensemble')
        n_passes: Number of MC passes (for MC Dropout)
        device: Device to run on
    
    Returns:
        Dictionary containing:
            - predictions: Mean predictions [N, num_classes]
            - uncertainty: Uncertainty scores [N]
            - labels: Ground truth labels [N]
            - (additional metrics depending on method)
    
    Example:
        >>> results = batch_uncertainty_estimation(
        ...     model, test_loader, method='mc_dropout', n_passes=20
        ... )
        >>> print(f"Mean uncertainty: {results['uncertainty'].mean():.4f}")
    """
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    if method == 'mc_dropout':
        all_variance = []
        all_mutual_info = []
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            # MC forward passes
            mc_preds = mc_forward_pass(model, batch_x, n_passes)
            
            # Calculate uncertainties
            uncertainty_metrics = calculate_mc_uncertainty(mc_preds)
            
            all_predictions.append(uncertainty_metrics['mean_prediction'].cpu())
            all_uncertainties.append(uncertainty_metrics['entropy'].cpu())
            all_variance.append(uncertainty_metrics['variance'].cpu())
            all_mutual_info.append(uncertainty_metrics['mutual_info'].cpu())
            all_labels.append(batch_y)
        
        return {
            'predictions': torch.cat(all_predictions, dim=0),
            'uncertainty': torch.cat(all_uncertainties, dim=0),
            'variance': torch.cat(all_variance, dim=0),
            'mutual_info': torch.cat(all_mutual_info, dim=0),
            'labels': torch.cat(all_labels, dim=0),
        }
    
    elif method == 'msp':
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            logits = model(batch_x)
            probs = F.softmax(logits, dim=-1)
            uncertainty = calculate_msp_uncertainty(probs)
            
            all_predictions.append(probs.cpu())
            all_uncertainties.append(uncertainty.cpu())
            all_labels.append(batch_y)
        
        return {
            'predictions': torch.cat(all_predictions, dim=0),
            'uncertainty': torch.cat(all_uncertainties, dim=0),
            'labels': torch.cat(all_labels, dim=0),
        }
    
    elif method == 'ensemble':
        if not isinstance(model, DeepEnsemble):
            raise TypeError("Method 'ensemble' requires a DeepEnsemble model")
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            predictions, uncertainty = model.predict_with_uncertainty(batch_x)
            
            all_predictions.append(predictions.cpu())
            all_uncertainties.append(uncertainty.cpu())
            all_labels.append(batch_y)
        
        return {
            'predictions': torch.cat(all_predictions, dim=0),
            'uncertainty': torch.cat(all_uncertainties, dim=0),
            'labels': torch.cat(all_labels, dim=0),
        }
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from ['mc_dropout', 'msp', 'ensemble']")


# Made with Bob