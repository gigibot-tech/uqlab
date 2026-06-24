"""MC Dropout uncertainty metrics from stacked softmax predictions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


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
    mean_pred = predictions.mean(dim=0)

    variance = predictions.var(dim=0)
    mean_variance = variance.mean(dim=1)

    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)

    expected_entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)
    mutual_info = entropy - expected_entropy

    return {
        "variance": variance,
        "mean_variance": mean_variance,
        "entropy": entropy,
        "mutual_info": mutual_info,
        "mean_prediction": mean_pred,
    }


def calculate_sirc_score(predictions: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Calculate SIRC (Softmax Information Retaining Combination) Score.

    SIRC combines MSP (Maximum Softmax Probability) with Entropy to handle both
    in-distribution errors and OOD samples.

    Mathematical formulation:
        SIRC(x) = -log(p_max) × H(p)
    """
    if predictions.dim() == 3:
        mean_pred = predictions.mean(dim=0)
    else:
        mean_pred = predictions

    p_max = mean_pred.max(dim=1)[0]
    neg_log_p_max = -torch.log(p_max + epsilon)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + epsilon), dim=1)

    return neg_log_p_max * entropy


def calculate_aleatoric_epistemic_split(predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Decompose uncertainty into aleatoric and epistemic components.

    Args:
        predictions: Stacked predictions [n_passes, batch_size, num_classes]

    Returns:
        Dictionary with aleatoric and epistemic estimates
    """
    aleatoric = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=2).mean(dim=0)

    mean_pred = predictions.mean(dim=0)
    epistemic = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)

    total = aleatoric + epistemic

    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": total,
    }


def get_top_uncertain_samples(
    uncertainty_scores: torch.Tensor,
    n_samples: int,
    indices: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    top_scores, top_idx = torch.topk(uncertainty_scores, n_samples)

    if indices is not None:
        selected_indices = indices[top_idx]
    else:
        selected_indices = top_idx

    return selected_indices, top_scores
