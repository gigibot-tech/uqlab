"""
Attribution-based uncertainty signal computation.

Computes various signals from DualXDA attribution traces to distinguish
between aleatoric and epistemic uncertainty.

ARCHITECTURE-AGNOSTIC DESIGN:
------------------------------
This module works with ANY model architecture by operating on the model's
internal representations (embeddings, features, or raw inputs):
- DINOv2: Pre-extracted embeddings [N, D]
- CNN/ResNet: Raw images [N, C, H, W] or intermediate features
- Custom models: Any tensor the model can process

DualXDA automatically hooks into the classifier layer (detected via
`infer_classifier_layer_name()`) and computes attributions on whatever
representation reaches that layer.

STANDARD DualXDA METRICS:
- Mass: Sum of absolute attribution magnitudes
- Coherence: Ratio of signed sum to absolute sum
- Dominance: Maximum attribution relative to total mass

SEMANTIC SIGNALS (Aleatoric indicators):
- Label disagreement: Entropy of supporter labels
- Noisy support ratio: Fraction of supporters with noisy labels
- Cross-class support: Fraction of supporters from different classes

STRUCTURAL SIGNALS (Epistemic indicators):
- Attribution concentration: Gini coefficient of attribution magnitudes

MATHEMATICAL FOUNDATION - Representer Theorem:
-----------------------------------------------
For linear classifiers with dual formulation:
    f(x) = Σ_i α_i y_i ⟨x_i, x⟩ + b

DualXDA attribution mass = Σ_i |α_i y_i ⟨x_i, x⟩| ≈ |f(x)| = |logit|

This means:
- High mass ≈ High |logit| → High confidence
- Low mass ≈ Low |logit| → Low confidence → Epistemic uncertainty
- inverse_mass ≈ 1/|logit| ≈ inverse_msp (for max logit)

The attribution-based signals provide interpretable, sample-level explanations
while being mathematically equivalent to logit-based confidence measures.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from uqlab.triage.dualxda_axioms import DualXDATracer


def topk_influence_metrics(
    row_attr: torch.Tensor,
    top_k: int,
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """
    Coherence, mass, and dominance on the top-k training influences by |T_i|.

    Full-training-set mass/dominance saturate (mass huge → ``inverse_mass`` ~ 0;
    dominance ~ 0). Top-k matches the supporter window used for label-based signals.
    """
    flat = row_attr.flatten()
    if flat.numel() == 0:
        return 0.0, 0.0, 0.0
    k = min(top_k, int(flat.numel()))
    _, idx = torch.topk(flat.abs(), k)
    top_attr = flat[idx]
    mass_k = float(top_attr.abs().sum().item())
    if mass_k <= 0.0:
        return 0.0, 0.0, 0.0
    signed_k = float(top_attr.sum().item())
    coherence = abs(signed_k) / (mass_k + eps)
    dominance = float(top_attr.abs().max().item()) / (mass_k + eps)
    return coherence, mass_k, dominance


def coherence_from_topk_supporters(row_attr: torch.Tensor, top_k: int) -> float:
    """Backward-compatible wrapper; prefer :func:`topk_influence_metrics`."""
    return topk_influence_metrics(row_attr, top_k)[0]


def inverse_coherence_from_coherence(coherence: torch.Tensor) -> torch.Tensor:
    """Map coherence in [0, 1] to an uncertainty score (higher = more uncertain)."""
    return (1.0 - coherence.clamp(0.0, 1.0)).float()


# Shared with inverse_mass / inverse_logit_magnitude (fast pilot + watsonx mappers).
DEFAULT_MASS_EPS = 1e-8


def reciprocal_uncertainty(values: torch.Tensor, eps: float = DEFAULT_MASS_EPS) -> torch.Tensor:
    """Higher score = lower magnitude (epistemic-style inverse mass)."""
    return 1.0 / (values + eps)


def map_attribution_structure_to_uncertainty(
    signals: Dict[str, torch.Tensor],
    *,
    eps: float = DEFAULT_MASS_EPS,
) -> Dict[str, torch.Tensor]:
    """
    Map raw DualXDA structure outputs to uncertainty-oriented signals.

    Used by ``run_fast_uncertainty_classification``, watsonx scorers, and any
    script that must not reimplement ``inverse_coherence`` / ``inverse_mass``.
    """
    coherence = signals["coherence"]
    inv_coh = signals.get("inverse_coherence")
    if inv_coh is None:
        inv_coh = inverse_coherence_from_coherence(coherence)
    return {
        "coherence": coherence,
        "inverse_coherence": inv_coh,
        "inverse_mass": reciprocal_uncertainty(signals["mass"], eps=eps),
        "dominance": signals["dominance"],
        "label_disagreement": signals["label_disagreement"],
        "noisy_support_ratio": signals["noisy_support_ratio"],
        "attribution_concentration": signals["attribution_concentration"],
        "cross_class_support": signals["cross_class_support"],
    }


def map_mc_dropout_to_predictive_signals(
    mc_uq: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """MC-dropout predictive baselines (entropy, MI, MSP)."""
    mean_pred = mc_uq["mean_prediction"]
    return {
        "msp_uncertainty": 1.0 - mean_pred.max(dim=1).values,
        "predictive_entropy": mc_uq["entropy"],
        "mutual_info": mc_uq["mutual_info"],
    }


def build_fast_pilot_signal_table(
    *,
    attribution_signals: Dict[str, torch.Tensor],
    logit_magnitude: torch.Tensor,
    mc_uq: Dict[str, torch.Tensor] | None = None,
    predictive_entropy: torch.Tensor | None = None,
    mutual_info: torch.Tensor | None = None,
    mean_prediction: torch.Tensor | None = None,
    eps: float = DEFAULT_MASS_EPS,
) -> Dict[str, torch.Tensor]:
    """
    Canonical ``signal_table`` for ``run_fast_uncertainty_classification`` /
    validation sweeps (single source of truth for exported columns).
    """
    if mc_uq is not None:
        predictive = map_mc_dropout_to_predictive_signals(mc_uq)
    else:
        if mean_prediction is None or predictive_entropy is None or mutual_info is None:
            raise ValueError("Provide mc_uq or all of mean_prediction, predictive_entropy, mutual_info")
        predictive = {
            "msp_uncertainty": 1.0 - mean_prediction.max(dim=1).values,
            "predictive_entropy": predictive_entropy,
            "mutual_info": mutual_info,
        }
    attr = map_attribution_structure_to_uncertainty(attribution_signals, eps=eps)
    return {
        **predictive,
        "coherence": attr["coherence"],
        "inverse_coherence": attr["inverse_coherence"],
        "dominance": attr["dominance"],
        "inverse_mass": attr["inverse_mass"],
        "inverse_logit_magnitude": reciprocal_uncertainty(logit_magnitude, eps=eps),
    }


def normalized_entropy_from_labels(labels: torch.Tensor, num_classes: int) -> float:
    """
    Calculate normalized entropy of a label distribution.
    
    Normalized entropy is in [0, 1] where:
    - 0 = all labels are the same (no uncertainty)
    - 1 = uniform distribution (maximum uncertainty)
    
    Args:
        labels: Label tensor [N]
        num_classes: Total number of classes
        
    Returns:
        Normalized entropy value in [0, 1]
    """
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    probs = counts / counts.sum().clamp_min(1.0)
    mask = probs > 0
    entropy = -(probs[mask] * torch.log(probs[mask])).sum()
    return float((entropy / math.log(num_classes)).item()) if num_classes > 1 else 0.0


@torch.no_grad()
def compute_attribution_structure_signals(
    tracer: DualXDATracer,
    model,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    train_dataset,
    *,
    device: torch.device,
    batch_size: int,
    top_k: int,
    num_classes: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute attribution-based uncertainty signals using DualXDA.
    
    ARCHITECTURE-AGNOSTIC: Works with any model architecture by operating on
    the model's expected input format:
    - DINOv2: Pre-extracted embeddings [N, D]
    - CNN/ResNet: Raw images [N, C, H, W]
    - Custom models: Any tensor format the model accepts
    
    DualXDA automatically hooks into the classifier layer (detected via
    `infer_classifier_layer_name()`) and computes attributions regardless
    of the input representation.
    
    For each evaluation sample, computes:
    
    Standard DualXDA metrics (top-k by |T_i|, same window as label disagreement):
    - mass: sum|T_i| over top-k influences (used for ``inverse_mass``)
    - coherence: |sum T_i| / sum|T_i| over top-k
    - dominance: max|T_i| / sum|T_i| over top-k
    
    Label-based signals:
    - label_disagreement: Entropy of supporter labels (aleatoric indicator)
    - noisy_support_ratio: Fraction of supporters with noisy labels
    
    NEW signals for better uncertainty distinction:
    - attribution_concentration: Gini coefficient of attribution magnitudes
      (High = dominated by few samples → epistemic)
    - cross_class_support: Fraction of supporters from different classes
      (High = cross-class conflict → aleatoric)
    
    Args:
        tracer: DualXDA tracer instance
        model: Trained classifier model (any architecture)
        eval_inputs: Evaluation inputs in model's expected format
                    - For DINOv2: pre-extracted embeddings [N, D]
                    - For CNN/ResNet: raw images [N, C, H, W]
                    - For custom models: any tensor the model accepts
        mean_predictions: Mean predictions [N, num_classes]
        train_dataset: Training dataset with labels and noise info
        device: Device to run on
        batch_size: Batch size for processing
        top_k: Number of top supporters to analyze
        num_classes: Number of classes used to normalize label entropy
        
    Returns:
        Dictionary of signal tensors, each of shape [N]
    """
    from uqlab.triage.dualxda_axioms import infer_classifier_layer_name
    
    layer_name = infer_classifier_layer_name(model)
    print(f"Using DualXDA classifier layer: {layer_name}")

    train_labels = train_dataset.targets
    train_is_noisy = train_dataset.is_noisy

    out = {
        "mass": [],
        "coherence": [],
        "dominance": [],
        "label_disagreement": [],
        "noisy_support_ratio": [],
        "attribution_concentration": [],
        "cross_class_support": [],
    }

    for start in range(0, int(eval_inputs.shape[0]), batch_size):
        end = min(start + batch_size, int(eval_inputs.shape[0]))
        xb = eval_inputs[start:end].to(device)
        targets = mean_predictions[start:end].argmax(dim=1).to(device)
        
        # Get DualXDA attributions
        attr = tracer.traces(x=xb, targets=targets, drop_zero_columns=False)
        batch_mass: List[float] = []
        batch_coherence: List[float] = []
        batch_dominance: List[float] = []
        batch_disagreement: List[float] = []
        batch_noisy_ratio: List[float] = []
        batch_attribution_concentration: List[float] = []
        batch_cross_class_support: List[float] = []

        for row in range(int(attr.shape[0])):
            row_attr = attr[row].detach().cpu()
            predicted_class = int(targets[row].item())
            pos_idx = torch.nonzero(row_attr > 0, as_tuple=False).view(-1)

            coh_k, mass_k, dom_k = topk_influence_metrics(row_attr, top_k)
            batch_coherence.append(coh_k)
            batch_mass.append(mass_k)
            batch_dominance.append(dom_k)

            if pos_idx.numel() == 0:
                # No positive attributions - use default values
                batch_disagreement.append(0.0)
                batch_noisy_ratio.append(0.0)
                batch_attribution_concentration.append(0.0)
                batch_cross_class_support.append(0.0)
                continue

            pos_vals = row_attr[pos_idx]
            k = min(top_k, int(pos_idx.numel()))
            _, top_sub = torch.topk(pos_vals, k)
            top_idx = pos_idx[top_sub]
            top_vals = pos_vals[top_sub]

            supporter_labels = train_labels[top_idx]
            supporter_noisy = train_is_noisy[top_idx].float()
            
            # Existing signals
            batch_disagreement.append(normalized_entropy_from_labels(supporter_labels, num_classes))
            batch_noisy_ratio.append(float(supporter_noisy.mean().item()))
            
            # NEW: Attribution concentration - Gini coefficient of attribution magnitudes
            # High concentration = dominated by few samples (epistemic uncertainty)
            sorted_vals, _ = torch.sort(top_vals)
            n = len(sorted_vals)
            cumsum = torch.cumsum(sorted_vals, dim=0)
            gini = (2.0 * torch.arange(1, n + 1, dtype=torch.float32) * sorted_vals).sum()
            gini = gini / (n * cumsum[-1]) - (n + 1) / n
            batch_attribution_concentration.append(float(gini.item()))
            
            # NEW: Cross-class support - fraction of supporters from different classes than prediction
            # High cross-class support indicates training data conflict (aleatoric uncertainty)
            cross_class_mask = supporter_labels != predicted_class
            cross_class_fraction = float(cross_class_mask.float().mean().item())
            batch_cross_class_support.append(cross_class_fraction)

        out["mass"].append(torch.tensor(batch_mass, dtype=torch.float32))
        out["coherence"].append(torch.tensor(batch_coherence, dtype=torch.float32))
        out["dominance"].append(torch.tensor(batch_dominance, dtype=torch.float32))
        out["label_disagreement"].append(torch.tensor(batch_disagreement, dtype=torch.float32))
        out["noisy_support_ratio"].append(torch.tensor(batch_noisy_ratio, dtype=torch.float32))
        out["attribution_concentration"].append(torch.tensor(batch_attribution_concentration, dtype=torch.float32))
        out["cross_class_support"].append(torch.tensor(batch_cross_class_support, dtype=torch.float32))

    combined = {k: torch.cat(v, dim=0) for k, v in out.items()}
    combined["inverse_coherence"] = inverse_coherence_from_coherence(combined["coherence"])
    return combined

# Made with Bob
