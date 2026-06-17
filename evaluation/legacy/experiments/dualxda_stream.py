"""
Streaming DualXDA (DualDA) scoring helpers.

Why this exists:
  - DualDA attributions are [B, N_train] and can be huge if you concatenate them.
  - For AURC we only need per-sample scalars (signal_uncertainty, triage_uncertainty, spurious_score + diagnostics),
    so we compute those scalars batch-wise and concatenate *only the scalars*.

This helper is used by experiments/run_aurc_benchmark.py and can be reused elsewhere.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from tqdm import tqdm


@torch.no_grad()
def compute_dualxda_scores_streaming(
    *,
    tracer,
    dataloader,
    mean_predictions: torch.Tensor,
    device: torch.device | str,
    drop_zero_columns: bool = False,
    desc: str = "DualXDA attributions",
) -> Dict[str, torch.Tensor]:
    """
    Args:
      tracer: DualXDATracer instance
      dataloader: evaluation loader (shuffle=False)
      mean_predictions: [N, C] probabilities aligned with dataloader order
      device: device for x and targets
    Returns:
      dict of concatenated tensors [N] for:
        - signal_uncertainty, triage_uncertainty, spurious_score
        - mass, signal, coherence, dominance
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device

    # Accumulators for scalars only.
    out: Dict[str, list[torch.Tensor]] = {
        "signal_uncertainty": [],
        "triage_uncertainty": [],
        "spurious_score": [],
        "mass": [],
        "signal": [],
        "coherence": [],
        "dominance": [],
    }

    offset = 0
    for batch in tqdm(dataloader, desc=desc):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 4:
            x, y, _, _ = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        b = int(y.shape[0])
        x = x.to(dev)
        # Target to attribute: predicted class (standard in attribution benchmarks).
        batch_targets = mean_predictions[offset : offset + b].argmax(dim=1).to(dev)

        attr = tracer.traces(x=x, targets=batch_targets, drop_zero_columns=drop_zero_columns)
        scores = tracer.uncertainty_scores(attr)

        for k in out.keys():
            out[k].append(scores[k].detach().cpu())

        offset += b

    if offset != int(mean_predictions.shape[0]):
        raise RuntimeError(f"DualXDA streaming mismatch: saw {offset} samples, but predictions has {int(mean_predictions.shape[0])}.")

    return {k: torch.cat(v, dim=0) if len(v) > 0 else torch.empty(0) for k, v in out.items()}


@torch.no_grad()
def compute_dualxda_attributed_softmax_scores_streaming(
    *,
    tracer,
    dataloader,
    mean_predictions: torch.Tensor,
    device: torch.device | str,
    drop_zero_columns: bool = False,
    desc: str = "DualXDA multiclass attributions",
) -> Dict[str, torch.Tensor]:
    """
    Compute classwise attributed-score softmax metrics for the model-predicted class.

    For each sample x and class c, DualDA provides signed traces tau_c(x, i). Summing over
    training points yields an attributed class score:

        s_tilde_c(x) = sum_i tau_c(x, i)

    We then form an attributed softmax over classes and derive three uncertainty-form scores:
      - attributed_softmax_uncertainty = 1 - p_tilde_hatc
      - attributed_coherence_uncertainty = 1 - C_hatc
      - attributed_softmax_coherence_uncertainty = 1 - p_tilde_hatc * C_hatc

    where hatc is the class predicted by the base model (mean_predictions.argmax).
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    num_classes = int(mean_predictions.shape[1])
    eps = float(getattr(getattr(tracer, "thresholds", None), "eps", 1e-12))

    out: Dict[str, list[torch.Tensor]] = {
        "attributed_softmax_confidence": [],
        "attributed_softmax_uncertainty": [],
        "attributed_coherence_confidence": [],
        "attributed_coherence_uncertainty": [],
        "attributed_softmax_coherence_confidence": [],
        "attributed_softmax_coherence_uncertainty": [],
    }

    offset = 0
    for batch in tqdm(dataloader, desc=desc):
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 4:
            x, y, _, _ = batch
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        b = int(y.shape[0])
        x = x.to(dev)
        batch_targets = mean_predictions[offset : offset + b].argmax(dim=1).to(dev)

        class_scores = []
        class_masses = []

        for c in range(num_classes):
            class_targets = torch.full((b,), c, dtype=torch.long, device=dev)
            attr_c = tracer.traces(x=x, targets=class_targets, drop_zero_columns=drop_zero_columns)
            class_scores.append(attr_c.sum(dim=1))
            class_masses.append(attr_c.abs().sum(dim=1))

        score_matrix = torch.stack(class_scores, dim=1)  # [B, C]
        mass_matrix = torch.stack(class_masses, dim=1)   # [B, C]
        attributed_probs = torch.softmax(score_matrix, dim=1)

        gather_idx = batch_targets.view(-1, 1)
        predicted_softmax_conf = attributed_probs.gather(1, gather_idx).squeeze(1)
        predicted_score_abs = score_matrix.abs().gather(1, gather_idx).squeeze(1)
        predicted_mass = mass_matrix.gather(1, gather_idx).squeeze(1)
        predicted_coherence_conf = predicted_score_abs / (predicted_mass + eps)
        predicted_combined_conf = predicted_softmax_conf * predicted_coherence_conf

        out["attributed_softmax_confidence"].append(predicted_softmax_conf.detach().cpu())
        out["attributed_softmax_uncertainty"].append((1.0 - predicted_softmax_conf).detach().cpu())
        out["attributed_coherence_confidence"].append(predicted_coherence_conf.detach().cpu())
        out["attributed_coherence_uncertainty"].append((1.0 - predicted_coherence_conf).detach().cpu())
        out["attributed_softmax_coherence_confidence"].append(predicted_combined_conf.detach().cpu())
        out["attributed_softmax_coherence_uncertainty"].append((1.0 - predicted_combined_conf).detach().cpu())

        offset += b

    if offset != int(mean_predictions.shape[0]):
        raise RuntimeError(
            f"DualXDA attributed-softmax streaming mismatch: saw {offset} samples, "
            f"but predictions has {int(mean_predictions.shape[0])}."
        )

    return {k: torch.cat(v, dim=0) if len(v) > 0 else torch.empty(0) for k, v in out.items()}
