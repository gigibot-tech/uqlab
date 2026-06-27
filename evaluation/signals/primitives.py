"""Primitive key namespace for the EK-FAK signal pipeline."""

from __future__ import annotations

from typing import Dict

import torch

PrimitiveStore = Dict[str, torch.Tensor]

DEFAULT_MASS_EPS = 1e-8


def reciprocal_uncertainty(values: torch.Tensor, eps: float = DEFAULT_MASS_EPS) -> torch.Tensor:
    return 1.0 / (values + eps)


def predicted_class_logit_magnitude(
    det_logits: torch.Tensor,
    mean_pred_det: torch.Tensor,
) -> torch.Tensor:
    """|logit| for the predicted class (deterministic eval forward)."""
    pred = mean_pred_det.argmax(dim=1)
    idx = torch.arange(int(pred.shape[0]), device=det_logits.device)
    return det_logits[idx, pred].abs()


def inverse_coherence_from_coherence(coherence: torch.Tensor) -> torch.Tensor:
    return (1.0 - coherence.clamp(0.0, 1.0)).float()


# deterministic_forward source
FWD_DET_LOGITS = "forward.det_logits"
FWD_MEAN_PRED = "forward.mean_pred"

# mc_dropout source
MC_MEAN_PRED = "mc.mean_prediction"
MC_ENTROPY = "mc.entropy"
MC_MUTUAL_INFO = "mc.mutual_info"

# DualXDA attribution backend
DUALXDA_COHERENCE = "dualxda.coherence"
DUALXDA_MASS = "dualxda.mass"
DUALXDA_DOMINANCE = "dualxda.dominance"
DUALXDA_ENTROPY = "dualxda.entropy"
DUALXDA_PARTICIPATION = "dualxda.participation"
DUALXDA_SIGNED_SPLIT = "dualxda.signed_split"
DUALXDA_VARIANCE = "dualxda.variance"

# EK-FAC / Kronfluence attribution backend
EK_FAK_COHERENCE = "ek_fak.coherence"
EK_FAK_MASS = "ek_fak.mass"
EK_FAK_DOMINANCE = "ek_fak.dominance"

# Grad-dot attribution backend
GRADDOT_COHERENCE = "graddot.coherence"
GRADDOT_MASS = "graddot.mass"
GRADDOT_DOMINANCE = "graddot.dominance"

# Full [n_eval, n_train] influence matrices (persisted under zwischen/)
INFLUENCE_DUALXDA = "influence.dualxda"
INFLUENCE_EK_FAK = "influence.ek_fak"
INFLUENCE_GRADDOT = "influence.graddot"

# Legacy attribution keys (write-through alias for DualXDA during migration)
ATTR_COHERENCE = "attribution.coherence"
ATTR_MASS = "attribution.mass"
ATTR_DOMINANCE = "attribution.dominance"


def namespaced_attribution_store(
    backend: str,
    coherence: torch.Tensor,
    mass: torch.Tensor,
    dominance: torch.Tensor,
    *,
    write_legacy_alias: bool = False,
) -> PrimitiveStore:
    """Write backend-prefixed primitive keys; optionally mirror legacy ``attribution.*``."""
    out: PrimitiveStore = {
        f"{backend}.coherence": coherence,
        f"{backend}.mass": mass,
        f"{backend}.dominance": dominance,
    }
    if write_legacy_alias and backend == "dualxda":
        out[ATTR_COHERENCE] = coherence
        out[ATTR_MASS] = mass
        out[ATTR_DOMINANCE] = dominance
    return out


def attribution_distribution_store(
    backend: str,
    *,
    entropy: torch.Tensor,
    participation: torch.Tensor,
    signed_split: torch.Tensor,
    variance: torch.Tensor,
) -> PrimitiveStore:
    """Write full-vector attribution distribution primitives for one backend."""
    return {
        f"{backend}.entropy": entropy,
        f"{backend}.participation": participation,
        f"{backend}.signed_split": signed_split,
        f"{backend}.variance": variance,
    }
