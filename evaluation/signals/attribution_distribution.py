"""
Full-vector attribution distribution measures (DualXDA and notebook analysis).

Unlike top-k structure signals (coherence, dominance, mass), these use the entire
signed attribution row T_i over all training samples.
"""

from __future__ import annotations

import math
from typing import Dict

import torch

DEFAULT_DIST_EPS = 1e-8


def attribution_entropy_row(row_attr: torch.Tensor, *, eps: float = DEFAULT_DIST_EPS) -> float:
    """Normalized Shannon entropy of |T_i| over the full training vector, in [0, 1]."""
    flat = row_attr.flatten().float()
    n = int(flat.numel())
    if n <= 1:
        return 0.0
    abs_a = flat.abs()
    total = float(abs_a.sum().item())
    if total <= eps:
        return 0.0
    p = abs_a / total
    mask = p > eps
    if not bool(mask.any()):
        return 0.0
    ent = float(-(p[mask] * torch.log(p[mask].clamp_min(eps))).sum().item())
    return ent / math.log(n)


def attribution_participation_row(row_attr: torch.Tensor, *, eps: float = DEFAULT_DIST_EPS) -> float:
    """Normalized participation ratio: (sum |T_i|)^2 / (N * sum T_i^2), in [0, 1]."""
    flat = row_attr.flatten().float()
    n = int(flat.numel())
    if n <= 0:
        return 0.0
    abs_a = flat.abs()
    sum_abs = float(abs_a.sum().item())
    sum_sq = float((abs_a * abs_a).sum().item())
    if sum_sq <= eps:
        return 0.0
    return (sum_abs * sum_abs) / (n * sum_sq)


def attribution_signed_split_row(row_attr: torch.Tensor, *, eps: float = DEFAULT_DIST_EPS) -> float:
    """Balance of positive vs negative attribution mass, in [0, 1]. High = for-and-against."""
    flat = row_attr.flatten().float()
    pos_mass = float(flat.clamp_min(0).sum().item())
    neg_mass = float((-flat.clamp_max(0)).sum().item())
    denom = pos_mass + neg_mass
    if denom <= eps:
        return 0.0
    return 2.0 * min(pos_mass, neg_mass) / denom


def attribution_variance_row(row_attr: torch.Tensor) -> float:
    """Variance of signed attributions over the full training vector."""
    flat = row_attr.flatten().float()
    if flat.numel() <= 1:
        return 0.0
    return float(flat.var(unbiased=False).item())


def compute_attribution_distribution_signals(
    attr: torch.Tensor,
    *,
    eps: float = DEFAULT_DIST_EPS,
) -> Dict[str, torch.Tensor]:
    """
    Batch helper for full attribution matrix.

    Args:
        attr: Signed attributions ``[B, N_train]``.

    Returns:
        Dict with ``entropy``, ``participation``, ``signed_split``, ``variance``
        each shaped ``[B]``.
    """
    if attr.ndim != 2:
        raise ValueError(f"attr must be [B, N_train], got shape {tuple(attr.shape)}")

    x = attr.float()
    b, n = int(x.shape[0]), int(x.shape[1])
    if b == 0:
        empty = torch.zeros(0, dtype=torch.float32)
        return {
            "entropy": empty,
            "participation": empty,
            "signed_split": empty,
            "variance": empty,
        }

    abs_a = x.abs()
    total = abs_a.sum(dim=-1).clamp_min(eps)
    p = abs_a / total.unsqueeze(-1)

    ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)
    log_n = math.log(max(n, 2))
    entropy = ent / log_n
    entropy = torch.where(total > eps, entropy, torch.zeros_like(entropy))

    sum_abs = abs_a.sum(dim=-1)
    sum_sq = (abs_a * abs_a).sum(dim=-1).clamp_min(eps)
    participation = (sum_abs * sum_abs) / (max(n, 1) * sum_sq)
    participation = torch.where(sum_sq > eps, participation, torch.zeros_like(participation))

    pos_mass = x.clamp_min(0).sum(dim=-1)
    neg_mass = (-x.clamp_max(0)).sum(dim=-1)
    denom = (pos_mass + neg_mass).clamp_min(eps)
    signed_split = 2.0 * torch.minimum(pos_mass, neg_mass) / denom
    signed_split = torch.where(denom > eps, signed_split, torch.zeros_like(signed_split))

    if n <= 1:
        variance = torch.zeros(b, dtype=torch.float32, device=x.device)
    else:
        variance = x.var(dim=-1, unbiased=False)

    return {
        "entropy": entropy.float(),
        "participation": participation.float(),
        "signed_split": signed_split.float(),
        "variance": variance.float(),
    }
