"""
Grad-dot (gradient dot product) attribution backend.

For each eval sample, compute the loss gradient w.r.t. model parameters and score
training samples by the dot product between eval and train gradients (TracIn-style).
Rows are mapped to coherence / mass / dominance via :func:`structure_signals_from_influence_matrix`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from uqlab.evaluation.signals.ek_fak import (
    _train_tensors,
    structure_signals_from_influence_matrix,
)

if TYPE_CHECKING:
    import torch.nn as nn

SCORES_CACHE_NAME = "grad_dot_scores.pt"


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            parts.append(param.grad.detach().flatten())
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def _per_sample_grad(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    xb = x.unsqueeze(0).to(device)
    yb = y.unsqueeze(0).long().to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    return _flatten_grads(model).cpu().float()


def _grad_matrix(
    model: nn.Module,
    xs: torch.Tensor,
    ys: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    was_training = model.training
    model.train()
    rows: list[torch.Tensor] = []
    for i in range(int(xs.shape[0])):
        with torch.enable_grad():
            rows.append(_per_sample_grad(model, xs[i], ys[i], device))
    if not was_training:
        model.eval()
    return torch.stack(rows, dim=0)


def _scores_cache_path(cache_dir: Path) -> Path:
    return cache_dir / SCORES_CACHE_NAME


def _load_cached_scores(cache_dir: Path) -> torch.Tensor | None:
    path = _scores_cache_path(cache_dir)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "scores" in payload:
        return payload["scores"]
    if isinstance(payload, torch.Tensor):
        return payload
    return None


def _save_cached_scores(cache_dir: Path, scores: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores.cpu()}, _scores_cache_path(cache_dir))


def _compute_pairwise_scores(
    *,
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    train_x, train_y = _train_tensors(train_dataset)
    train_grads = _grad_matrix(model, train_x, train_y, device)
    eval_targets = mean_predictions.argmax(dim=1).long()
    eval_grads = _grad_matrix(model, eval_inputs.cpu(), eval_targets.cpu(), device)
    if train_grads.shape[1] == 0:
        raise ValueError("Grad-dot: model produced no parameter gradients")
    return (eval_grads @ train_grads.T).float()


def compute_graddot_structure_signals(
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    *,
    device: torch.device,
    top_k: int,
    run_cache_dir: Path,
) -> Dict[str, torch.Tensor]:
    """Compute grad-dot structure primitives (coherence, mass, dominance) per eval sample."""
    cache_dir = run_cache_dir / "graddot"
    scores = _load_cached_scores(cache_dir)
    if scores is None:
        print("Grad-dot: computing per-sample gradient dot products...")
        scores = _compute_pairwise_scores(
            model=model,
            train_dataset=train_dataset,
            eval_inputs=eval_inputs,
            mean_predictions=mean_predictions,
            device=device,
        )
        _save_cached_scores(cache_dir, scores)
    else:
        print(f"Grad-dot: loaded cached pairwise scores from {cache_dir}")

    if int(scores.shape[0]) != int(eval_inputs.shape[0]):
        raise ValueError(
            f"Grad-dot score rows ({scores.shape[0]}) != eval samples ({eval_inputs.shape[0]}). "
            "Delete the graddot cache and re-run."
        )
    with torch.no_grad():
        return structure_signals_from_influence_matrix(scores, top_k)
