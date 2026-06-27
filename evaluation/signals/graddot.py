"""
Grad-dot (gradient dot product) attribution backend.

For each eval sample, compute the loss gradient w.r.t. model parameters and score
training samples by the dot product between eval and train gradients (TracIn-style).
Rows are mapped to coherence / mass / dominance via :func:`structure_signals_from_influence_matrix`.

Large models (ResNet, etc.) use chunked Johnson–Lindenstrauss projection plus
streaming batched dot products so we never materialize ``[n_train, n_params]`` gradients.
"""

from __future__ import annotations

import json
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
META_CACHE_NAME = "grad_dot_meta.json"

# Above this parameter count, use random projection instead of full gradients.
PARAM_COUNT_PROJECTION_THRESHOLD = 100_000
DEFAULT_PROJECTION_DIM = 128
GRAD_CHUNK_SIZE = 65_536
TRAIN_DOT_BATCH_SIZE = 32


def _param_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def _projection_dim(model: nn.Module) -> int | None:
    if _param_count(model) > PARAM_COUNT_PROJECTION_THRESHOLD:
        return DEFAULT_PROJECTION_DIM
    return None


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is not None:
            parts.append(param.grad.detach().flatten())
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def _project_flat_grad(flat: torch.Tensor, *, proj_dim: int, seed: int) -> torch.Tensor:
    """Memory-efficient JL-style projection without storing the full random matrix."""
    flat = flat.cpu().float()
    out = torch.zeros(proj_dim, dtype=torch.float32)
    n_chunks = (flat.numel() + GRAD_CHUNK_SIZE - 1) // GRAD_CHUNK_SIZE
    scale = 1.0 / (n_chunks**0.5) if n_chunks > 1 else 1.0
    for ci, start in enumerate(range(0, flat.numel(), GRAD_CHUNK_SIZE)):
        end = min(start + GRAD_CHUNK_SIZE, flat.numel())
        chunk = flat[start:end]
        gen = torch.Generator().manual_seed(seed + ci)
        R = torch.randn(proj_dim, end - start, generator=gen, dtype=torch.float32)
        R.mul_(1.0 / (proj_dim**0.5))
        out.add_(R @ chunk, alpha=scale)
    return out


def _encode_grad(
    flat: torch.Tensor,
    *,
    proj_dim: int | None,
    seed: int,
) -> torch.Tensor:
    if proj_dim is None:
        return flat.cpu().float()
    return _project_flat_grad(flat, proj_dim=proj_dim, seed=seed)


def _per_sample_grad(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    proj_dim: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    xb = x.unsqueeze(0).to(device)
    yb = y.unsqueeze(0).long().to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    flat = _flatten_grads(model)
    return _encode_grad(flat, proj_dim=proj_dim, seed=seed)


def _scores_cache_path(cache_dir: Path) -> Path:
    return cache_dir / SCORES_CACHE_NAME


def _meta_cache_path(cache_dir: Path) -> Path:
    return cache_dir / META_CACHE_NAME


def _load_cached_scores(cache_dir: Path, *, n_eval: int) -> torch.Tensor | None:
    path = _scores_cache_path(cache_dir)
    meta_path = _meta_cache_path(cache_dir)
    if not path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("n_eval", -1)) != n_eval:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "scores" in payload:
        scores = payload["scores"]
    elif isinstance(payload, torch.Tensor):
        scores = payload
    else:
        return None
    return scores


def _save_cached_scores(
    cache_dir: Path,
    scores: torch.Tensor,
    *,
    meta: dict,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores.cpu()}, _scores_cache_path(cache_dir))
    _meta_cache_path(cache_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _compute_pairwise_scores(
    *,
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    device: torch.device,
    run_cache_dir: Path,
) -> torch.Tensor:
    """Streaming grad-dot scores with bounded peak memory."""
    from tqdm.auto import tqdm

    train_x, train_y = _train_tensors(train_dataset)
    eval_targets = mean_predictions.argmax(dim=1).long()
    n_train = int(train_x.shape[0])
    n_eval = int(eval_inputs.shape[0])
    proj_dim = _projection_dim(model)
    proj_seed = 42

    was_training = model.training
    model.eval()

    eval_vecs: list[torch.Tensor] = []
    for i in tqdm(range(n_eval), desc="Grad-dot eval gradients", unit="sample"):
        with torch.enable_grad():
            eval_vecs.append(
                _per_sample_grad(
                    model,
                    eval_inputs[i].cpu(),
                    eval_targets[i].cpu(),
                    device,
                    proj_dim=proj_dim,
                    seed=proj_seed + 1_000_003 + i,
                )
            )
    eval_mat = torch.stack(eval_vecs, dim=0)

    scores = torch.empty(n_eval, n_train, dtype=torch.float32)
    for j0 in tqdm(
        range(0, n_train, TRAIN_DOT_BATCH_SIZE),
        desc="Grad-dot train dot products",
        unit="batch",
    ):
        j1 = min(j0 + TRAIN_DOT_BATCH_SIZE, n_train)
        batch_vecs: list[torch.Tensor] = []
        for j in range(j0, j1):
            with torch.enable_grad():
                batch_vecs.append(
                    _per_sample_grad(
                        model,
                        train_x[j],
                        train_y[j],
                        device,
                        proj_dim=proj_dim,
                        seed=proj_seed + j,
                    )
                )
        train_batch = torch.stack(batch_vecs, dim=0)
        scores[:, j0:j1] = eval_mat @ train_batch.T
        del batch_vecs, train_batch
        if device.type in ("mps", "cuda"):
            getattr(torch, device.type).empty_cache()

    if was_training:
        model.train()
    else:
        model.eval()

    if scores.shape[1] == 0:
        raise ValueError("Grad-dot: model produced no parameter gradients")

    meta = {
        "n_eval": n_eval,
        "n_train": n_train,
        "proj_dim": proj_dim,
        "param_count": _param_count(model),
        "version": 2,
    }
    _save_cached_scores(run_cache_dir / "graddot", scores, meta=meta)
    return scores


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
    n_eval = int(eval_inputs.shape[0])
    scores = _load_cached_scores(cache_dir, n_eval=n_eval)
    if scores is None:
        train_x, _ = _train_tensors(train_dataset)
        n_train = int(train_x.shape[0])
        n_params = _param_count(model)
        proj_dim = _projection_dim(model)
        if proj_dim is not None:
            print(
                f"Grad-dot: {n_train:,} train + {n_eval:,} eval samples "
                f"({n_params:,} params → {proj_dim}-dim projection)..."
            )
        else:
            print(
                f"Grad-dot: {n_train:,} train + {n_eval:,} eval backward passes..."
            )
        scores = _compute_pairwise_scores(
            model=model,
            train_dataset=train_dataset,
            eval_inputs=eval_inputs,
            mean_predictions=mean_predictions,
            device=device,
            run_cache_dir=run_cache_dir,
        )
    else:
        print(f"Grad-dot: loaded cached pairwise scores from {cache_dir}")

    if int(scores.shape[0]) != n_eval:
        raise ValueError(
            f"Grad-dot score rows ({scores.shape[0]}) != eval samples ({n_eval}). "
            "Delete the graddot cache and re-run."
        )
    with torch.no_grad():
        structure = structure_signals_from_influence_matrix(scores, top_k)
    structure["influence_matrix"] = scores
    return structure
