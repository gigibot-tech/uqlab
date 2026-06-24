"""
EK-FAC / Kronfluence attribution backend for fast-pilot structure signals.

Ports the toy demo workflow (``prepare_model`` → ``Analyzer`` → pairwise scores)
and maps per-query influence rows through :func:`topk_influence_metrics`.

v1 scope: feature-space MLP heads with Linear layers (same constraint as the
Kronfluence toy notebook). Requires optional ``kronfluence`` package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from uqlab.evaluation.signals.attribution import topk_influence_metrics

if TYPE_CHECKING:
    import torch.nn as nn

SCORES_CACHE_NAME = "pairwise_scores.pt"
SCORES_NAME = "ekfac_fast_pilot"


def _require_kronfluence():
    if importlib.util.find_spec("kronfluence") is None:
        raise ImportError(
            "EK-FAC attribution requires the optional `kronfluence` package. "
            "Install with: pip install kronfluence "
            "(or `uv sync --extra ek_fak` when that extra is enabled)."
        )
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.task import Task

    return Analyzer, prepare_model, Task


def structure_signals_from_influence_matrix(
    scores: torch.Tensor,
    top_k: int,
) -> Dict[str, torch.Tensor]:
    """Map ``[n_query, n_train]`` influence rows to coherence / mass / dominance."""
    n_query = int(scores.shape[0])
    coherence = torch.zeros(n_query, dtype=torch.float32)
    mass = torch.zeros(n_query, dtype=torch.float32)
    dominance = torch.zeros(n_query, dtype=torch.float32)
    for i in range(n_query):
        c, m, d = topk_influence_metrics(scores[i], top_k)
        coherence[i] = c
        mass[i] = m
        dominance[i] = d
    return {"coherence": coherence, "mass": mass, "dominance": dominance}


def _train_tensors(train_dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(train_dataset, "features"):
        x = train_dataset.features
        y = train_dataset.targets
        return x, y
    if hasattr(train_dataset, "__getitem__"):
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            if isinstance(sample, (tuple, list)):
                xs.append(sample[0])
                label = sample[1] if len(sample) > 1 else torch.tensor(0)
                ys.append(label if isinstance(label, torch.Tensor) else torch.tensor(label))
            else:
                xs.append(sample)
                ys.append(torch.tensor(0))
        return torch.stack(xs), torch.stack(ys)
    raise TypeError(f"Unsupported train_dataset type: {type(train_dataset)!r}")


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
    device: torch.device,
    batch_size: int,
    train_batch_size: int,
    cache_dir: Path,
) -> torch.Tensor:
    Analyzer, prepare_model, Task = _require_kronfluence()

    train_x, train_y = _train_tensors(train_dataset)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y.long()),
        batch_size=train_batch_size,
        shuffle=False,
    )
    query_loader = DataLoader(
        TensorDataset(eval_inputs.cpu(), torch.zeros(len(eval_inputs), dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )

    class _ExperimentTask(Task):
        def compute_train_loss(self, batch, model, sample=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            return F.cross_entropy(model(inputs), labels)

        def compute_measurement(self, batch, model):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds = model(inputs).argmax(dim=-1)
            return (preds == labels).float()

    model = model.to(device)
    model.eval()
    prepared = prepare_model(model)
    analyzer = Analyzer(
        analysis_name="uqlab_ek_fak",
        model=prepared,
        task=_ExperimentTask(),
    )
    analyzer.set_kwargs("disable_tqdm", True)

    factors_dir = cache_dir / "factors"
    scores_dir = cache_dir / "scores"
    factors_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_loader,
        per_device_batch_size=None,
        factor_args=None,
        overwrite_output_dir=True,
    )
    analyzer.compute_pairwise_scores(
        scores_name=SCORES_NAME,
        factors_name="ekfac",
        query_dataset=query_loader,
        train_dataset=train_loader,
        per_device_query_batch_size=None,
        per_device_train_batch_size=None,
        overwrite_output_dir=True,
    )
    loaded = analyzer.load_pairwise_scores(SCORES_NAME)
    if isinstance(loaded, dict):
        scores = loaded.get("all_modules")
        if scores is None:
            scores = next(iter(loaded.values()))
    else:
        scores = loaded
    return scores.detach().cpu().float()


@torch.no_grad()
def compute_ek_fak_structure_signals(
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    top_k: int,
    run_cache_dir: Path,
    train_batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Compute EK-FAC structure primitives (coherence, mass, dominance) per eval sample."""
    del mean_predictions  # predicted class not required for pairwise score rows
    cache_dir = run_cache_dir / "ek_fak"
    scores = _load_cached_scores(cache_dir)
    if scores is None:
        print("EK-FAC: fitting Kronfluence factors and computing pairwise scores...")
        scores = _compute_pairwise_scores(
            model=model,
            train_dataset=train_dataset,
            eval_inputs=eval_inputs,
            device=device,
            batch_size=batch_size,
            train_batch_size=train_batch_size,
            cache_dir=cache_dir,
        )
        _save_cached_scores(cache_dir, scores)
    else:
        print(f"EK-FAC: loaded cached pairwise scores from {cache_dir}")

    if int(scores.shape[0]) != int(eval_inputs.shape[0]):
        raise ValueError(
            f"EK-FAC score rows ({scores.shape[0]}) != eval samples ({eval_inputs.shape[0]}). "
            "Delete the ek_fak cache and re-run."
        )
    return structure_signals_from_influence_matrix(scores, top_k)
