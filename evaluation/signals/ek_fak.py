"""
EK-FAC / Kronfluence attribution backend for fast-pilot structure signals.

Ports the toy demo workflow (``prepare_model`` → ``Analyzer`` → pairwise scores)
and maps per-query influence rows through :func:`topk_influence_metrics`.

v1 scope: feature-space MLP heads with Linear layers (same constraint as the
Kronfluence toy notebook). Requires optional ``kronfluence`` package.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

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


def _extract_score_matrix(loaded: dict | torch.Tensor | None) -> torch.Tensor:
    if loaded is None:
        raise RuntimeError("EK-FAC: Kronfluence returned no pairwise scores")
    if isinstance(loaded, torch.Tensor):
        return loaded.float()
    if isinstance(loaded, dict):
        from kronfluence.utils.constants import ALL_MODULE_NAME

        scores = loaded.get(ALL_MODULE_NAME)
        if scores is None:
            scores = loaded.get("all_modules")
        if scores is None and loaded:
            scores = next(iter(loaded.values()))
        if isinstance(scores, torch.Tensor):
            return scores.float()
    raise TypeError(f"EK-FAC: unexpected score payload type: {type(loaded)!r}")


def _ek_fak_device(device: torch.device) -> torch.device:
    """Kronfluence is CPU/CUDA-only; MPS inputs vs CPU weights cause device errors."""
    if device.type == "mps":
        return torch.device("cpu")
    return device


def _compute_pairwise_scores(
    *,
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    mean_predictions: torch.Tensor,
    device: torch.device,
    batch_size: int,
    train_batch_size: int,
    cache_dir: Path,
) -> torch.Tensor:
    Analyzer, prepare_model, Task = _require_kronfluence()
    ek_device = _ek_fak_device(device)
    if ek_device != device:
        print(f"EK-FAC: running on {ek_device} (Kronfluence does not support {device.type})")

    train_x, train_y = _train_tensors(train_dataset)
    train_ds = TensorDataset(train_x, train_y.long())
    eval_targets = mean_predictions.argmax(dim=1).long()
    query_ds = TensorDataset(eval_inputs.cpu(), eval_targets.cpu())

    class _ExperimentTask(Task):
        @staticmethod
        def _prepare_inputs(inputs: torch.Tensor) -> torch.Tensor:
            # Kronfluence 1.x hooks need a graph; frozen params require input grad.
            if not inputs.requires_grad:
                inputs = inputs.detach().requires_grad_(True)
            return inputs

        def compute_train_loss(self, batch, model, sample=False):
            inputs, labels = batch
            inputs = self._prepare_inputs(inputs.to(ek_device))
            labels = labels.to(ek_device)
            return F.cross_entropy(model(inputs), labels)

        def compute_measurement(self, batch, model):
            inputs, labels = batch
            inputs = self._prepare_inputs(inputs.to(ek_device))
            labels = labels.to(ek_device)
            logits = model(inputs)
            return F.cross_entropy(logits, labels)

    task = _ExperimentTask()
    work_model = copy.deepcopy(model)
    work_model.to(ek_device).eval()
    prepared = prepare_model(work_model, task)
    analyzer = Analyzer(
        analysis_name="uqlab_ek_fak",
        model=prepared,
        task=task,
        disable_tqdm=False,
        output_dir=str(cache_dir),
    )

    per_query_bs = max(1, min(int(batch_size), len(query_ds)))
    per_train_bs = max(1, min(int(train_batch_size), len(train_ds)))

    print(f"EK-FAC: fitting factors ({len(train_ds):,} train samples, batch {per_train_bs})...")
    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_ds,
        per_device_batch_size=per_train_bs,
        overwrite_output_dir=True,
    )
    print(f"EK-FAC: computing pairwise scores ({len(query_ds):,} eval × {len(train_ds):,} train)...")
    loaded = analyzer.compute_pairwise_scores(
        scores_name=SCORES_NAME,
        factors_name="ekfac",
        query_dataset=query_ds,
        train_dataset=train_ds,
        per_device_query_batch_size=per_query_bs,
        per_device_train_batch_size=per_train_bs,
        overwrite_output_dir=True,
    )
    if loaded is None:
        loaded = analyzer.load_pairwise_scores(SCORES_NAME)
    print("EK-FAC: done.")
    return _extract_score_matrix(loaded).detach().cpu()


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
    cache_dir = run_cache_dir / "ek_fak"
    scores = _load_cached_scores(cache_dir)
    if scores is None:
        print("EK-FAC: fitting Kronfluence factors and computing pairwise scores...")
        scores = _compute_pairwise_scores(
            model=model,
            train_dataset=train_dataset,
            eval_inputs=eval_inputs,
            mean_predictions=mean_predictions,
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
    with torch.no_grad():
        structure = structure_signals_from_influence_matrix(scores, top_k)
    structure["influence_matrix"] = scores
    return structure
