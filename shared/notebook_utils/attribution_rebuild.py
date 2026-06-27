"""Rebuild DualXDA attributions from a completed run directory (notebook helper)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from uqlab.evaluation.signals.dualxda_tracer import DualXDATracer, infer_classifier_layer_name
from uqlab.evaluation.signals.attribution_distribution import compute_attribution_distribution_signals
from uqlab.models.classification_models import EmbeddingDataset
from uqlab.models.factory import build_model
from uqlab.shared.config.classification import ExperimentConfig


def _load_run_experiment_config(run_dir: Path) -> ExperimentConfig:
    run_dir = Path(run_dir)
    for candidate in (
        run_dir / "config.yaml",
        run_dir.parent / "config.yaml",
    ):
        if candidate.is_file():
            return ExperimentConfig.from_yaml(candidate)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text())
        nested = payload.get("config") or {}
        config_file = nested.get("config_file")
        if config_file and Path(config_file).is_file():
            return ExperimentConfig.from_yaml(Path(config_file))
    training_cfg = run_dir / "training_data.config.json"
    if training_cfg.is_file():
        data = json.loads(training_cfg.read_text())
        tmp = run_dir / "_notebook_config.yaml"
        tmp.write_text(yaml.safe_dump(data, sort_keys=False))
        try:
            return ExperimentConfig.from_yaml(tmp)
        finally:
            tmp.unlink(missing_ok=True)
    raise FileNotFoundError(f"No config for run under {run_dir}")


def _stratified_eval_indices(
    group_labels: torch.Tensor,
    max_eval_samples: int,
) -> torch.Tensor:
    """Pick up to *max_eval_samples* eval rows balanced across group codes."""
    labels = group_labels.long().flatten()
    n = int(labels.numel())
    if max_eval_samples >= n:
        return torch.arange(n, dtype=torch.long)
    unique = torch.unique(labels)
    per_group = max(1, max_eval_samples // max(1, int(unique.numel())))
    picked: list[torch.Tensor] = []
    remaining = max_eval_samples
    for g in unique.tolist():
        if remaining <= 0:
            break
        mask = (labels == g).nonzero(as_tuple=True)[0]
        take = min(per_group, int(mask.numel()), remaining)
        if take > 0:
            picked.append(mask[:take])
            remaining -= take
    if not picked:
        return torch.arange(min(max_eval_samples, n), dtype=torch.long)
    out = torch.cat(picked)
    if out.numel() > max_eval_samples:
        out = out[:max_eval_samples]
    return out


def rebuild_tracer_and_attr(
    run_dir: Path,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """
    Reload checkpoint + ``results.pt``, recompute full DualXDA attribution matrix.

    Returns dict with ``attr`` [N_eval, N_train], distribution tensors, ``group_labels``,
    ``config``, ``model``, ``train_dataset``.
    """
    run_dir = Path(run_dir)
    device = torch.device(device)
    config = _load_run_experiment_config(run_dir)

    ckpt_path = run_dir / "checkpoint.pt"
    results_path = run_dir / "results.pt"
    if not ckpt_path.is_file() or not results_path.is_file():
        raise FileNotFoundError(f"Need checkpoint.pt and results.pt in {run_dir}")

    results = torch.load(results_path, map_location="cpu", weights_only=False)
    train_emb = results.get("train_embeddings")
    eval_emb = results.get("eval_embeddings")
    if train_emb is None or eval_emb is None:
        raise ValueError("results.pt must contain train_embeddings and eval_embeddings")

    feature_dim = int(train_emb.shape[1])
    model = build_model(
        config=config.model,
        num_classes=10,
        feature_dim=feature_dim,
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") or checkpoint
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    train_dataset = EmbeddingDataset(
        features=train_emb,
        labels=results["train_noisy_labels"],
        clean_labels=results["train_labels"],
        is_noisy=results["train_is_noisy"],
        original_indices=results["train_indices"],
    )

    eval_inputs = eval_emb.float()
    eval_indices = torch.arange(int(eval_inputs.shape[0]), dtype=torch.long)
    group_labels_full = results["eval_group_labels"].long().flatten()
    if max_eval_samples is not None:
        eval_indices = _stratified_eval_indices(group_labels_full, max_eval_samples)
        eval_inputs = eval_inputs[eval_indices]
        group_labels_full = group_labels_full[eval_indices]

    mean_pred = results.get("mean_prediction_deterministic")
    if mean_pred is not None:
        mean_pred = mean_pred[eval_indices]
    if mean_pred is None:
        with torch.no_grad():
            chunks = []
            for start in range(0, int(eval_inputs.shape[0]), batch_size):
                end = min(start + batch_size, int(eval_inputs.shape[0]))
                chunks.append(model(eval_inputs[start:end].to(device)).cpu())
            mean_pred = torch.cat(chunks, dim=0)
    else:
        mean_pred = mean_pred[: eval_inputs.shape[0]]

    tracer = DualXDATracer(
        model=model,
        train_dataset=train_dataset,
        layer_name=infer_classifier_layer_name(model),
        device=str(device),
        cache_dir=str(run_dir / "cache" / "dualxda_notebook"),
    )

    attr_chunks: list[torch.Tensor] = []
    dist_chunks: dict[str, list[torch.Tensor]] = {
        k: [] for k in ("entropy", "participation", "signed_split", "variance")
    }
    for start in range(0, int(eval_inputs.shape[0]), batch_size):
        end = min(start + batch_size, int(eval_inputs.shape[0]))
        xb = eval_inputs[start:end].to(device)
        targets = mean_pred[start:end].argmax(dim=1).to(device)
        batch_attr = tracer.traces(x=xb, targets=targets, drop_zero_columns=False).cpu()
        dist = compute_attribution_distribution_signals(batch_attr)
        attr_chunks.append(batch_attr)
        for key in dist_chunks:
            dist_chunks[key].append(dist[key].cpu())

    attr = torch.cat(attr_chunks, dim=0)
    dist_out = {k: torch.cat(v, dim=0) for k, v in dist_chunks.items()}
    group_labels = group_labels_full[: attr.shape[0]].long()

    return {
        "attr": attr,
        "distribution": dist_out,
        "group_labels": group_labels,
        "config": config,
        "model": model,
        "train_dataset": train_dataset,
        "eval_inputs": eval_inputs,
        "mean_pred": mean_pred,
    }
