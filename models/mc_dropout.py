"""MC Dropout forward passes for trainable PyTorch models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _mc_forward_one_chunk(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int,
) -> torch.Tensor:
    """
    MC passes on one sample chunk [B, ...] → [T, B, C].

    CNN/ResNet: backbone features computed once per chunk; only dropout+head repeat.
    MLP: T lightweight forwards (dropout is the main cost).
    """
    if hasattr(model, "enable_dropout"):
        model.eval()
        model.enable_dropout()

    if hasattr(model, "extract_features") and hasattr(model, "classifier"):
        features = model.extract_features(x)
        dropout = getattr(model, "dropout", None)
        if dropout is None:
            raise AttributeError("Model has extract_features but no dropout module")
        preds = []
        for _ in range(n_passes):
            h = dropout(features)
            logits = model.classifier(h)
            preds.append(F.softmax(logits, dim=1))
        return torch.stack(preds, dim=0)

    preds = []
    for _ in range(n_passes):
        logits = model(x)
        preds.append(F.softmax(logits, dim=1))
    return torch.stack(preds, dim=0)


@torch.no_grad()
def mc_forward_efficient(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int,
    *,
    sample_batch_size: int = 256,
) -> torch.Tensor:
    """
    Batched MC Dropout over eval samples.

    Chunks the eval set along batch dimension to limit memory; reuses CNN/ResNet
    backbone features within each chunk (see :func:`_mc_forward_one_chunk`).
    """
    if n_passes < 1:
        raise ValueError(f"n_passes must be >= 1, got {n_passes}")
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("empty eval tensor")
    chunks: list[torch.Tensor] = []
    for start in range(0, n, sample_batch_size):
        end = min(start + sample_batch_size, n)
        chunks.append(_mc_forward_one_chunk(model, x[start:end], n_passes))
    return torch.cat(chunks, dim=1)


def batch_mc_dropout_uncertainty(model, dataloader, n_passes=50, device="cuda"):
    """
    Calculate MC Dropout uncertainty for entire dataset.

    Args:
        model: MC Dropout model
        dataloader: Data loader
        n_passes: Number of MC forward passes
        device: Device to run on

    Returns:
        all_uncertainties: Dictionary of uncertainty metrics
        all_labels: Ground truth labels
        all_predictions: Mean predictions
    """
    from uqlab.evaluation.signals.mc_dropout import (
        calculate_mc_dropout_uncertainty,
        calculate_sirc_score,
    )

    model.eval()

    all_mean_variance = []
    all_entropy = []
    all_mutual_info = []
    all_sirc_score = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for _batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)

            mc_predictions = model.mc_forward(data, n_passes=n_passes)

            uncertainties = calculate_mc_dropout_uncertainty(mc_predictions)
            sirc = calculate_sirc_score(mc_predictions)

            all_mean_variance.append(uncertainties["mean_variance"].cpu())
            all_entropy.append(uncertainties["entropy"].cpu())
            all_mutual_info.append(uncertainties["mutual_info"].cpu())
            all_sirc_score.append(sirc.cpu())
            all_labels.append(target)
            all_predictions.append(uncertainties["mean_prediction"].cpu())

    return {
        "mean_variance": torch.cat(all_mean_variance),
        "entropy": torch.cat(all_entropy),
        "mutual_info": torch.cat(all_mutual_info),
        "sirc_score": torch.cat(all_sirc_score),
        "labels": torch.cat(all_labels),
        "predictions": torch.cat(all_predictions),
    }
