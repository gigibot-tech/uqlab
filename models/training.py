"""Training loops and model construction for the fast-pilot runner (paper ``fit``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from uqlab.models.factory import build_model
from uqlab.models.feature_extractors import create_feature_extractor
from uqlab.shared.config.classification import ExperimentConfig

logger = logging.getLogger(__name__)


def train_feature_model(
    model: nn.Module,
    train_dataset: Dataset,
    training_config,
    device: torch.device,
) -> nn.Module:
    """Train a model on embedding datasets."""
    from torch.utils.data import DataLoader

    loader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    for epoch in range(training_config.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % max(1, training_config.epochs // 3) == 0 or epoch == training_config.epochs - 1:
            print(
                f"  Epoch {epoch + 1}/{training_config.epochs}, "
                f"loss={total_loss / max(1, len(loader)):.4f}"
            )

    return model


def train_image_model(
    model: nn.Module,
    train_dataset: Dataset,
    training_config,
    device: torch.device,
) -> nn.Module:
    """Train model end-to-end on images."""
    from torch.utils.data import DataLoader
    import torch.optim as optim

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.train_batch_size,
        shuffle=True,
        num_workers=0,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()
    for epoch in range(training_config.epochs):
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(
            f"Epoch {epoch + 1}/{training_config.epochs}, "
            f"Loss: {total_loss / max(1, len(train_loader)):.4f}"
        )

    return model


def build_model_for_run(
    *,
    config: ExperimentConfig,
    num_classes: int,
    feature_dim: int | None,
    mode: str,
    device: torch.device,
    feature_batch_size: int,
    epochs: int,
) -> tuple[nn.Module, int]:
    """
    Build the classifier and optionally resume from ``config.model.checkpoint_path``.

    Returns ``(model, prior_epoch_loaded)``.
    """
    model = build_model(
        config=config.model,
        num_classes=num_classes,
        feature_dim=feature_dim if mode == "embeddings" else None,
    )
    model = model.to(device)

    prior_epoch_loaded = 0
    checkpoint_path = getattr(config.model, "checkpoint_path", None) or (
        config.model.get("checkpoint_path") if isinstance(config.model, dict) else None
    )
    if checkpoint_path:
        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            print(f"🔁 Loading checkpoint: {ckpt_file}")
            checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
            state = checkpoint.get("model_state_dict")
            if state:
                model.load_state_dict(state, strict=False)
                print(f"   ✅ Loaded model_state_dict ({len(state)} tensors)")
            elif checkpoint.get("model") is not None:
                print("   ⚠️  Full model object in checkpoint — using state_dict only when available")
            prior_epoch_loaded = int(checkpoint.get("epoch") or 0)
            print(f"   Prior training: {prior_epoch_loaded} epoch(s) → training {epochs} more")
        else:
            print(f"⚠️  checkpoint_path set but file missing: {ckpt_file}")

    if mode == "images":
        create_feature_extractor(
            config.model,
            device=device,
            model=model,
            batch_size=feature_batch_size,
        )

    return model, prior_epoch_loaded


__all__ = [
    "build_model_for_run",
    "train_feature_model",
    "train_image_model",
]
