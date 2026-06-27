"""
Run-time train/eval packs — one contract, two backends (embeddings | images).

Data stack (bottom → top):

1. **``loaders/*`` + ``dataset_registry``** — per-dataset disk I/O (CIFAR-10, MNIST, …)
2. **``setup.py``** — ``ExperimentConfig`` → loaded ``dataset`` + ``SplitSpec``
3. **This module** — ``SplitSpec`` → ``train_dataset`` + eval packs for the runner

Eval pack contract (every pack dict has these keys):

- ``features`` — model inputs (embeddings ``[N,D]`` or images ``[N,3,H,W]``)
- ``noisy_labels``, ``clean_labels``, ``is_noisy``, ``original_indices``

Paper ``fit`` data phase: :func:`prepare_run_data_context`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from uqlab.data.experiment_loader import SplitSpec
from uqlab.data.image_dataset import load_image_datasets
from uqlab.models.architecture import normalize_architecture
from uqlab.models.classification_models import EmbeddingDataset
from uqlab.models.feature_extractors import DINOv2FeatureExtractor, create_feature_extractor
from uqlab.run_artifacts import GROUP_ALEATORIC, GROUP_CLEAN, GROUP_EPISTEMIC, GROUP_OOD
from uqlab.shared.config.classification import ExperimentConfig

logger = logging.getLogger(__name__)

RunDataMode = Literal["embeddings", "images"]

EVAL_PACK_KEYS = frozenset(
    {"features", "noisy_labels", "clean_labels", "is_noisy", "original_indices"}
)


@dataclass(frozen=True)
class RunDataPacks:
    """Train subset + four eval pools + concatenated eval tensors."""

    train_dataset: Any
    clean_eval_pack: dict[str, torch.Tensor]
    aleatoric_eval_pack: dict[str, torch.Tensor]
    epistemic_eval_pack: dict[str, torch.Tensor]
    ood_eval_pack: dict[str, torch.Tensor]
    eval_data: dict[str, torch.Tensor]
    eval_inputs: torch.Tensor
    mode: RunDataMode
    feature_dim: int | None


def get_data_loading_mode(config: ExperimentConfig) -> RunDataMode:
    """Map ``ModelConfig.training_mode`` to ``embeddings`` or ``images``."""
    model_config = config.model
    if model_config is None:
        raise ValueError("ExperimentConfig.model must be set")

    if model_config.training_mode == "feature_space":
        return "embeddings"
    if model_config.training_mode == "end_to_end":
        return "images"
    raise ValueError(f"Unknown training mode: {model_config.training_mode}")


def resolve_run_data_mode(config: ExperimentConfig) -> RunDataMode:
    """
    Effective run mode after architecture-specific overrides.

    ResNet in ``feature_space`` config uses frozen backbone on images (no DINO cache).
    """
    mode = get_data_loading_mode(config)
    if normalize_architecture(config.model.architecture) == "resnet18" and mode == "embeddings":
        logger.info(
            "ResNet with feature_space mode: Using images with frozen backbone "
            "(ResNet doesn't support feature caching like DINOv2)"
        )
        return "images"
    return mode


def prepare_eval_tensors(
    clean_eval_pack: dict,
    aleatoric_eval_pack: dict,
    epistemic_eval_pack: dict,
    ood_eval_pack: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Concatenate eval packs into shared tensors (group labels, indices, inputs)."""
    packs: list[tuple[dict, int]] = [
        (clean_eval_pack, GROUP_CLEAN),
        (aleatoric_eval_pack, GROUP_ALEATORIC),
        (epistemic_eval_pack, GROUP_EPISTEMIC),
    ]
    if ood_eval_pack is not None and len(ood_eval_pack.get("features", [])) > 0:
        packs.append((ood_eval_pack, GROUP_OOD))

    eval_inputs = torch.cat([p["features"] for p, _ in packs], dim=0)
    eval_group_labels = torch.cat(
        [torch.full((len(p["features"]),), code, dtype=torch.long) for p, code in packs],
        dim=0,
    )
    eval_clean_labels = torch.cat([p["clean_labels"] for p, _ in packs], dim=0)
    eval_is_noisy = torch.cat([p["is_noisy"] for p, _ in packs], dim=0)
    eval_noisy_labels = torch.cat([p["noisy_labels"] for p, _ in packs], dim=0)
    eval_dataset_index = torch.cat([p["original_indices"] for p, _ in packs], dim=0)
    return {
        "eval_inputs": eval_inputs,
        "eval_group_labels": eval_group_labels,
        "eval_clean_labels": eval_clean_labels,
        "eval_is_noisy": eval_is_noisy,
        "eval_noisy_labels": eval_noisy_labels,
        "eval_dataset_index": eval_dataset_index,
    }


prepare_eval_data = prepare_eval_tensors


def _packs_from_embedding_extractor(
    feature_extractor: DINOv2FeatureExtractor,
) -> tuple[Any, dict, dict, dict, dict, int]:
    feature_extractor.organizer.load_or_compute_features()

    train_pack = feature_extractor.get_train_pack()
    clean_eval_pack = feature_extractor.get_clean_eval_pack()
    aleatoric_eval_pack = feature_extractor.get_aleatoric_eval_pack()
    epistemic_eval_pack = feature_extractor.get_epistemic_eval_pack()
    ood_eval_pack = feature_extractor.get_ood_eval_pack()

    train_dataset = EmbeddingDataset(
        train_pack["features"],
        train_pack["noisy_labels"],
        train_pack["clean_labels"],
        train_pack["is_noisy"],
        train_pack["original_indices"],
    )
    feature_dim = int(train_pack["features"].shape[1])
    return (
        train_dataset,
        clean_eval_pack,
        aleatoric_eval_pack,
        epistemic_eval_pack,
        ood_eval_pack,
        feature_dim,
    )


def _packs_from_images(
    dataset,
    split_spec: SplitSpec,
    *,
    dataset_name: str,
) -> tuple[Any, dict, dict, dict, dict]:
    train_dataset, eval_packs = load_image_datasets(
        dataset, split_spec, dataset_name=dataset_name
    )
    return (
        train_dataset,
        eval_packs["clean"],
        eval_packs["aleatoric"],
        eval_packs["epistemic"],
        eval_packs["ood"],
    )


def _finalize_run_packs(
    *,
    train_dataset,
    clean_eval_pack: dict,
    aleatoric_eval_pack: dict,
    epistemic_eval_pack: dict,
    ood_eval_pack: dict,
    mode: RunDataMode,
    feature_dim: int | None,
) -> RunDataPacks:
    eval_data = prepare_eval_tensors(
        clean_eval_pack,
        aleatoric_eval_pack,
        epistemic_eval_pack,
        ood_eval_pack,
    )
    return RunDataPacks(
        train_dataset=train_dataset,
        clean_eval_pack=clean_eval_pack,
        aleatoric_eval_pack=aleatoric_eval_pack,
        epistemic_eval_pack=epistemic_eval_pack,
        ood_eval_pack=ood_eval_pack,
        eval_data=eval_data,
        eval_inputs=eval_data["eval_inputs"],
        mode=mode,
        feature_dim=feature_dim,
    )


def prepare_run_data_context(
    *,
    config: ExperimentConfig,
    dataset,
    split_spec: SplitSpec,
    dataset_name: str,
    device: torch.device,
    feature_cache_dir: Path,
    noise_type: str,
    feature_batch_size: int,
    ds_spec=None,
) -> dict[str, Any]:
    """
    Build train/eval packs for one run (embeddings or images).

    Single entry for the runner after :func:`uqlab.data.setup.prepare_experiment_data`.
    Does **not** build or train the model.
    """
    del ds_spec  # kept for call-site compatibility with experiment_core

    if config.model is not None:
        from uqlab.models.backbones.dinov2_backbone import DINOv2Backbone

        normalized = DINOv2Backbone.normalize_model_name(config.model.dinov2_model)
        if normalized != config.model.dinov2_model:
            config.model.dinov2_model = normalized

    mode = resolve_run_data_mode(config)
    feature_dim: int | None = None

    if mode == "embeddings":
        feature_extractor = create_feature_extractor(
            config.model,
            device=device,
            dataset=dataset,
            split_spec=split_spec,
            feature_cache_dir=feature_cache_dir,
            noise_type=noise_type,
            batch_size=feature_batch_size,
        )
        if not isinstance(feature_extractor, DINOv2FeatureExtractor):
            raise TypeError("Expected DINOv2FeatureExtractor for feature_space mode")

        train_dataset, clean, alea, epis, ood, feature_dim = _packs_from_embedding_extractor(
            feature_extractor
        )
    elif mode == "images":
        train_dataset, clean, alea, epis, ood = _packs_from_images(
            dataset, split_spec, dataset_name=dataset_name
        )
    else:
        raise ValueError(f"Unsupported data loading mode: {mode}")

    packs = _finalize_run_packs(
        train_dataset=train_dataset,
        clean_eval_pack=clean,
        aleatoric_eval_pack=alea,
        epistemic_eval_pack=epis,
        ood_eval_pack=ood,
        mode=mode,
        feature_dim=feature_dim,
    )
    return {
        "train_dataset": packs.train_dataset,
        "clean_eval_pack": packs.clean_eval_pack,
        "aleatoric_eval_pack": packs.aleatoric_eval_pack,
        "epistemic_eval_pack": packs.epistemic_eval_pack,
        "ood_eval_pack": packs.ood_eval_pack,
        "eval_data": packs.eval_data,
        "eval_inputs": packs.eval_inputs,
        "mode": packs.mode,
        "feature_dim": packs.feature_dim,
    }


__all__ = [
    "EVAL_PACK_KEYS",
    "RunDataPacks",
    "get_data_loading_mode",
    "prepare_eval_data",
    "prepare_eval_tensors",
    "prepare_run_data_context",
    "resolve_run_data_mode",
]
