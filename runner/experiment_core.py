"""Fast-pilot experiment engine: train, evaluate signals, write artifacts.

Called only from :func:`uqlab.runner.pipeline.run` (or ``run_config``).
CLI wrappers live under ``scripts/runners/``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ============================================================================
# MODULAR IMPORTS: Explicit, specific imports from each module
# ============================================================================

# Data loading
from uqlab.data.classification_dataset import dataset_clean_labels
from uqlab.data.dataset_registry import get_dataset_spec, load_classification_dataset
from uqlab.data.preprocessing import get_dataset_image_transform
from uqlab.data.image_dataset import (
    ClassificationImageDataset,
    load_image_datasets,
)
from uqlab.evaluation.pipeline.experiment_setup import (
    apply_data_context,
    extract_run_config,
    print_dataset_loaded,
    print_experiment_configuration,
    validate_eval_splits,
)
from uqlab.models.architecture import normalize_architecture
from uqlab.evaluation.pipeline.data_setup import prepare_experiment_data

# Uncertainty metrics (used in run_experiment_core via experiment_eval)
# UQ Classification package - Constants
GROUP_CLEAN = 0
GROUP_ALEATORIC = 1
GROUP_EPISTEMIC = 2
GROUP_OOD = 3
GROUP_NAMES: dict[int, str] = {
    GROUP_CLEAN: "clean",
    GROUP_ALEATORIC: "aleatoric_like",
    GROUP_EPISTEMIC: "epistemic_like",
    GROUP_OOD: "ood_like",
}

# UQ Classification package - Models
from uqlab.models.classification_models import EmbeddingDataset
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.models.factory import build_model
from uqlab.models.feature_extractors import create_feature_extractor, DINOv2FeatureExtractor

# UQ Classification package - Utils
from uqlab.shared.utils.classification import auto_device, dino_transform, set_seed

# UQ Classification package - Data loading
from uqlab.data.experiment_loader import (
    EmbeddingOrganizer,
    SplitSpec,
    sample_indices_for_experiment,
)

from uqlab.evaluation.signals.formulas import build_signal_formula_manifest

# UQ Classification package - Evaluation
from uqlab.evaluation.result_writers import (
    persist_experiment_summaries,
    save_training_data_csv,
)
from uqlab.evaluation.metrics import (
    split_group_balanced_targets,
    train_signal_classifier,
)


def get_data_loading_mode(config: ExperimentConfig) -> str:
    """Determine data loading mode from config."""
    model_config = config.model
    if model_config is None:
        raise ValueError("ExperimentConfig.model must be set")

    if model_config.training_mode == "feature_space":
        return "embeddings"
    if model_config.training_mode == "end_to_end":
        return "images"
    raise ValueError(f"Unknown training mode: {model_config.training_mode}")


def prepare_eval_data(
    clean_eval_pack: dict[str, torch.Tensor],
    aleatoric_eval_pack: dict[str, torch.Tensor],
    epistemic_eval_pack: dict[str, torch.Tensor],
    ood_eval_pack: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Concatenate eval packs and build shared eval metadata tensors."""
    packs: list[tuple[dict[str, torch.Tensor], int]] = [
        (clean_eval_pack, GROUP_CLEAN),
        (aleatoric_eval_pack, GROUP_ALEATORIC),
        (epistemic_eval_pack, GROUP_EPISTEMIC),
    ]
    if ood_eval_pack is not None and len(ood_eval_pack.get("features", [])) > 0:
        packs.append((ood_eval_pack, GROUP_OOD))

    eval_inputs = torch.cat([p["features"] for p, _ in packs], dim=0)
    eval_group_labels = torch.cat(
        [
            torch.full((len(p["features"]),), code, dtype=torch.long)
            for p, code in packs
        ],
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


from uqlab.evaluation.pipeline.experiment_eval import (
    EvalSignalConfig,
    collect_uncertainty_signals,
    score_uncertainty_signals,
)
from uqlab.run_artifacts import save_zwischen_result

def _format_auroc_console(value: object, skip_reason: str | None) -> str:
    if value is None:
        if skip_reason:
            return f"— (skipped: {skip_reason.replace('_', ' ')})"
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _unpack_auroc_row(row: tuple) -> tuple[str, Any, Any, Any | None]:
    name, alea, epis = row[0], row[1], row[2]
    ood = row[3] if len(row) > 3 else None
    return name, alea, epis, ood


def _prepare_eval_data_and_model(
    *,
    config: ExperimentConfig,
    dataset,
    split_spec: SplitSpec,
    ds_spec,
    dataset_name: str,
    device: torch.device,
    feature_cache_dir: Path,
    noise_type: str,
    feature_batch_size: int,
    epochs: int,
    results_dir: Path,
) -> dict[str, Any]:
    """Data-loading phase: feature/image packs, model build, optional checkpoint resume."""
    mode = get_data_loading_mode(config)

    # IMPORTANT: ResNet in feature_space mode works differently than DINOv2:
    # - DINOv2: Uses pre-computed cached features (embeddings mode)
    # - ResNet: Uses images directly but with frozen backbone (images mode)
    # Both achieve "feature space" training, but through different mechanisms
    if normalize_architecture(config.model.architecture) == "resnet18" and mode == "embeddings":
        logger.info(
            "ResNet with feature_space mode: Using images with frozen backbone "
            "(ResNet doesn't support feature caching like DINOv2)"
        )
        mode = "images"  # Use images, but model will have freeze_backbone=True

    feature_dim: int | None = None

    if mode == "embeddings":
        # Only DINOv2 supports embeddings mode with feature caching
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
        eval_data = prepare_eval_data(
            clean_eval_pack,
            aleatoric_eval_pack,
            epistemic_eval_pack,
            ood_eval_pack,
        )
        eval_inputs = eval_data["eval_inputs"]
        feature_dim = int(train_pack["features"].shape[1])
    elif mode == "images":
        train_dataset, eval_packs = load_image_datasets(
            dataset, split_spec, dataset_name=dataset_name
        )
        clean_eval_pack = eval_packs["clean"]
        aleatoric_eval_pack = eval_packs["aleatoric"]
        epistemic_eval_pack = eval_packs["epistemic"]
        ood_eval_pack = eval_packs["ood"]
        eval_data = prepare_eval_data(
            clean_eval_pack,
            aleatoric_eval_pack,
            epistemic_eval_pack,
            ood_eval_pack,
        )
        eval_inputs = eval_data["eval_inputs"]
    else:
        raise ValueError(f"Unsupported data loading mode: {mode}")

    model = build_model(
        config=config.model,
        num_classes=ds_spec.num_classes,
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

    print(f"Using device: {device}")
    print(f"Results directory: {results_dir}")
    print(f"Train samples: {len(train_dataset)}")
    print(
        "Eval groups: "
        f"clean={len(clean_eval_pack['features'])}, "
        f"aleatoric_like={len(aleatoric_eval_pack['features'])}, "
        f"epistemic_like={len(epistemic_eval_pack['features'])}, "
        f"ood_like={len(ood_eval_pack['features']) if ood_eval_pack is not None else 0}"
    )

    return {
        "model": model,
        "train_dataset": train_dataset,
        "clean_eval_pack": clean_eval_pack,
        "aleatoric_eval_pack": aleatoric_eval_pack,
        "epistemic_eval_pack": epistemic_eval_pack,
        "ood_eval_pack": ood_eval_pack,
        "eval_data": eval_data,
        "eval_inputs": eval_inputs,
        "mode": mode,
        "prior_epoch_loaded": prior_epoch_loaded,
    }


def _build_run_summary(
    *,
    config_path: Path | None,
    seed: int,
    device: torch.device,
    data_config,
    model_config,
    training_config,
    eval_config,
    split_spec: SplitSpec,
    train_dataset: Dataset,
    clean_eval_pack: dict[str, torch.Tensor],
    aleatoric_eval_pack: dict[str, torch.Tensor],
    epistemic_eval_pack: dict[str, torch.Tensor],
    ood_eval_pack: dict[str, torch.Tensor] | None,
    eval_per_group: int,
    top_k: int,
    mc_passes: int,
    one_vs_rest_auroc: list[dict],
    auroc_rows: list[tuple],
    clf_rows: list[tuple[str, float]],
) -> tuple[dict, dict]:
    """Summary assembly phase: eval protocol, signal formulas, and summary dict."""
    eval_protocol = {
        "architecture_invariant": True,
        "rationale": (
            "Eval indices sampled from CIFAR-10N pools before training; "
            "all architectures at same sweep point use same seed/eval_per_group/under_supported_classes "
            "(fixed test set, varying train UQ method - same as uq_disentanglement design)."
        ),
        "eval_per_group": eval_per_group,
        "groups": list(GROUP_NAMES.values()),
        "under_supported_classes": list(split_spec.under_supported_classes),
        "seed": seed,
    }

    signal_formulas = build_signal_formula_manifest(
        top_k=top_k,
        mc_passes=mc_passes,
        eval_protocol=eval_protocol,
    )

    summary = {
        "config": {
            "config_file": str(config_path),
            "seed": seed,
            "device": str(device),
            "data": vars(data_config),
            "model": model_config.dict(),
            "training": vars(training_config),
            "evaluation": vars(eval_config),
        },
        "under_supported_classes": split_spec.under_supported_classes,
        "train_size": len(train_dataset),
        "eval_sizes": {
            "clean": len(clean_eval_pack["clean_labels"]),
            "aleatoric_like": len(aleatoric_eval_pack["clean_labels"]),
            "epistemic_like": len(epistemic_eval_pack["clean_labels"]),
            "ood_like": len(ood_eval_pack["clean_labels"]) if ood_eval_pack is not None else 0,
        },
        "eval_protocol": eval_protocol,
        "signal_formulas": signal_formulas,
        "dualxda_svm": {"max_iter": 1_000_000},
        "one_vs_rest_auroc": one_vs_rest_auroc,
        "auroc_rows": [
            {
                "signal": row[0],
                "aleatoric_auroc": row[1],
                "epistemic_auroc": row[2],
                **({"ood_auroc": row[3]} if len(row) > 3 and row[3] is not None else {}),
            }
            for row in auroc_rows
        ],
        "macro_f1": [
            {
                "signal_set": name,
                "macro_f1": score,
            }
            for name, score in clf_rows
        ],
    }
    return summary, signal_formulas


def _export_run_results(
    results_dir: Path,
    *,
    uq: dict,
    mean_pred_det: torch.Tensor | None,
    train_dataset: Dataset,
    eval_inputs: torch.Tensor,
    mode: str,
    eval_clean_labels: torch.Tensor,
    eval_is_noisy: torch.Tensor,
    eval_group_labels: torch.Tensor,
    clean_eval_pack: dict[str, torch.Tensor],
    aleatoric_eval_pack: dict[str, torch.Tensor],
    epistemic_eval_pack: dict[str, torch.Tensor],
    ood_eval_pack: dict[str, torch.Tensor] | None,
    signal_table: dict,
    auroc_rows: list[tuple],
) -> None:
    """Artifact export phase: write results.pt for sweep pools and downstream consumers."""
    mean_for_results = uq.get("mean_prediction") if uq else mean_pred_det
    if mean_for_results is None:
        raise TypeError("collect_uncertainty_signals() returned no mean predictions for results export")

    eval_packs_for_export = [clean_eval_pack, aleatoric_eval_pack, epistemic_eval_pack]
    if ood_eval_pack is not None and len(ood_eval_pack.get("features", [])) > 0:
        eval_packs_for_export.append(ood_eval_pack)

    results_data = {
        "predictions": mean_for_results.argmax(dim=1),
        "confidences": mean_for_results.max(dim=1).values,
        "mean_prediction_deterministic": mean_pred_det,
        "train_embeddings": getattr(train_dataset, "features", None),
        "train_images": eval_inputs.new_empty((0,)) if not hasattr(train_dataset, "features") else None,
        "train_labels": train_dataset.clean_labels,
        "train_noisy_labels": train_dataset.targets,
        "train_is_noisy": train_dataset.is_noisy,
        "train_indices": train_dataset.original_indices,
        "eval_embeddings": eval_inputs if mode == "embeddings" else None,
        "eval_images": eval_inputs if mode == "images" else None,
        "eval_clean_labels": eval_clean_labels,
        "eval_noisy_labels": torch.cat(
            [p["noisy_labels"] for p in eval_packs_for_export], dim=0
        ),
        "eval_is_noisy": eval_is_noisy,
        "eval_group_labels": eval_group_labels,
        "eval_indices": torch.cat(
            [p["original_indices"] for p in eval_packs_for_export], dim=0
        ),
        "signal_table": signal_table,
        "auroc_rows": auroc_rows,
    }
    try:
        torch.save(results_data, results_dir / "results.pt")
        print(f"✅ Saved results data to {results_dir / 'results.pt'}")
    except Exception as exc:
        print(f"⚠️ results.pt save failed (summary.json is available): {exc}")


def _print_auroc_tables(
    *,
    auroc_rows: list[tuple],
    clf_rows: list[tuple[str, float]],
    alea_skip: str | None,
    epis_skip: str | None,
    ood_skip: str | None,
    eval_group_labels: torch.Tensor,
) -> None:
    """Console reporting phase: AUROC tables grouped by signal family."""
    from uqlab.evaluation.signals.registry import METRICS

    def _auroc_rows_for_family(family: str) -> list[tuple[str, Any, Any, Any | None]]:
        ids = {mid for mid, metric in METRICS.items() if metric.family == family}
        return [_unpack_auroc_row(row) for row in auroc_rows if row[0] in ids]

    print("\n" + "=" * 70)
    print("ATTRIBUTION-BASED SIGNALS (DualXDA / EK-FAC)")
    print("=" * 70)
    attr_rows = _auroc_rows_for_family("attribution")
    for name, alea_auc, epis_auc, ood_auc in sorted(
        attr_rows,
        key=lambda row: max(v for v in row[1:] if v is not None) if any(v is not None for v in row[1:]) else 0,
        reverse=True,
    ):
        ood_part = (
            f", ood={_format_auroc_console(ood_auc, ood_skip)}"
            if ood_auc is not None
            else ""
        )
        print(
            f"  {name:<30} aleatoric={_format_auroc_console(alea_auc, alea_skip)}, "
            f"epistemic={_format_auroc_console(epis_auc, epis_skip)}{ood_part}"
        )

    print("\n" + "=" * 70)
    print("LOGIT-BASED SIGNALS (via Representer Theorem)")
    print("=" * 70)
    logit_rows = _auroc_rows_for_family("logit")
    for name, alea_auc, epis_auc, ood_auc in sorted(
        logit_rows,
        key=lambda row: max(v for v in row[1:] if v is not None) if any(v is not None for v in row[1:]) else 0,
        reverse=True,
    ):
        ood_part = (
            f", ood={_format_auroc_console(ood_auc, ood_skip)}"
            if ood_auc is not None
            else ""
        )
        print(
            f"  {name:<30} aleatoric={_format_auroc_console(alea_auc, alea_skip)}, "
            f"epistemic={_format_auroc_console(epis_auc, epis_skip)}{ood_part}"
        )

    print("\n" + "=" * 70)
    print("PREDICTIVE UNCERTAINTY BASELINE")
    print("=" * 70)
    pred_rows = _auroc_rows_for_family("predictive")
    for name, alea_auc, epis_auc, ood_auc in pred_rows:
        ood_part = (
            f", ood={_format_auroc_console(ood_auc, ood_skip)}"
            if ood_auc is not None
            else ""
        )
        print(
            f"  {name:<30} aleatoric={_format_auroc_console(alea_auc, alea_skip)}, "
            f"epistemic={_format_auroc_console(epis_auc, epis_skip)}{ood_part}"
        )

    num_groups = int(eval_group_labels.max().item()) + 1 if len(eval_group_labels) else 3
    print(f"\n{num_groups}-way macro-F1:")
    for name, score in clf_rows:
        print(f"  {name}: {score:.4f}")


def train_feature_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    training_config,
    device: torch.device,
) -> torch.nn.Module:
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
        num_workers=4,
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


def run_experiment_core(
    config: ExperimentConfig,
    results_dir: Path,
    *,
    seed: int,
    device_str: str,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict:
    """Shared fast-pilot pipeline (CLI, facade, backend DirectExecutor).

    Config path: UI workflow → YAML → ``ExperimentConfig`` → ``RunConfigView`` (log/validate)
    → ``prepare_experiment_data`` → ``ExperimentDataContext`` (dataset + splits).
    """
    from uqlab.runtime_paths import repository_root

    root = project_root if project_root is not None else repository_root()

    run_cfg = extract_run_config(config)
    print_experiment_configuration(run_cfg)

    set_seed(seed)
    device = auto_device(device_str)

    feature_cache_dir = root / config.paths.feature_cache_dir
    run_cache_dir = results_dir / "cache"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_config = config.data
    model_config = config.model
    training_config = config.training
    eval_config = config.evaluation

    data_ctx = prepare_experiment_data(config, root, seed=seed)
    apply_data_context(run_cfg, data_ctx)
    dataset = data_ctx.dataset
    split_spec = data_ctx.split_spec
    ds_spec = run_cfg.dataset_spec
    dataset_name = run_cfg.dataset_name

    print_dataset_loaded(data_ctx, dataset)
    validate_eval_splits(run_cfg, split_spec)

    # Shorthand aliases used throughout train/eval (from run_cfg after data phase)
    noise_type = run_cfg.noise_type
    under_supported_classes = run_cfg.under_supported_classes
    under_supported_classes_str = run_cfg.under_supported_classes_str
    under_train_per_class = run_cfg.under_train_per_class
    regular_train_per_class = run_cfg.regular_train_per_class
    eval_per_group = run_cfg.eval_per_group
    aleatoric_noise_percentage = run_cfg.aleatoric_noise_percentage
    epochs = run_cfg.epochs
    learning_rate = run_cfg.learning_rate
    weight_decay = run_cfg.weight_decay
    train_batch_size = run_cfg.train_batch_size
    feature_batch_size = run_cfg.feature_batch_size
    mc_passes = run_cfg.mc_passes
    top_k = run_cfg.top_k
    attribution_method = run_cfg.attribution_method
    enabled_signals = set(run_cfg.enabled_signals)
    aleatoric_expected = run_cfg.aleatoric_expected
    epistemic_expected = run_cfg.epistemic_expected
    dinov2_model = run_cfg.dinov2_model
    hidden_dim = run_cfg.hidden_dim
    dropout = run_cfg.dropout

    packs_ctx = _prepare_eval_data_and_model(
        config=config,
        dataset=dataset,
        split_spec=split_spec,
        ds_spec=ds_spec,
        dataset_name=dataset_name,
        device=device,
        feature_cache_dir=feature_cache_dir,
        noise_type=noise_type,
        feature_batch_size=feature_batch_size,
        epochs=epochs,
        results_dir=results_dir,
    )
    model = packs_ctx["model"]
    train_dataset = packs_ctx["train_dataset"]
    clean_eval_pack = packs_ctx["clean_eval_pack"]
    aleatoric_eval_pack = packs_ctx["aleatoric_eval_pack"]
    epistemic_eval_pack = packs_ctx["epistemic_eval_pack"]
    ood_eval_pack = packs_ctx["ood_eval_pack"]
    eval_data = packs_ctx["eval_data"]
    eval_inputs = packs_ctx["eval_inputs"]
    mode = packs_ctx["mode"]
    prior_epoch_loaded = packs_ctx["prior_epoch_loaded"]

    if mode == "embeddings":
        model = train_feature_model(model, train_dataset, training_config, device)
    elif mode == "images":
        model = train_image_model(model, train_dataset, training_config, device)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")

    model = model.to(device)
    model.eval()

    # ============================================================================
    # POST-TRAINING PHASE: Save Training Statistics & Prepare Evaluation
    # ============================================================================
    # This corresponds to Progressive UI Step 4: After model training completes,
    # we save training data statistics and prepare for uncertainty evaluation.
    
    # Serialize config for CSV export (convert Path/Pydantic objects to JSON-safe types)
    from dataclasses import asdict
    config_dict = asdict(config)
    if config_dict.get('paths'):
        config_dict['paths'] = {k: str(v) if isinstance(v, Path) else v
                                for k, v in config_dict['paths'].items()}
    if config_dict.get('model'):
        config_dict['model'] = dict(config.model)
    
    # Save training data distribution to CSV (includes class balance, noise rates)
    save_training_data_csv(
        output_path=results_dir / "training_data.csv",
        train_dataset=train_dataset,
        config=config_dict,
    )

    # Extract evaluation metadata (ground truth labels, noise indicators, dataset indices)
    eval_group_labels = eval_data["eval_group_labels"]      # Which UQ group: clean/aleatoric/epistemic
    eval_clean_labels = eval_data["eval_clean_labels"]      # True CIFAR-10 labels
    eval_is_noisy = eval_data["eval_is_noisy"]              # Boolean: has label noise?
    eval_noisy_labels = eval_data["eval_noisy_labels"]      # Noisy labels (if applicable)
    eval_dataset_index = eval_data["eval_dataset_index"]    # Original CIFAR-10N indices

    # Save evaluation setup as intermediate result (for debugging/reproducibility)
    save_zwischen_result(
        results_dir, "00_eval_setup",
        {
            "eval_group_labels": eval_group_labels.cpu(),
            "eval_clean_labels": eval_clean_labels.cpu(),
            "eval_is_noisy": eval_is_noisy.cpu(),
            "eval_noisy_labels": eval_noisy_labels.cpu(),
            "eval_dataset_index": eval_dataset_index.cpu(),
            "n_eval": int(eval_inputs.shape[0]),
            "mc_passes": mc_passes,
        },
    )

    # ============================================================================
    # UNCERTAINTY SIGNAL COMPUTATION PHASE
    # ============================================================================
    # This corresponds to Progressive UI Step 5: Compute uncertainty signals
    # (epistemic, aleatoric, attribution-based) using the trained model.
    
    # Compute all enabled uncertainty signals (MC dropout, attribution, etc.)
    eval_signal_config = EvalSignalConfig.from_run_config(
        run_cfg,
        results_dir=results_dir,
        run_cache_dir=run_cache_dir,
    )
    eval_outputs = collect_uncertainty_signals(
        model=model,
        train_dataset=train_dataset,
        eval_inputs=eval_inputs,
        device=device,
        config=eval_signal_config,
    )
    
    # Extract computed signals and validate output structure
    det_logits = eval_outputs.get("det_logits")
    mean_pred_det = eval_outputs.get("mean_pred_det")
    uq = eval_outputs.get("uq") or {}
    signal_table = eval_outputs["signal_table"]
    if signal_table is None or not isinstance(signal_table, dict):
        raise TypeError("collect_uncertainty_signals() returned invalid `signal_table` payload")
    print(f"✅ Zwischenergebnisse: {results_dir / 'zwischen'}/")

    # NOTE: Some signals removed due to poor AUROC performance (see AUROC_METRICS_EXPLAINED.md)
    # - attribution_concentration, label_disagreement, compound_uncertainty: ~0.50 AUROC (random)
    # - noisy_support_ratio, inverse_max_logit: redundant with better-performing signals

    # ============================================================================
    # SIGNAL EVALUATION & AUROC COMPUTATION PHASE
    # ============================================================================
    # This corresponds to Progressive UI Step 6: Evaluate how well each signal
    # distinguishes clean vs. aleatoric vs. epistemic samples using AUROC.
    
    # Compute AUROC scores for all signals (one-vs-rest classification)
    eval_summary = score_uncertainty_signals(
        signal_table=signal_table,
        eval_group_labels=eval_group_labels,
        eval_clean_labels=eval_clean_labels,
        eval_is_noisy=eval_is_noisy,
        eval_noisy_labels=eval_noisy_labels,
        eval_dataset_index=eval_dataset_index,
        results_dir=results_dir,
        device=device,
        seed=seed,
    )
    
    # Extract AUROC results and check for skipped evaluations
    auroc_rows = eval_summary["auroc_rows"]
    one_vs_rest_auroc = eval_summary["one_vs_rest_auroc"]
    clf_rows = eval_summary["clf_rows"]
    alea_skip = one_vs_rest_auroc[0].get("aleatoric_skip_reason") if one_vs_rest_auroc else None
    epis_skip = one_vs_rest_auroc[0].get("epistemic_skip_reason") if one_vs_rest_auroc else None

    ood_skip = one_vs_rest_auroc[0].get("ood_skip_reason") if one_vs_rest_auroc else None

    summary, signal_formulas = _build_run_summary(
        config_path=config_path,
        seed=seed,
        device=device,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        eval_config=eval_config,
        split_spec=split_spec,
        train_dataset=train_dataset,
        clean_eval_pack=clean_eval_pack,
        aleatoric_eval_pack=aleatoric_eval_pack,
        epistemic_eval_pack=epistemic_eval_pack,
        ood_eval_pack=ood_eval_pack,
        eval_per_group=eval_per_group,
        top_k=top_k,
        mc_passes=mc_passes,
        one_vs_rest_auroc=one_vs_rest_auroc,
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
    )

    config_ns = argparse.Namespace(
        noise_type=noise_type,
        under_supported_classes=under_supported_classes_str,
        under_train_per_class=under_train_per_class,
        regular_train_per_class=regular_train_per_class,
        eval_per_group=eval_per_group,
        dinov2_model=dinov2_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_batch_size=train_batch_size,
        feature_batch_size=feature_batch_size,
        mc_passes=mc_passes,
        top_k=top_k,
        seed=seed,
        device=device_str,
    )

    persist_experiment_summaries(
        results_dir,
        summary=summary,
        args=config_ns,
        split_spec=split_spec,
        train_size=len(train_dataset),
        eval_sizes=summary["eval_sizes"],
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
    )

    from uqlab.run_artifacts import save_signal_formula_manifest

    save_signal_formula_manifest(results_dir, signal_formulas)

    # Save model checkpoint for watsonx.ai export
    # Remove any forward hooks before saving (DualXDA adds hooks that can't be pickled)
    model.eval()
    # Remove hooks from all modules
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
    
    checkpoint = {
        'model': model,
        'model_state_dict': model.state_dict(),
        'epoch': prior_epoch_loaded + epochs,
        'loss': 0.0,
        'config': {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'num_classes': ds_spec.num_classes,
            'dinov2_model': dinov2_model,
        }
    }
    try:
        torch.save(checkpoint, results_dir / "checkpoint.pt")
        print(f"✅ Saved model checkpoint to {results_dir / 'checkpoint.pt'}")
    except Exception as exc:
        print(f"⚠️ Checkpoint save failed (training metrics still saved): {exc}")

    _export_run_results(
        results_dir,
        uq=uq,
        mean_pred_det=mean_pred_det,
        train_dataset=train_dataset,
        eval_inputs=eval_inputs,
        mode=mode,
        eval_clean_labels=eval_clean_labels,
        eval_is_noisy=eval_is_noisy,
        eval_group_labels=eval_group_labels,
        clean_eval_pack=clean_eval_pack,
        aleatoric_eval_pack=aleatoric_eval_pack,
        epistemic_eval_pack=epistemic_eval_pack,
        ood_eval_pack=ood_eval_pack,
        signal_table=signal_table,
        auroc_rows=auroc_rows,
    )

    try:
        from uqlab.run_artifacts import metrics_row_from_run, print_run_metrics_summary

        print("\n" + "=" * 70)
        print("SIGNAL MEANS & AUROC (all uncertainties)")
        print("=" * 70)
        print_run_metrics_summary(metrics_row_from_run(results_dir))
    except ImportError:
        pass

    _print_auroc_tables(
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
        alea_skip=alea_skip,
        epis_skip=epis_skip,
        ood_skip=ood_skip,
        eval_group_labels=eval_group_labels,
    )

    print(f"\nSaved per-sample signals to: {results_dir / 'per_sample_signals.csv'}")
    print(f"Saved summary to: {results_dir / 'summary.json'} and {results_dir / 'summary.md'}")
    return summary


__all__ = [
    "GROUP_ALEATORIC",
    "GROUP_CLEAN",
    "GROUP_EPISTEMIC",
    "GROUP_NAMES",
    "get_data_loading_mode",
    "prepare_eval_data",
    "run_experiment_core",
    "train_feature_model",
    "train_image_model",
]
