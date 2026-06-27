"""Validate, extract, and print fast-pilot experiment configuration.

Two config shapes appear in the training pipeline:

- ``ExperimentConfig`` — nested dataclass loaded from run YAML (source of truth at train time).
- ``RunConfigView`` — flat read-only view for logging / split validation (from ``extract_run_config``).

After ``prepare_experiment_data`` runs, ``ExperimentDataContext`` holds the loaded dataset and
index splits. ``apply_data_context`` copies dataset-derived fields back into ``RunConfigView``
(e.g. effective noise type after seeing the on-disk dataset).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from uqlab.data.dataset_registry import DatasetSpec, get_dataset_spec
from uqlab.data.benchmark_axes import (
    expects_aleatoric_eval,
    expects_epistemic_eval,
)
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.data.experiment_loader import SplitSpec
from uqlab.data.setup import ExperimentDataContext
from uqlab.shared.config.signals import flatten_signals, normalize_evaluation_signals, prune_signals_for_runtime, derive_attribution_backends_from_signals

logger = logging.getLogger(__name__)


@dataclass
class RunConfigView:
    """Flattened experiment parameters used by ``run_experiment_core``."""

    dataset_name: str
    dataset_spec: DatasetSpec
    noise_type: str
    effective_noise_type: str
    under_supported_classes: List[int]
    under_supported_classes_str: str
    under_train_per_class: int
    regular_train_per_class: int
    eval_per_group: int
    aleatoric_noise_percentage: Optional[float]

    architecture: str
    dinov2_model: str
    hidden_dim: int
    dropout: float

    epochs: int
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    feature_batch_size: int

    mc_passes: int
    top_k: int
    attribution_method: str
    attribution_backends: List[str]
    enabled_signals: List[str]

    aleatoric_expected: bool
    epistemic_expected: bool


def require_complete_config(config: ExperimentConfig) -> None:
    """Fail fast if any required config block is missing."""
    if (
        config.data is None
        or config.model is None
        or config.training is None
        or config.evaluation is None
        or config.paths is None
    ):
        raise ValueError(
            "ExperimentConfig is incomplete; data/model/training/evaluation/paths must be defined"
        )


def extract_run_config(config: ExperimentConfig) -> RunConfigView:
    """Parse nested ``ExperimentConfig`` into a flat view and validate MC passes."""
    require_complete_config(config)

    data = config.data
    model = config.model
    training = config.training
    evaluation = config.evaluation

    under_supported_classes = list(data.under_supported_classes or [])
    aleatoric_noise_percentage = data.aleatoric_noise_percentage
    noise_type = data.noise_type
    dataset_name = getattr(data, "dataset_name", None) or "cifar10"
    ds_spec = get_dataset_spec(dataset_name)

    effective_noise_type = noise_type
    if aleatoric_noise_percentage == 0:
        effective_noise_type = "clean_label"

    mc_passes = evaluation.mc_passes
    if mc_passes < 0:
        raise ValueError(
            f"mc_passes must be >= 0, got {mc_passes}. "
            "Set to 0 to disable MC Dropout (faster but no uncertainty), "
            "or use 5-10 for efficient uncertainty estimation (recommended: 10-50 for accuracy)."
        )
    if mc_passes == 0:
        logger.warning(
            "⚠️  MC Dropout disabled (mc_passes=0): No uncertainty quantification will be performed. "
            "This is faster but provides no epistemic uncertainty estimates. "
            "Consider using mc_passes=5-10 for efficient uncertainty estimation."
        )

    aleatoric_expected = expects_aleatoric_eval(aleatoric_noise_percentage)
    epistemic_expected = expects_epistemic_eval(
        under_supported_classes,
        under_train_per_class=data.under_train_per_class,
        regular_train_per_class=data.regular_train_per_class,
    )

    pruned_signals = prune_signals_for_runtime(
        normalize_evaluation_signals(evaluation.signals),
        mc_passes=mc_passes,
        dropout=float(model.dropout),
    )
    enabled_signals = flatten_signals(pruned_signals)
    derived_backends = list(derive_attribution_backends_from_signals(enabled_signals))

    return RunConfigView(
        dataset_name=dataset_name,
        dataset_spec=ds_spec,
        noise_type=noise_type,
        effective_noise_type=effective_noise_type,
        under_supported_classes=under_supported_classes,
        under_supported_classes_str=",".join(str(x) for x in under_supported_classes),
        under_train_per_class=data.under_train_per_class,
        regular_train_per_class=data.regular_train_per_class,
        eval_per_group=data.eval_per_group,
        aleatoric_noise_percentage=aleatoric_noise_percentage,
        architecture=model.architecture,
        dinov2_model=model.dinov2_model,
        hidden_dim=model.hidden_dim,
        dropout=model.dropout,
        epochs=training.epochs,
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        train_batch_size=training.train_batch_size,
        feature_batch_size=training.feature_batch_size,
        mc_passes=mc_passes,
        top_k=evaluation.top_k,
        attribution_method=getattr(evaluation, "attribution_method", None) or "dualxda",
        attribution_backends=derived_backends,
        enabled_signals=enabled_signals,
        aleatoric_expected=aleatoric_expected,
        epistemic_expected=epistemic_expected,
    )


def print_experiment_configuration(view: RunConfigView) -> None:
    """Print the pre-run configuration banner (stdout, for backend logs)."""
    print("\n" + "=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print("📊 Dataset Configuration:")
    print(f"   • Dataset: {view.dataset_name} ({view.dataset_spec.label})")
    print(f"   • Architecture: {view.architecture}")
    print(f"   • Noise type (saved): {view.noise_type}")
    print(f"   • Noise type (effective): {view.effective_noise_type}")
    print(f"   • Aleatoric noise: {view.aleatoric_noise_percentage}%")
    print(f"   • Under-supported classes: {view.under_supported_classes_str or 'None'}")
    print(f"   • Under-train per class: {view.under_train_per_class}")
    print(f"   • Regular-train per class: {view.regular_train_per_class}")
    print(f"   • Eval per group: {view.eval_per_group}")
    print("\n🧠 Model Configuration:")
    if view.architecture == "dinov2_mlp":
        print(f"   • DINOv2 model: {view.dinov2_model}")
    print(f"   • Hidden dim: {view.hidden_dim}")
    print(f"   • Dropout: {view.dropout}")
    print("\n🎯 Training Configuration:")
    print(f"   • Epochs: {view.epochs}")
    print(f"   • Learning rate: {view.learning_rate}")
    print(f"   • Weight decay: {view.weight_decay}")
    print(f"   • Train batch size: {view.train_batch_size}")
    print("\n📈 Evaluation Configuration:")
    print(f"   • MC passes: {view.mc_passes}")
    print(f"   • Top-k attribution: {view.top_k}")
    print(f"   • Attribution method: {view.attribution_method}")
    if view.attribution_backends:
        print(f"   • Attribution backends: {', '.join(view.attribution_backends)}")
    print(f"   • Enabled signals: {', '.join(view.enabled_signals)}")
    print("\n💡 Expected AUROC:")
    if view.aleatoric_expected:
        print("   ✅ Aleatoric AUROC — label-noise eval pool expected")
    else:
        print("   ℹ️  Aleatoric AUROC skipped (0% noise — normal for Fig. 3)")
    if view.epistemic_expected:
        print("   ✅ Epistemic AUROC — under-trained eval pool expected")
    else:
        print("   ℹ️  Epistemic AUROC skipped (balanced training — normal for Fig. 4)")
    print("=" * 80 + "\n")


def print_dataset_loaded(data_ctx: ExperimentDataContext, dataset: object) -> None:
    """Log dataset factory load summary."""
    print(f"\n🎯 Loaded {data_ctx.dataset_name} via dataset factory (root={data_ctx.data_root})")
    if getattr(dataset, "noise_mask", None) is not None:
        rate = float(getattr(dataset, "noise_rate", 0)) * 100
        print(f"   ✅ {len(dataset)} samples, noise rate {rate:.2f}%")
    else:
        print(f"   ✅ {len(dataset)} samples")


def validate_eval_splits(view: RunConfigView, split_spec: SplitSpec) -> None:
    """Warn or fail when expected benchmark pools are empty after splitting."""
    if len(split_spec.clean_eval_indices) == 0:
        logger.warning("⚠️  Clean evaluation group is empty — clean AUROC will be NaN.")

    if len(split_spec.aleatoric_eval_indices) == 0:
        if view.aleatoric_expected:
            raise RuntimeError(
                f"❌ Aleatoric benchmark requested ({view.aleatoric_noise_percentage}% label noise) "
                "but the aleatoric eval pool is empty. "
                "Check aleatoric_noise_percentage and noise_type=clean_label in config."
            )
        logger.info(
            "ℹ️  Aleatoric AUROC skipped (0% label noise — this is normal for Fig. 3 runs)."
        )

    if len(split_spec.epistemic_eval_indices) == 0:
        if view.epistemic_expected:
            logger.warning(
                "⚠️  Epistemic evaluation group is empty — epistemic AUROC will be NaN."
            )
        else:
            logger.info(
                "ℹ️  Epistemic AUROC skipped (balanced training — normal for Fig. 4 runs)."
            )


def apply_data_context(view: RunConfigView, data_ctx: ExperimentDataContext) -> RunConfigView:
    """Refresh derived fields from the data phase (dataset name, noise type)."""
    view.dataset_name = data_ctx.dataset_name
    view.dataset_spec = get_dataset_spec(data_ctx.dataset_name)
    view.effective_noise_type = data_ctx.effective_noise_type
    view.noise_type = data_ctx.noise_type
    view.under_supported_classes = list(data_ctx.under_supported_classes)
    view.under_supported_classes_str = ",".join(str(x) for x in view.under_supported_classes)
    return view
