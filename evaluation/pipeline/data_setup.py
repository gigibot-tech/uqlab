"""
Load datasets and build train/eval index splits for the fast uncertainty pilot.

Config flow (Streamlit → training script)
-----------------------------------------
1. **Progressive UI** (`streamlit_app_progressive.py`) stores choices in
   ``st.session_state.workflow`` — four dicts keyed by step:

   - ``dataset_config``  ← Step 1 (dataset, noise_type, stats)
   - ``training_config``   ← Step 2 (architecture, epochs, lr, …)
   - ``uncertainty_config``← Step 3 (epistemic/aleatoric, sweep grid)
   - ``evaluation_config`` ← Step 4 (eval_per_group, mc_passes, signals)

2. **Run spec** (`uqlab_orchestrator.run_spec.build_run_yaml`) merges those dicts
   into one nested YAML dict (``data`` / ``model`` / ``training`` / …).

3. **Backend** saves YAML → ``ExperimentConfig.from_yaml()`` parses it into typed
   dataclasses (``DataConfig``, ``ModelConfig``, …).

4. **This module** reads ``ExperimentConfig`` and produces ``ExperimentDataContext``
   (loaded ``dataset`` + ``SplitSpec`` indices). That is what callers mean by
   ``data_ctx`` — runtime objects, not the UI workflow dict.

``RunConfigView`` (``experiment_setup.py``) is a *print/validate* flattening of the
same YAML; ``ExperimentDataContext`` is the *data phase output* after disk load + split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from uqlab.data.benchmark_axes import (
    expects_aleatoric_eval,
    expects_epistemic_eval,
)
from uqlab.data.class_regions import (
    apply_region_noise,
    normalize_class_regions,
    sample_indices_for_four_region,
)
from uqlab.data.dataset_registry import get_dataset_spec, load_classification_dataset
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.data.experiment_loader import SplitSpec, sample_indices_for_experiment
from uqlab.shared.utils.classification import dino_transform

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PilotDataRequest:
    """Normalized data-phase inputs extracted from ``ExperimentConfig``."""

    dataset_name: str
    num_classes: int
    data_root: Path
    noise_type: str
    effective_noise_type: str
    aleatoric_noise_percentage: Optional[float]
    alea_for_load: Optional[float]
    under_supported_classes: list[int]
    under_train_per_class: int
    regular_train_per_class: int
    eval_per_group: int
    partition_mode: str = "legacy"
    class_regions: Optional[dict] = None
    per_class_config: Optional[dict] = None


@dataclass
class ExperimentDataContext:
    """Outputs of the data phase shared by facade coordinators and the runner."""

    dataset: object
    split_spec: SplitSpec
    dataset_name: str
    num_classes: int
    data_root: Path
    noise_type: str
    effective_noise_type: str
    aleatoric_noise_percentage: Optional[float]
    alea_for_load: Optional[float]
    under_supported_classes: list[int]

    @classmethod
    def from_request(
        cls,
        request: PilotDataRequest,
        *,
        dataset: object,
        split_spec: SplitSpec,
    ) -> ExperimentDataContext:
        return cls(
            dataset=dataset,
            split_spec=split_spec,
            dataset_name=request.dataset_name,
            num_classes=request.num_classes,
            data_root=request.data_root,
            noise_type=request.noise_type,
            effective_noise_type=request.effective_noise_type,
            aleatoric_noise_percentage=request.aleatoric_noise_percentage,
            alea_for_load=request.alea_for_load,
            under_supported_classes=list(request.under_supported_classes),
        )


def _resolve_data_root(config: ExperimentConfig, project_root: Path) -> Path:
    root = Path(getattr(config.paths, "data_root", None) or config.paths.cifar10n_root)
    return root if root.is_absolute() else project_root / root


def _noise_pct_for_dataset_load(
    *,
    dataset_name: str,
    noise_type: str,
    aleatoric_noise_percentage: Optional[float],
) -> Optional[float]:
    """
    How much synthetic noise to inject when calling ``load_classification_dataset``.

    - Explicit sweep percentage > 0 → inject that much on ``cifar10`` / ``mnist``.
    - ``cifar10n`` with a human-noise split → ``None`` (loader uses ``noise_type``).
    - Otherwise → ``0.0`` (clean labels).
    """
    if aleatoric_noise_percentage is not None and aleatoric_noise_percentage > 0:
        return float(aleatoric_noise_percentage)
    if dataset_name == "cifar10n" and noise_type not in (
        "clean_label",
        "none",
        "clean",
        "no_noise",
    ):
        return None
    return 0.0


def parse_pilot_data_request(
    config: ExperimentConfig,
    project_root: Path,
) -> PilotDataRequest:
    """Read and normalize ``config.data`` + ``config.paths`` for the data phase."""
    if config.data is None or config.paths is None:
        raise ValueError("ExperimentConfig.data and .paths are required")

    data = config.data
    dataset_name = getattr(data, "dataset_name", None) or "cifar10"
    ds_spec = get_dataset_spec(dataset_name)
    noise_type = data.noise_type
    aleatoric_noise_percentage = data.aleatoric_noise_percentage
    under_supported_classes = list(data.under_supported_classes or [])
    partition_mode = str(getattr(data, "partition_mode", "legacy") or "legacy")
    class_regions = getattr(data, "class_regions", None)
    per_class_config = getattr(data, "per_class_config", None)
    if per_class_config is not None:
        partition_mode = "per_class"
    if partition_mode == "four_region":
        class_regions = normalize_class_regions(class_regions)
        sparse = class_regions.get("sparse", {}).get("classes") or []
        if sparse:
            under_supported_classes = [int(c) for c in sparse]

    effective_noise_type = noise_type
    if partition_mode == "four_region":
        effective_noise_type = "clean_label"
        aleatoric_noise_percentage = 0.0
    elif aleatoric_noise_percentage == 0:
        effective_noise_type = "clean_label"

    alea_for_load = _noise_pct_for_dataset_load(
        dataset_name=dataset_name,
        noise_type=noise_type,
        aleatoric_noise_percentage=aleatoric_noise_percentage,
    )
    if partition_mode == "four_region":
        alea_for_load = 0.0

    return PilotDataRequest(
        dataset_name=dataset_name,
        num_classes=ds_spec.num_classes,
        data_root=_resolve_data_root(config, project_root),
        noise_type=noise_type,
        effective_noise_type=effective_noise_type,
        aleatoric_noise_percentage=aleatoric_noise_percentage,
        alea_for_load=alea_for_load,
        under_supported_classes=under_supported_classes,
        under_train_per_class=int(data.under_train_per_class),
        regular_train_per_class=int(data.regular_train_per_class)
        if data.regular_train_per_class is not None
        else int(data.under_train_per_class),
        eval_per_group=int(data.eval_per_group),
        partition_mode=partition_mode,
        class_regions=class_regions,
        per_class_config=per_class_config,
    )


def validate_pilot_data_request(request: PilotDataRequest) -> None:
    """Fail fast on invalid budgets before touching disk."""
    if request.regular_train_per_class < 0:
        raise ValueError(f"Invalid regular_train_per_class={request.regular_train_per_class}")
    if request.under_train_per_class < 0:
        raise ValueError(f"Invalid under_train_per_class={request.under_train_per_class}")
    if request.eval_per_group <= 0:
        raise ValueError(f"Invalid eval_per_group={request.eval_per_group}")

    if request.partition_mode == "per_class":
        raise NotImplementedError(
            "partition_mode=per_class is configured but not yet implemented in the "
            "data pipeline; use legacy or four_region"
        )

    if request.partition_mode == "four_region":
        if not request.class_regions:
            raise ValueError("class_regions required when partition_mode=four_region")
        from uqlab.data.class_regions import validate_class_regions

        validate_class_regions(request.class_regions, num_classes=request.num_classes)
        return

    if not request.under_supported_classes:
        raise ValueError("under_supported_classes must specify at least one class")

    for cls in request.under_supported_classes:
        if cls < 0 or cls >= request.num_classes:
            raise ValueError(
                f"Invalid class {cls} in under_supported_classes "
                f"(valid: 0..{request.num_classes - 1})"
            )


def load_pilot_dataset(request: PilotDataRequest, *, seed: int = 42) -> object:
    """Download/load the full training split via the dataset registry."""
    logger.info(
        "Loading %s via dataset factory (root=%s, alea_for_load=%s)",
        request.dataset_name,
        request.data_root,
        request.alea_for_load,
    )
    dataset = load_classification_dataset(
        request.dataset_name,
        root=request.data_root,
        noise_type=request.noise_type,
        aleatoric_noise_percentage=request.alea_for_load,
        train=True,
        download=True,
        transform=dino_transform(),
    )
    if getattr(dataset, "noise_mask", None) is not None:
        logger.info(
            "Loaded %s samples, noise rate %.2f%%",
            len(dataset),
            float(getattr(dataset, "noise_rate", 0)) * 100,
        )
    else:
        logger.info("Loaded %s samples", len(dataset))
    return dataset


def build_pilot_split_spec(request: PilotDataRequest, dataset: object, *, seed: int) -> SplitSpec:
    """Sample train/eval index pools (clean / aleatoric-like / epistemic-like / optional OOD)."""
    if request.partition_mode == "per_class":
        raise NotImplementedError(
            "partition_mode=per_class is configured but not yet implemented in the "
            "data pipeline; use legacy or four_region"
        )

    if request.partition_mode == "four_region":
        regions = normalize_class_regions(request.class_regions)
        apply_region_noise(dataset, regions, seed=seed)
        split_spec = sample_indices_for_four_region(
            dataset,
            regions,
            regular_train_per_class=request.regular_train_per_class,
            eval_per_group=request.eval_per_group,
            seed=seed,
        )
        logger.info(
            "Four-region splits: train=%s clean=%s aleatoric=%s epistemic=%s ood=%s",
            len(split_spec.train_indices),
            len(split_spec.clean_eval_indices),
            len(split_spec.aleatoric_eval_indices),
            len(split_spec.epistemic_eval_indices),
            len(split_spec.ood_eval_indices),
        )
        return split_spec

    split_spec = sample_indices_for_experiment(
        dataset,
        under_supported_classes=request.under_supported_classes,
        under_train_per_class=request.under_train_per_class,
        regular_train_per_class=request.regular_train_per_class,
        eval_per_group=request.eval_per_group,
        seed=seed,
        aleatoric_noise_percentage=request.aleatoric_noise_percentage or 0.0,
    )
    logger.info(
        "Splits: train=%s clean=%s aleatoric=%s epistemic=%s ood=%s",
        len(split_spec.train_indices),
        len(split_spec.clean_eval_indices),
        len(split_spec.aleatoric_eval_indices),
        len(split_spec.epistemic_eval_indices),
        len(split_spec.ood_eval_indices),
    )
    return split_spec


def validate_pilot_split_spec(request: PilotDataRequest, split_spec: SplitSpec) -> None:
    """Ensure expected benchmark pools exist for the configured axes."""
    all_empty = (
        len(split_spec.clean_eval_indices) == 0
        and len(split_spec.aleatoric_eval_indices) == 0
        and len(split_spec.epistemic_eval_indices) == 0
        and len(split_spec.ood_eval_indices) == 0
    )
    if all_empty:
        raise RuntimeError("All evaluation groups are empty — check training/eval budget.")

    if len(split_spec.aleatoric_eval_indices) == 0 and expects_aleatoric_eval(
        request.aleatoric_noise_percentage
    ):
        raise RuntimeError(
            f"Aleatoric benchmark requested ({request.aleatoric_noise_percentage}% noise) "
            "but aleatoric eval pool is empty."
        )

    if len(split_spec.epistemic_eval_indices) == 0 and expects_epistemic_eval(
        request.under_supported_classes,
        under_train_per_class=request.under_train_per_class,
        regular_train_per_class=request.regular_train_per_class,
    ):
        logger.warning("Epistemic eval pool is empty — epistemic AUROC will be NaN.")


def prepare_experiment_data(
    config: ExperimentConfig,
    project_root: Path,
    *,
    seed: int,
) -> ExperimentDataContext:
    """
    Data phase entry point: YAML ``ExperimentConfig`` → loaded dataset + splits.

    Called from ``run_experiment_core`` after ``extract_run_config`` prints the banner.
    Does **not** read Streamlit session state — only the saved YAML / dataclass.
    """
    request = parse_pilot_data_request(config, project_root)
    validate_pilot_data_request(request)
    dataset = load_pilot_dataset(request, seed=seed)
    split_spec = build_pilot_split_spec(request, dataset, seed=seed)
    validate_pilot_split_spec(request, split_spec)
    return ExperimentDataContext.from_request(request, dataset=dataset, split_spec=split_spec)
