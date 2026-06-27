"""Fast-pilot experiment engine: train, evaluate signals, write artifacts.

Called only from :func:`uqlab.runner.execute.run_from_yaml` (or ``run_from_python_config``).

Paper API map: ``docs/features/PAPER_FLOW.md``

One run = one paper sweep point. Multi-run DE + PNG = campaign end (not here).
"""

from __future__ import annotations

import logging
from pathlib import Path

from uqlab.data.packs import (
    get_data_loading_mode,
    prepare_eval_data,
    prepare_eval_tensors,
    prepare_run_data_context,
)
from uqlab.data.setup import prepare_experiment_data
from uqlab.runner.console_log import log_run_data_context
from uqlab.runner.phases.config_view import (
    apply_data_context,
    extract_run_config,
    print_dataset_loaded,
    print_experiment_configuration,
    validate_eval_splits,
)
from uqlab.runner.train_eval import run_train_and_eval_phases
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.shared.utils.classification import auto_device, set_seed

logger = logging.getLogger(__name__)

from uqlab.run_artifacts import GROUP_ALEATORIC, GROUP_CLEAN, GROUP_EPISTEMIC, GROUP_OOD

GROUP_NAMES: dict[int, str] = {
    GROUP_CLEAN: "clean",
    GROUP_ALEATORIC: "aleatoric_like",
    GROUP_EPISTEMIC: "epistemic_like",
    GROUP_OOD: "ood_like",
}


def run_experiment_core(
    config: ExperimentConfig,
    results_dir: Path,
    *,
    seed: int,
    device_str: str,
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> dict:
    """
    One paper sweep point: fit + materialize ``predict_disentangling`` vectors on disk.

    Phases:

    1. **Config + data** — ``prepare_experiment_data`` → ``prepare_run_data_context``
    2. **Train + eval** — ``run_train_and_eval_phases`` (see module docstring for Keras mapping)
    3. **Not here** — ``calculate_disentanglement_error`` / campaign PNG (needs N runs)
    """
    from uqlab.runtime_paths import repository_root

    root = project_root if project_root is not None else repository_root()

    # LOG: experiment banner (stdout)
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

    # DATA: YAML → dataset + SplitSpec (legacy or four_region — see data/setup.py)
    data_ctx = prepare_experiment_data(config, root, seed=seed)
    apply_data_context(run_cfg, data_ctx)
    dataset = data_ctx.dataset
    split_spec = data_ctx.split_spec
    dataset_name = run_cfg.dataset_name
    ds_spec = run_cfg.dataset_spec

    print_dataset_loaded(data_ctx, dataset)
    validate_eval_splits(run_cfg, split_spec)

    # DATA: SplitSpec → train_dataset + eval packs (embeddings | images)
    data_pack = prepare_run_data_context(
        config=config,
        dataset=dataset,
        split_spec=split_spec,
        dataset_name=dataset_name,
        device=device,
        feature_cache_dir=feature_cache_dir,
        noise_type=run_cfg.noise_type,
        feature_batch_size=run_cfg.feature_batch_size,
    )

    # LOG: device, group sizes (stdout)
    log_run_data_context(
        device=device,
        results_dir=results_dir,
        train_dataset=data_pack["train_dataset"],
        clean_eval_pack=data_pack["clean_eval_pack"],
        aleatoric_eval_pack=data_pack["aleatoric_eval_pack"],
        epistemic_eval_pack=data_pack["epistemic_eval_pack"],
        ood_eval_pack=data_pack["ood_eval_pack"],
    )

    result = run_train_and_eval_phases(
        config=config,
        run_cfg=run_cfg,
        results_dir=results_dir,
        run_cache_dir=run_cache_dir,
        data_pack=data_pack,
        split_spec=split_spec,
        device=device,
        seed=seed,
        training_config=training_config,
        data_config=data_config,
        model_config=model_config,
        eval_config=eval_config,
        ds_spec=ds_spec,
        config_path=config_path,
        persist=True,
        log=True,
    )
    return result["summary"]


__all__ = [
    "GROUP_ALEATORIC",
    "GROUP_CLEAN",
    "GROUP_EPISTEMIC",
    "GROUP_NAMES",
    "get_data_loading_mode",
    "prepare_eval_data",
    "prepare_eval_tensors",
    "prepare_run_data_context",
    "run_experiment_core",
    "run_train_and_eval_phases",
]

from uqlab.models.training import train_feature_model, train_image_model  # noqa: E402

__all__ += ["train_feature_model", "train_image_model"]
