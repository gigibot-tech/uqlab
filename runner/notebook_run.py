"""
Optional one-liner after YAML config → train + eval + artifacts.

Notebooks should call ``run_train_and_eval_phases`` directly (see
``notebooks/cifar10_paper_flow.ipynb``); this wrapper only saves boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uqlab.data.packs import prepare_run_data_context
from uqlab.data.setup import prepare_experiment_data
from uqlab.runner.phases.config_view import apply_data_context, extract_run_config
from uqlab.runner.train_eval import run_train_and_eval_phases
from uqlab.shared.config.classification import ExperimentConfig
from uqlab.shared.utils.classification import auto_device, set_seed


def run_notebook_experiment(
    config: ExperimentConfig,
    results_dir: Path,
    *,
    project_root: Path,
    seed: int = 42,
    device_str: str = "auto",
    config_path: Path | None = None,
    persist: bool = True,
    log: bool = False,
) -> dict[str, Any]:
    """
    Notebook-sized wrapper around ``run_train_and_eval_phases``.

    Skips Streamlit banners; still writes the same artifacts when ``persist=True``.
    """
    set_seed(seed)
    device = auto_device(device_str)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_cache_dir = results_dir / "cache"

    run_cfg = extract_run_config(config)
    data_ctx = prepare_experiment_data(config, project_root, seed=seed)
    apply_data_context(run_cfg, data_ctx)

    data_pack = prepare_run_data_context(
        config=config,
        dataset=data_ctx.dataset,
        split_spec=data_ctx.split_spec,
        dataset_name=run_cfg.dataset_name,
        device=device,
        feature_cache_dir=project_root / config.paths.feature_cache_dir,
        noise_type=run_cfg.noise_type,
        feature_batch_size=run_cfg.feature_batch_size,
    )

    return run_train_and_eval_phases(
        config=config,
        run_cfg=run_cfg,
        results_dir=results_dir,
        run_cache_dir=run_cache_dir,
        data_pack=data_pack,
        split_spec=data_ctx.split_spec,
        device=device,
        seed=seed,
        training_config=config.training,
        data_config=config.data,
        model_config=config.model,
        eval_config=config.evaluation,
        ds_spec=run_cfg.dataset_spec,
        config_path=config_path,
        persist=persist,
        log=log,
    )


__all__ = ["run_notebook_experiment"]
