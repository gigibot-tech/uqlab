"""
Train + uncertainty eval phases (paper ``fit`` + ``predict_disentangling`` in one job).

Notebooks: call after ``prepare_experiment_data`` + ``prepare_run_data_context``.
Full runs: ``run_experiment_core`` wraps this plus config banner and persist/log.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uqlab.evaluation.reporting.run_summary import build_run_summary
from uqlab.models.training import (
    build_model_for_run,
    train_feature_model,
    train_image_model,
)
from uqlab.runner.console_log import log_run_complete, log_zwischen_dir
from uqlab.runner.phases.eval import (
    EvalSignalConfig,
    collect_uncertainty_signals,
    score_uncertainty_signals,
)
from uqlab.run_artifacts import persist_run_outputs, save_zwischen_result
from uqlab.shared.config.classification import ExperimentConfig

logger = logging.getLogger(__name__)


def run_train_and_eval_phases(
    *,
    config: ExperimentConfig,
    run_cfg,
    results_dir: Path,
    run_cache_dir: Path,
    data_pack: dict[str, Any],
    split_spec,
    device,
    seed: int,
    training_config,
    data_config,
    model_config,
    eval_config,
    ds_spec,
    config_path: Path | None = None,
    persist: bool = True,
    log: bool = True,
) -> dict[str, Any]:
    """
    Core ML block: train → MC/attribution signals → optional persist.

    Paper mapping (Keras ``InformationTheoreticModel``):

    - ``fit`` → ``build_model_for_run`` + ``train_*_model``
    - ``predict_disentangling`` → ``collect_uncertainty_signals`` (MC → ``expected_entropy``,
      ``mutual_info`` in ``signal_table``) + ``score_uncertainty_signals`` (writes
      ``per_sample_signals.csv``)

    Returns ``eval_summary`` plus ``summary`` when ``persist=True``.
    """
    train_dataset = data_pack["train_dataset"]
    clean_eval_pack = data_pack["clean_eval_pack"]
    aleatoric_eval_pack = data_pack["aleatoric_eval_pack"]
    epistemic_eval_pack = data_pack["epistemic_eval_pack"]
    ood_eval_pack = data_pack["ood_eval_pack"]
    eval_data = data_pack["eval_data"]
    eval_inputs = data_pack["eval_inputs"]
    mode = data_pack["mode"]
    feature_dim = data_pack["feature_dim"]

    epochs = run_cfg.epochs
    mc_passes = run_cfg.mc_passes
    top_k = run_cfg.top_k
    dinov2_model = run_cfg.dinov2_model
    hidden_dim = run_cfg.hidden_dim
    dropout = run_cfg.dropout
    feature_batch_size = run_cfg.feature_batch_size

    # --- paper fit ---
    model, prior_epoch_loaded = build_model_for_run(
        config=config,
        num_classes=ds_spec.num_classes,
        feature_dim=feature_dim,
        mode=mode,
        device=device,
        feature_batch_size=feature_batch_size,
        epochs=epochs,
    )
    if mode == "embeddings":
        model = train_feature_model(model, train_dataset, training_config, device)
    elif mode == "images":
        model = train_image_model(model, train_dataset, training_config, device)
    else:
        raise ValueError(f"Unsupported training mode: {mode}")
    model = model.to(device)
    model.eval()

    eval_group_labels = eval_data["eval_group_labels"]
    eval_clean_labels = eval_data["eval_clean_labels"]
    eval_is_noisy = eval_data["eval_is_noisy"]
    eval_noisy_labels = eval_data["eval_noisy_labels"]
    eval_dataset_index = eval_data["eval_dataset_index"]

    # SAVE: zwischen/00_eval_setup.pt (debug / recovery)
    eval_setup_path = save_zwischen_result(
        results_dir,
        "00_eval_setup",
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

    # --- paper predict_disentangling (computed in-job) ---
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
    uq = eval_outputs.get("uq") or {}
    mean_pred_det = eval_outputs.get("mean_pred_det")
    signal_table = eval_outputs["signal_table"]
    if signal_table is None or not isinstance(signal_table, dict):
        raise TypeError("collect_uncertainty_signals() returned invalid `signal_table` payload")
    if log:
        log_zwischen_dir(results_dir)

    # SAVE: per_sample_signals.csv (inside score_uncertainty_signals)
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

    out: dict[str, Any] = {
        "eval_summary": eval_summary,
        "signal_table": signal_table,
        "model": model,
        "eval_setup_path": eval_setup_path,
    }

    if not persist:
        return out

    config_dict = asdict(config)
    if config_dict.get("paths"):
        config_dict["paths"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in config_dict["paths"].items()
        }
    if config_dict.get("model"):
        config_dict["model"] = dict(config.model)

    auroc_rows = eval_summary["auroc_rows"]
    one_vs_rest_auroc = eval_summary["one_vs_rest_auroc"]
    clf_rows = eval_summary["clf_rows"]

    summary, signal_formulas = build_run_summary(
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
        eval_per_group=run_cfg.eval_per_group,
        top_k=top_k,
        mc_passes=mc_passes,
        one_vs_rest_auroc=one_vs_rest_auroc,
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
    )

    config_ns = argparse.Namespace(
        noise_type=run_cfg.noise_type,
        under_supported_classes=run_cfg.under_supported_classes_str,
        under_train_per_class=run_cfg.under_train_per_class,
        regular_train_per_class=run_cfg.regular_train_per_class,
        eval_per_group=run_cfg.eval_per_group,
        dinov2_model=dinov2_model,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        learning_rate=run_cfg.learning_rate,
        weight_decay=run_cfg.weight_decay,
        train_batch_size=run_cfg.train_batch_size,
        feature_batch_size=feature_batch_size,
        mc_passes=mc_passes,
        top_k=top_k,
        seed=seed,
        device=str(device),
    )

    written = persist_run_outputs(
        results_dir,
        train_dataset=train_dataset,
        config_dict=config_dict,
        summary=summary,
        signal_formulas=signal_formulas,
        config_ns=config_ns,
        split_spec=split_spec,
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
        per_sample_csv_path=eval_summary.get("per_sample_csv_path"),
        eval_setup_zwischen_path=eval_setup_path,
        model=model,
        prior_epoch_loaded=prior_epoch_loaded,
        epochs=epochs,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=ds_spec.num_classes,
        dinov2_model=dinov2_model,
        uq=uq,
        mean_pred_det=mean_pred_det,
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
    )

    if log:
        log_run_complete(written, results_dir=results_dir, eval_summary=eval_summary, summary=summary)

    out["summary"] = summary
    out["written"] = written
    return out


__all__ = ["run_train_and_eval_phases"]
