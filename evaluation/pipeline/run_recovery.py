"""Finalize failed runs from on-disk ``zwischen/`` artifacts (no re-training)."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml

from uqlab.data.experiment_loader import SplitSpec
from uqlab.evaluation.result_writers import persist_experiment_summaries
from uqlab.evaluation.pipeline.experiment_eval import score_uncertainty_signals
from uqlab.run_artifacts import GROUP_ALEATORIC, GROUP_CLEAN, GROUP_EPISTEMIC, load_run_directory

logger = logging.getLogger(__name__)

RecoveryTier = Literal["db_sync", "zwischen_finalize", "partial", "none"]

_REQUIRED_ZWISCHEN = ("00_eval_setup", "05_signal_table")


@dataclass
class RecoverabilityReport:
    tier: RecoveryTier
    missing: list[str] = field(default_factory=list)
    zwischen_stages: list[str] = field(default_factory=list)
    error_hint: str | None = None
    has_summary: bool = False
    has_results_pt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _zwischen_dir(results_dir: Path) -> Path:
    return Path(results_dir) / "zwischen"


def _zwischen_stage_path(results_dir: Path, stage: str) -> Path:
    safe = stage.replace(" ", "_").replace("/", "_")
    return _zwischen_dir(results_dir) / f"{safe}.pt"


def list_zwischen_stages(results_dir: Path) -> list[str]:
    zwischen = _zwischen_dir(results_dir)
    manifest = zwischen / "manifest.json"
    if manifest.is_file():
        try:
            entries = json.loads(manifest.read_text())
            return [str(e.get("stage", "")) for e in entries if e.get("stage")]
        except json.JSONDecodeError:
            pass
    return sorted(p.stem for p in zwischen.glob("*.pt") if p.name != "manifest.json")


def load_zwischen_stage(results_dir: Path, stage: str) -> dict[str, Any]:
    path = _zwischen_stage_path(results_dir, stage)
    if not path.is_file():
        raise FileNotFoundError(f"Missing zwischen stage {stage!r} at {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Zwischen stage {stage!r} must be a dict, got {type(data)}")
    return data


def assess_run_recovery(results_dir: Path) -> RecoverabilityReport:
    """Classify what recovery action is possible from artifacts under *results_dir*."""
    results_dir = Path(results_dir)
    summary = results_dir / "summary.json"
    results_pt = results_dir / "results.pt"
    stages = list_zwischen_stages(results_dir)

    has_summary = summary.is_file()
    has_results_pt = results_pt.is_file()

    if has_summary or has_results_pt:
        return RecoverabilityReport(
            tier="db_sync",
            zwischen_stages=stages,
            has_summary=has_summary,
            has_results_pt=has_results_pt,
            error_hint="Disk artifacts present; sync DB from summary.json or results.pt",
        )

    missing = [s for s in _REQUIRED_ZWISCHEN if not _zwischen_stage_path(results_dir, s).is_file()]
    if not missing and stages:
        return RecoverabilityReport(
            tier="zwischen_finalize",
            zwischen_stages=stages,
            error_hint="Can finalize scoring from 00_eval_setup + 05_signal_table",
        )

    if stages:
        return RecoverabilityReport(
            tier="partial",
            missing=missing,
            zwischen_stages=stages,
            error_hint="Incomplete zwischen chain; cannot finalize without required stages",
        )

    return RecoverabilityReport(
        tier="none",
        missing=list(_REQUIRED_ZWISCHEN),
        zwischen_stages=stages,
        error_hint="No recoverable artifacts on disk",
    )


def _load_run_config(experiment_dir: Path) -> dict[str, Any]:
    cfg_path = experiment_dir / "config.yaml"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _infer_train_size(results_dir: Path, cfg: dict[str, Any]) -> int:
    csv_path = results_dir / "training_data.csv"
    if csv_path.is_file():
        try:
            with csv_path.open(encoding="utf-8") as f:
                return max(sum(1 for _ in f) - 1, 0)
        except OSError:
            pass
    data = cfg.get("data") or {}
    under = int(data.get("under_train_per_class") or 50)
    regular = int(data.get("regular_train_per_class") or 300)
    under_classes = str(data.get("under_supported_classes") or "3,7")
    n_under = len([x for x in under_classes.replace("random:", "").split(",") if x.strip()])
    n_regular = max(10 - n_under, 1)
    return n_under * under + n_regular * regular


def _eval_sizes_from_labels(group_labels: torch.Tensor) -> dict[str, int]:
    labels = group_labels.detach().cpu() if hasattr(group_labels, "detach") else group_labels
    return {
        "clean": int((labels == GROUP_CLEAN).sum().item()),
        "aleatoric_like": int((labels == GROUP_ALEATORIC).sum().item()),
        "epistemic_like": int((labels == GROUP_EPISTEMIC).sum().item()),
    }


def _minimal_split_spec(cfg: dict[str, Any]) -> SplitSpec:
    import numpy as np

    data = cfg.get("data") or {}
    raw = data.get("under_supported_classes") or []
    if isinstance(raw, str):
        under = [int(x.strip()) for x in raw.replace("random:", "").split(",") if x.strip()]
    else:
        under = [int(x) for x in raw]
    empty = np.array([], dtype=np.int64)
    return SplitSpec(
        train_indices=empty,
        clean_eval_indices=empty,
        aleatoric_eval_indices=empty,
        epistemic_eval_indices=empty,
        under_supported_classes=under,
    )


def _args_namespace(cfg: dict[str, Any], *, seed: int, device: str) -> argparse.Namespace:
    data = cfg.get("data") or {}
    model = cfg.get("model") or {}
    training = cfg.get("training") or {}
    evaluation = cfg.get("evaluation") or {}
    under_raw = data.get("under_supported_classes") or ""
    if isinstance(under_raw, list):
        under_str = ",".join(str(x) for x in under_raw)
    else:
        under_str = str(under_raw)
    return argparse.Namespace(
        noise_type=str(data.get("noise_type") or "clean_label"),
        under_supported_classes=under_str,
        under_train_per_class=int(data.get("under_train_per_class") or 50),
        regular_train_per_class=int(data.get("regular_train_per_class") or 300),
        eval_per_group=int(data.get("eval_per_group") or 100),
        dinov2_model=str(model.get("dinov2_model") or "small"),
        hidden_dim=int(model.get("hidden_dim") or 256),
        dropout=float(model.get("dropout") or 0.0),
        epochs=int(training.get("epochs") or 10),
        learning_rate=float(training.get("learning_rate") or 0.001),
        weight_decay=float(training.get("weight_decay") or 0.0001),
        train_batch_size=int(training.get("train_batch_size") or 256),
        feature_batch_size=int(training.get("feature_batch_size") or 64),
        mc_passes=int(evaluation.get("mc_passes") or 0),
        top_k=int(evaluation.get("top_k") or 10),
        seed=seed,
        device=device,
    )


def _signal_table_from_zwischen(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    table: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            table[str(key)] = value.float()
        else:
            table[str(key)] = torch.as_tensor(value).float()
    return table


def _mean_prediction_from_zwischen(results_dir: Path) -> torch.Tensor | None:
    for stage in ("01_deterministic_forward", "04_mc_dropout"):
        path = _zwischen_stage_path(results_dir, stage)
        if not path.is_file():
            continue
        try:
            data = load_zwischen_stage(results_dir, stage)
        except (FileNotFoundError, TypeError):
            continue
        pred = data.get("mean_prediction")
        if isinstance(pred, torch.Tensor):
            return pred
    return None


def sync_run_from_disk(results_dir: Path) -> dict[str, Any]:
    """Load existing summary.json or results.pt without rewriting disk."""
    artifacts = load_run_directory(results_dir)
    if artifacts.source == "none":
        raise FileNotFoundError(f"No summary.json or results.pt under {results_dir}")
    summary_path = results_dir / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    return {
        "train_size": artifacts.train_size or 0,
        "eval_sizes": artifacts.eval_sizes or {},
        "one_vs_rest_auroc": artifacts.one_vs_rest_auroc or [],
    }


def finalize_run_from_zwischen(
    results_dir: Path,
    *,
    experiment_dir: Path | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Re-run post-train scoring from ``zwischen/`` and write ``summary.json`` + ``results.pt``.

    Does not re-train the model.
    """
    results_dir = Path(results_dir)
    report = assess_run_recovery(results_dir)
    if report.tier != "zwischen_finalize":
        raise ValueError(
            f"Cannot finalize from zwischen (tier={report.tier!r}, missing={report.missing})"
        )

    exp_dir = Path(experiment_dir) if experiment_dir else results_dir.parent
    cfg = _load_run_config(exp_dir)

    setup = load_zwischen_stage(results_dir, "00_eval_setup")
    signal_payload = load_zwischen_stage(results_dir, "05_signal_table")
    signal_table = _signal_table_from_zwischen(signal_payload)

    eval_group_labels = setup["eval_group_labels"].long()
    eval_clean_labels = setup["eval_clean_labels"].long()
    eval_is_noisy = setup["eval_is_noisy"].bool()
    eval_noisy_labels = setup["eval_noisy_labels"].long()
    eval_dataset_index = setup["eval_dataset_index"].long()

    eval_summary = score_uncertainty_signals(
        signal_table=signal_table,
        eval_group_labels=eval_group_labels,
        eval_clean_labels=eval_clean_labels,
        eval_is_noisy=eval_is_noisy,
        eval_noisy_labels=eval_noisy_labels,
        eval_dataset_index=eval_dataset_index,
        results_dir=results_dir,
        device=torch.device(device),
        seed=seed,
    )

    auroc_rows = eval_summary["auroc_rows"]
    one_vs_rest_auroc = eval_summary["one_vs_rest_auroc"]
    clf_rows = eval_summary["clf_rows"]
    eval_sizes = _eval_sizes_from_labels(eval_group_labels)
    train_size = _infer_train_size(results_dir, cfg)
    split_spec = _minimal_split_spec(cfg)
    config_ns = _args_namespace(cfg, seed=seed, device=device)

    eval_protocol = {
        "architecture_invariant": True,
        "rationale": "Recovered from zwischen artifacts (no re-training).",
        "eval_per_group": int((cfg.get("data") or {}).get("eval_per_group") or 100),
        "groups": ["clean", "aleatoric_like", "epistemic_like"],
        "under_supported_classes": list(split_spec.under_supported_classes),
        "seed": seed,
        "recovered_from_zwischen": True,
    }

    summary: dict[str, Any] = {
        "config": cfg,
        "under_supported_classes": split_spec.under_supported_classes,
        "train_size": train_size,
        "eval_sizes": eval_sizes,
        "eval_protocol": eval_protocol,
        "one_vs_rest_auroc": one_vs_rest_auroc,
        "auroc_rows": [
            {"signal": name, "aleatoric_auroc": alea, "epistemic_auroc": epis}
            for name, alea, epis in auroc_rows
        ],
        "macro_f1": [{"signal_set": name, "macro_f1": score} for name, score in clf_rows],
        "recovered_from_zwischen": True,
    }

    persist_experiment_summaries(
        results_dir,
        summary=summary,
        args=config_ns,
        split_spec=split_spec,
        train_size=train_size,
        eval_sizes=eval_sizes,
        auroc_rows=auroc_rows,
        clf_rows=clf_rows,
    )

    mean_pred = _mean_prediction_from_zwischen(results_dir)
    if mean_pred is None:
        raise ValueError("Cannot build results.pt: missing mean_prediction in zwischen stages")

    results_data: dict[str, Any] = {
        "predictions": mean_pred.argmax(dim=1),
        "confidences": mean_pred.max(dim=1).values,
        "eval_clean_labels": eval_clean_labels,
        "eval_noisy_labels": eval_noisy_labels,
        "eval_is_noisy": eval_is_noisy,
        "eval_group_labels": eval_group_labels,
        "eval_indices": eval_dataset_index,
        "signal_table": signal_table,
        "auroc_rows": auroc_rows,
        "recovered_from_zwischen": True,
    }
    torch.save(results_data, results_dir / "results.pt")
    logger.info("Recovered run artifacts under %s", results_dir)
    return summary


def recover_run_on_disk(
    results_dir: Path,
    *,
    experiment_dir: Path | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Dispatch to db_sync or zwischen_finalize based on disk assessment."""
    report = assess_run_recovery(results_dir)
    if report.tier == "db_sync":
        return sync_run_from_disk(results_dir)
    if report.tier == "zwischen_finalize":
        return finalize_run_from_zwischen(
            results_dir,
            experiment_dir=experiment_dir,
            seed=seed,
            device=device,
        )
    raise ValueError(
        f"Run not recoverable (tier={report.tier!r}, missing={report.missing}, "
        f"stages={report.zwischen_stages})"
    )


__all__ = [
    "RecoverabilityReport",
    "RecoveryTier",
    "assess_run_recovery",
    "finalize_run_from_zwischen",
    "list_zwischen_stages",
    "load_zwischen_stage",
    "recover_run_on_disk",
    "sync_run_from_disk",
]
