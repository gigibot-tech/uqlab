"""In-memory run summary assembly (no disk I/O)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from uqlab.data.experiment_loader import SplitSpec
from uqlab.evaluation.signals.formulas import build_signal_formula_manifest
from uqlab.runner.phases.eval import GROUP_NAMES


def build_run_summary(
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
    """Build summary dict and signal formula manifest (memory only)."""
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


__all__ = ["build_run_summary"]
