"""
Fast-pilot evaluation in two steps:

1. **collect_uncertainty_signals** — paper MC/attribution in-job (``predict_disentangling`` data).
2. **score_uncertainty_signals** — AUROC + writes ``per_sample_signals.csv``.

Paper mapping: see ``docs/features/PAPER_FLOW.md``.

Zwischen stages (under ``<run_dir>/zwischen/``):

- ``00_eval_setup`` — written by orchestrator before this module runs
- ``01_deterministic_forward`` — logits for DualXDA targets
- ``02_*`` — attribution primitives (DualXDA / EK-FAC / GradDot)
- ``03_logit_signals`` — representer logit magnitudes
- ``04_mc_dropout`` — entropy, mutual_info (paper aleatoric/epistemic proxies)
- ``05_signal_table`` — final per-sample signal columns
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from uqlab.evaluation.reporting.result_writers import save_per_sample_csv
from uqlab.evaluation.metrics.scoring import (
    auroc_skip_reason,
    binary_auroc_or_none,
    binary_auroc_vs_group,
    evaluate_three_way_classification,
)
from uqlab.evaluation.signals.registry import (
    METRICS,
    build_signal_table_from_store,
    prune_enabled_metrics,
)
from uqlab.evaluation.signals.sources import (
    EvalContext,
    logit_magnitude_from_store,
    run_sources,
    sources_for_metrics,
)
from uqlab.runner.phases.eval_signal_config import EvalSignalConfig
from uqlab.run_artifacts import GROUP_ALEATORIC, GROUP_CLEAN, GROUP_EPISTEMIC, GROUP_OOD, save_zwischen_result
from uqlab.shared.config.signals import derive_attribution_backends_from_signals

GROUP_NAMES: dict[int, str] = {
    0: "clean",
    1: "aleatoric_like",
    2: "epistemic_like",
    3: "ood_like",
}


def _persist_influence_matrices(
    results_dir: Path,
    store: dict[str, Any],
    needed: set[str],
) -> None:
    """Save full [n_eval, n_train] attribution score matrices under ``zwischen/``."""
    from uqlab.evaluation.signals.primitives import (
        INFLUENCE_DUALXDA,
        INFLUENCE_EK_FAK,
        INFLUENCE_GRADDOT,
    )

    specs = (
        (frozenset({"attribution", "attribution_dualxda"}), INFLUENCE_DUALXDA, "02_influence_dualxda"),
        (frozenset({"attribution_graddot"}), INFLUENCE_GRADDOT, "02_influence_graddot"),
        (frozenset({"attribution_ek_fak"}), INFLUENCE_EK_FAK, "02_influence_ek_fak"),
    )
    for source_ids, key, slug in specs:
        if not needed.intersection(source_ids):
            continue
        matrix = store.get(key)
        if matrix is not None:
            save_zwischen_result(
                results_dir,
                slug,
                {"scores": matrix.cpu(), "shape": list(matrix.shape)},
            )


def collect_uncertainty_signals(
    *,
    model: nn.Module,
    train_dataset,
    eval_inputs: torch.Tensor,
    device: torch.device,
    config: EvalSignalConfig,
) -> dict[str, Any]:
    """
    Score every eval sample with uncertainty signals (attribution + logits + optional MC dropout).

    Returns a ``signal_table``: dict[name → tensor[N]] — one number per sample per signal.
    """
    eval_x = eval_inputs.to(device)
    results_dir = config.results_dir

    enabled_signals = config.resolved_enabled_signals()
    if enabled_signals is None:
        enabled_signals = set(METRICS.keys())
    enabled_signals = prune_enabled_metrics(
        enabled_signals,
        mc_passes=config.mc_passes,
        dropout=config.dropout,
    )

    needed = sources_for_metrics(enabled_signals)

    derived_backends = derive_attribution_backends_from_signals(enabled_signals)

    ctx = EvalContext(
        model=model,
        train_dataset=train_dataset,
        eval_inputs=eval_inputs,
        eval_x=eval_x,
        device=device,
        train_batch_size=config.train_batch_size,
        top_k=config.top_k,
        mc_passes=config.mc_passes,
        dropout=config.dropout,
        attribution_method=config.attribution_method,
        attribution_backends=config.attribution_backends,
        run_cache_dir=config.run_cache_dir,
    )

    if "deterministic_forward" in needed:
        print("Deterministic eval forward (DualXDA targets)...")
    if needed & {"attribution_dualxda", "attribution_ek_fak", "attribution_graddot", "attribution"}:
        backends_label = ", ".join(derived_backends) or "none"
        print(f"Attribution primitives ({backends_label})...")
    if "mc_dropout" in needed:
        if config.mc_passes > 0:
            print(f"MC Dropout ({config.mc_passes} passes, batched eval)...")
        else:
            print("⚠️  MC Dropout disabled (mc_passes=0)")

    store = run_sources(needed, ctx)
    _persist_influence_matrices(results_dir, store, needed)

    if "deterministic_forward" in needed:
        save_zwischen_result(
            results_dir,
            "01_deterministic_forward",
            {
                "det_logits": store["forward.det_logits"],
                "mean_prediction": store["forward.mean_pred"],
            },
        )
    if "attribution" in needed or "attribution_dualxda" in needed:
        save_zwischen_result(
            results_dir,
            "02_attribution_signals",
            {
                "coherence": store["attribution.coherence"].cpu(),
                "mass": store["attribution.mass"].cpu(),
                "dominance": store["attribution.dominance"].cpu(),
            },
        )
    if "attribution_ek_fak" in needed:
        from uqlab.evaluation.signals.primitives import (
            EK_FAK_COHERENCE,
            EK_FAK_DOMINANCE,
            EK_FAK_MASS,
        )

        save_zwischen_result(
            results_dir,
            "02b_ek_fak_attribution_signals",
            {
                "coherence": store[EK_FAK_COHERENCE].cpu(),
                "mass": store[EK_FAK_MASS].cpu(),
                "dominance": store[EK_FAK_DOMINANCE].cpu(),
            },
        )
    if "attribution_graddot" in needed:
        from uqlab.evaluation.signals.primitives import (
            GRADDOT_COHERENCE,
            GRADDOT_DOMINANCE,
            GRADDOT_MASS,
        )

        save_zwischen_result(
            results_dir,
            "02c_graddot_attribution_signals",
            {
                "coherence": store[GRADDOT_COHERENCE].cpu(),
                "mass": store[GRADDOT_MASS].cpu(),
                "dominance": store[GRADDOT_DOMINANCE].cpu(),
            },
        )
    if needed & {
        "attribution",
        "attribution_dualxda",
        "attribution_ek_fak",
        "attribution_graddot",
        "deterministic_forward",
    }:
        logit_magnitude = logit_magnitude_from_store(store)
        save_zwischen_result(
            results_dir,
            "03_logit_signals",
            {
                "det_logits": store.get("forward.det_logits"),
                "logit_magnitude_pred_class": logit_magnitude,
            },
        )
    else:
        logit_magnitude = None

    if "mc_dropout" in needed:
        save_zwischen_result(
            results_dir,
            "04_mc_dropout",
            {
                "entropy": store["mc.entropy"].cpu(),
                "mutual_info": store["mc.mutual_info"].cpu(),
                "mean_prediction": store["mc.mean_prediction"].cpu(),
                "n_passes": config.mc_passes,
            },
        )

    signal_table = build_signal_table_from_store(
        store,
        enabled=enabled_signals,
        mc_passes=config.mc_passes,
        dropout=config.dropout,
    )
    save_zwischen_result(
        results_dir,
        "05_signal_table",
        {k: v.cpu() if hasattr(v, "cpu") else v for k, v in signal_table.items()},
    )

    det_logits = store.get("forward.det_logits")
    mean_pred_det = store.get("forward.mean_pred")
    attribution_signals = None
    if "attribution" in needed:
        attribution_signals = {
            "coherence": store["attribution.coherence"],
            "mass": store["attribution.mass"],
            "dominance": store["attribution.dominance"],
        }
    uq = None
    if "mc_dropout" in needed:
        uq = {
            "entropy": store["mc.entropy"],
            "mutual_info": store["mc.mutual_info"],
            "mean_prediction": store["mc.mean_prediction"],
        }
    elif "deterministic_forward" in needed:
        mean = store["forward.mean_pred"]
        n = int(mean.shape[0])
        uq = {
            "entropy": torch.zeros(n),
            "mutual_info": torch.zeros(n),
            "mean_prediction": mean,
        }

    return {
        "eval_x": eval_x,
        "det_logits": det_logits,
        "mean_pred_det": mean_pred_det,
        "attribution_signals": attribution_signals,
        "logit_magnitude": logit_magnitude,
        "mc_predictions": None,
        "uq": uq,
        "signal_table": signal_table,
        "primitive_store": store,
    }


def _auroc_per_signal(
    signal_table: dict[str, torch.Tensor],
    *,
    eval_group_labels: torch.Tensor,
    aleatoric_positive: torch.Tensor,
    epistemic_positive: torch.Tensor,
    ood_positive: torch.Tensor,
) -> tuple[list[tuple], list[dict], str | None, str | None, str | None]:
    """
    For each signal column: AUROC vs noisy-label group, under-trained-class group,
    OOD one-vs-rest, and region-vs-clean separation when four pools are present.
    """
    n_alea_pos = int(aleatoric_positive.sum().item())
    n_epi_pos = int(epistemic_positive.sum().item())
    n_ood_pos = int(ood_positive.sum().item())
    n_alea_neg = int((~aleatoric_positive).sum().item())
    n_epi_neg = int((~epistemic_positive).sum().item())
    n_ood_neg = int((~ood_positive).sum().item())
    alea_skip = auroc_skip_reason(n_alea_pos, n_alea_neg, axis="aleatoric")
    epis_skip = auroc_skip_reason(n_epi_pos, n_epi_neg, axis="epistemic")
    ood_skip = auroc_skip_reason(n_ood_pos, n_ood_neg, axis="ood")

    has_four_regions = n_ood_pos > 0
    n_clean = int((eval_group_labels == GROUP_CLEAN).sum().item())

    auroc_rows: list[tuple] = []
    one_vs_rest: list[dict] = []
    for name, values in signal_table.items():
        alea_auc = binary_auroc_or_none(values, aleatoric_positive)
        epis_auc = binary_auroc_or_none(values, epistemic_positive)
        ood_auc = binary_auroc_or_none(values, ood_positive) if n_ood_pos > 0 else None
        noisy_vs_clean = (
            binary_auroc_vs_group(
                values,
                eval_group_labels,
                positive_group=GROUP_ALEATORIC,
                negative_group=GROUP_CLEAN,
            )
            if has_four_regions and n_alea_pos > 0 and n_clean > 0
            else None
        )
        sparse_vs_clean = (
            binary_auroc_vs_group(
                values,
                eval_group_labels,
                positive_group=GROUP_EPISTEMIC,
                negative_group=GROUP_CLEAN,
            )
            if has_four_regions and n_epi_pos > 0 and n_clean > 0
            else None
        )
        ood_vs_clean = (
            binary_auroc_vs_group(
                values,
                eval_group_labels,
                positive_group=GROUP_OOD,
                negative_group=GROUP_CLEAN,
            )
            if has_four_regions and n_ood_pos > 0 and n_clean > 0
            else None
        )
        auroc_rows.append((name, alea_auc, epis_auc, ood_auc))
        one_vs_rest.append(
            {
                "signal": name,
                "aleatoric_like_auroc": alea_auc,
                "epistemic_like_auroc": epis_auc,
                "ood_like_auroc": ood_auc,
                "noisy_vs_clean_auroc": noisy_vs_clean,
                "sparse_vs_clean_auroc": sparse_vs_clean,
                "ood_vs_clean_auroc": ood_vs_clean,
                "aleatoric_skip_reason": alea_skip,
                "epistemic_skip_reason": epis_skip,
                "ood_skip_reason": ood_skip,
            }
        )
    return auroc_rows, one_vs_rest, alea_skip, epis_skip, ood_skip


def score_uncertainty_signals(
    *,
    signal_table: dict[str, torch.Tensor],
    eval_group_labels: torch.Tensor,
    eval_clean_labels: torch.Tensor,
    eval_is_noisy: torch.Tensor,
    eval_noisy_labels: torch.Tensor,
    eval_dataset_index: torch.Tensor,
    results_dir: Path,
    device: torch.device,
    seed: int,
) -> dict[str, list]:
    """
    Grade each signal: AUROC for noisy-label detection and under-trained-class detection.

    Also trains a simple N-way classifier on signals (macro-F1) and writes
    ``per_sample_signals.csv`` (paper ``predict_disentangling`` per-sample columns).
    """
    aleatoric_positive = eval_group_labels == GROUP_ALEATORIC
    epistemic_positive = eval_group_labels == GROUP_EPISTEMIC
    ood_positive = eval_group_labels == GROUP_OOD

    auroc_rows, one_vs_rest, _, _, _ = _auroc_per_signal(
        signal_table,
        eval_group_labels=eval_group_labels,
        aleatoric_positive=aleatoric_positive,
        epistemic_positive=epistemic_positive,
        ood_positive=ood_positive,
    )

    clf_rows = evaluate_three_way_classification(
        signal_table=signal_table,
        eval_group_labels=eval_group_labels,
        device=device,
        seed=seed,
        train_fraction=0.5,
    )

    per_sample_csv_path = results_dir / "per_sample_signals.csv"
    save_per_sample_csv(
        per_sample_csv_path,
        eval_group_labels,
        eval_clean_labels,
        eval_is_noisy,
        signal_table,
        GROUP_NAMES,
        eval_noisy_labels=eval_noisy_labels,
        eval_dataset_index=eval_dataset_index,
    )

    return {
        "auroc_rows": auroc_rows,
        "one_vs_rest_auroc": one_vs_rest,
        "clf_rows": clf_rows,
        "per_sample_csv_path": per_sample_csv_path,
        "eval_group_labels": eval_group_labels,
    }


__all__ = [
    "EvalSignalConfig",
    "GROUP_NAMES",
    "collect_uncertainty_signals",
    "score_uncertainty_signals",
]
