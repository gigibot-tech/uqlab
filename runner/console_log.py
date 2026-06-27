"""Stdout-only reporting for the experiment runner (no disk writes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from uqlab.run_artifacts import WrittenArtifacts


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


def log_run_data_context(
    *,
    device: torch.device,
    results_dir: Path,
    train_dataset,
    clean_eval_pack: dict,
    aleatoric_eval_pack: dict,
    epistemic_eval_pack: dict,
    ood_eval_pack: dict | None,
) -> None:
    """Log device, paths, and eval group sizes after data prep."""
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


def log_zwischen_dir(results_dir: Path) -> None:
    print(f"✅ Zwischenergebnisse: {results_dir / 'zwischen'}/")


def log_auroc_by_family(
    *,
    auroc_rows: list[tuple],
    clf_rows: list[tuple[str, float]],
    alea_skip: str | None,
    epis_skip: str | None,
    ood_skip: str | None,
    eval_group_labels: torch.Tensor,
) -> None:
    """Print AUROC tables grouped by signal family."""
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


def log_run_complete(
    written: WrittenArtifacts,
    *,
    results_dir: Path,
    eval_summary: dict,
    summary: dict,
) -> None:
    """List all artifact paths in write order and print AUROC tables."""
    one_vs_rest = eval_summary.get("one_vs_rest_auroc") or []
    alea_skip = one_vs_rest[0].get("aleatoric_skip_reason") if one_vs_rest else None
    epis_skip = one_vs_rest[0].get("epistemic_skip_reason") if one_vs_rest else None
    ood_skip = one_vs_rest[0].get("ood_skip_reason") if one_vs_rest else None

    try:
        from uqlab.run_artifacts import metrics_row_from_run, print_run_metrics_summary

        print("\n" + "=" * 70)
        print("SIGNAL MEANS & AUROC (all uncertainties)")
        print("=" * 70)
        print_run_metrics_summary(metrics_row_from_run(results_dir))
    except ImportError:
        pass

    auroc_rows = eval_summary.get("auroc_rows") or []
    clf_rows = eval_summary.get("clf_rows") or []
    eval_group_labels = eval_summary.get("eval_group_labels")
    if eval_group_labels is not None:
        log_auroc_by_family(
            auroc_rows=auroc_rows,
            clf_rows=clf_rows,
            alea_skip=alea_skip,
            epis_skip=epis_skip,
            ood_skip=ood_skip,
            eval_group_labels=eval_group_labels,
        )

    print("\n" + "=" * 70)
    print("SAVED ARTIFACTS (disk)")
    print("=" * 70)
    for label, path in written.labeled_paths():
        if path is not None:
            print(f"  {label}: {path}")


__all__ = [
    "log_auroc_by_family",
    "log_run_complete",
    "log_run_data_context",
    "log_zwischen_dir",
]
