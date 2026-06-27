"""Config-implicit eval-pool expectations for sweep line plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from uqlab.data.benchmark_axes import (
    expects_aleatoric_eval,
    expects_epistemic_eval,
)
from uqlab.shared.config.classification import parse_under_supported_classes
from uqlab.run_artifacts import GROUP_ALEATORIC, GROUP_CLEAN, GROUP_EPISTEMIC, GROUP_OOD
from uqlab.shared.notebook_utils.metrics import ALEATORIC_COLOR, EPISTEMIC_COLOR

SWEEP_KIND_LABEL_NOISE = "label_noise"
SWEEP_KIND_DATASET_SIZE = "dataset_size"

PoolName = Literal["epistemic", "aleatoric"]

_POOL_DISPLAY = {
    "epistemic": "epistemic_like",
    "aleatoric": "aleatoric_like",
}


@dataclass(frozen=True)
class PoolExpectations:
    epistemic_pool_expected: bool
    aleatoric_pool_expected: bool
    ood_pool_expected: bool = False


@dataclass(frozen=True)
class SweepPlotTraceSpec:
    column: str
    label: str
    dash: str
    color: str
    pool: PoolName
    primary: bool


def pool_expectations_from_data_config(
    data: dict[str, Any],
    *,
    seed: int = 42,
) -> PoolExpectations:
    """Derive which eval pools should exist from run YAML ``data`` block."""
    partition_mode = str(data.get("partition_mode") or "legacy")
    if partition_mode == "four_region":
        return PoolExpectations(
            epistemic_pool_expected=True,
            aleatoric_pool_expected=True,
            ood_pool_expected=True,
        )
    under_classes = parse_under_supported_classes(
        data.get("under_supported_classes"),
        seed=seed,
    )
    return PoolExpectations(
        epistemic_pool_expected=expects_epistemic_eval(
            under_classes,
            under_train_per_class=int(data.get("under_train_per_class") or 10),
            regular_train_per_class=data.get("regular_train_per_class"),
        ),
        aleatoric_pool_expected=expects_aleatoric_eval(
            data.get("aleatoric_noise_percentage"),
        ),
    )


def primary_pool_for_sweep(sweep_kind: str) -> PoolName:
    if sweep_kind == SWEEP_KIND_LABEL_NOISE:
        return "aleatoric"
    if sweep_kind == SWEEP_KIND_DATASET_SIZE:
        return "epistemic"
    raise ValueError(f"Unknown sweep_kind: {sweep_kind!r}")


def secondary_pool_for_sweep(sweep_kind: str) -> PoolName:
    primary = primary_pool_for_sweep(sweep_kind)
    return "epistemic" if primary == "aleatoric" else "aleatoric"


def mean_column(signal: str, pool: PoolName) -> str:
    return f"{signal}_mean_{pool}"


def pool_has_values(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return bool(df[column].notna().any())


def list_plottable_signals_for_sweep(
    df: pd.DataFrame,
    sweep_kind: str,
) -> list[str]:
    """Signals with a non-empty primary-pool mean column for this sweep axis."""
    from uqlab.run_artifacts import FAST_PILOT_SIGNAL_NAMES
    from uqlab.shared.config.signals import signal_id_column_candidates

    if df.empty:
        return []
    primary = primary_pool_for_sweep(sweep_kind)
    available: list[str] = []
    for signal in FAST_PILOT_SIGNAL_NAMES:
        for candidate in signal_id_column_candidates(signal):
            col = mean_column(candidate, primary)
            if pool_has_values(df, col):
                available.append(signal)
                break
    return available


def resolve_sweep_plot_traces(
    signal: str,
    sweep_kind: str,
    plot_df: pd.DataFrame,
) -> list[SweepPlotTraceSpec]:
    """Primary solid line + optional dashed mirror when artifact column has data."""
    from uqlab.evaluation.signals.catalog import normalize_signal_id
    from uqlab.shared.config.signals import signal_id_column_candidates

    primary = primary_pool_for_sweep(sweep_kind)
    secondary = secondary_pool_for_sweep(sweep_kind)
    traces: list[SweepPlotTraceSpec] = []

    def _resolve_column(pool: PoolName) -> str | None:
        for candidate in signal_id_column_candidates(normalize_signal_id(signal)):
            col = mean_column(candidate, pool)
            if pool_has_values(plot_df, col):
                return col
        return None

    primary_col = _resolve_column(primary)
    if primary_col is not None:
        traces.append(
            SweepPlotTraceSpec(
                column=primary_col,
                label=f"Mean on {_POOL_DISPLAY[primary]} eval pack (swept axis)",
                dash="solid",
                color=ALEATORIC_COLOR if primary == "aleatoric" else EPISTEMIC_COLOR,
                pool=primary,
                primary=True,
            )
        )

    secondary_col = _resolve_column(secondary)
    if secondary_col is not None:
        traces.append(
            SweepPlotTraceSpec(
                column=secondary_col,
                label=f"Mean on {_POOL_DISPLAY[secondary]} eval pack (fixed mirror)",
                dash="dash",
                color=EPISTEMIC_COLOR if secondary == "epistemic" else ALEATORIC_COLOR,
                pool=secondary,
                primary=False,
            )
        )

    if not traces:
        raise ValueError(
            f"No pool-mean columns for signal {signal!r} on {sweep_kind} sweep "
            f"(need {primary_col!r} with at least one value)."
        )
    return traces


def sweep_pool_caption(
    sweep_kind: str,
    *,
    has_mirror: bool,
    mirror_note: str | None = None,
) -> str:
    primary = primary_pool_for_sweep(sweep_kind)
    primary_label = _POOL_DISPLAY[primary]
    if sweep_kind == SWEEP_KIND_LABEL_NOISE:
        base = (
            f"Left Y = mean on **{primary_label}** eval pool (label noise swept). "
            "Dashed line = fixed epistemic mirror when clean under-class eval exists."
        )
    else:
        base = (
            f"Left Y = mean on **{primary_label}** eval pool (under-train swept). "
            "Dashed line = aleatoric mirror when label-noise eval exists."
        )
    if has_mirror:
        return base
    if mirror_note:
        return f"{base} {mirror_note}"
    return (
        f"{base} Mirror line omitted when that eval pool has no samples "
        f"(e.g. 100% label noise → no epistemic pool)."
    )


def mirror_omitted_note(
    df: pd.DataFrame,
    signal: str,
    sweep_kind: str,
) -> str | None:
    secondary = secondary_pool_for_sweep(sweep_kind)
    col = mean_column(signal, secondary)
    if pool_has_values(df, col):
        return None
    if (
        sweep_kind == SWEEP_KIND_LABEL_NOISE
        and "noise_percent" in df.columns
        and df["noise_percent"].notna().any()
        and float(df["noise_percent"].max()) >= 100.0
    ):
        return "**100% noise** — no clean epistemic eval samples (mirror omitted)."
    return f"No **{_POOL_DISPLAY[secondary]}** pool means in artifacts (mirror omitted)."


def eval_pool_counts_from_results_dir(results_dir: Path) -> dict[str, int]:
    """Count eval samples per group from ``results.pt`` (for debug / sanity checks)."""
    results_pt = results_dir / "results.pt"
    if not results_pt.is_file():
        return {}

    import torch

    data = torch.load(results_pt, map_location="cpu", weights_only=False)
    labels = data.get("eval_group_labels")
    if labels is None:
        return {}
    if hasattr(labels, "cpu"):
        labels = labels.cpu()
    counts = {
        "clean": int((labels == GROUP_CLEAN).sum().item()),
        "aleatoric_like": int((labels == GROUP_ALEATORIC).sum().item()),
        "epistemic_like": int((labels == GROUP_EPISTEMIC).sum().item()),
        "ood_like": int((labels == GROUP_OOD).sum().item()),
    }
    return counts


def eval_pool_debug_rows(
    run_ids: list[str],
    experiments_dir: Path,
) -> list[dict[str, Any]]:
    """Per-run eval pool counts + config expectations for UI debug table."""
    import yaml

    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        run_dir = experiments_dir / run_id
        results_dir = run_dir / "results"
        cfg_path = run_dir / "config.yaml"
        cfg: dict[str, Any] = {}
        if cfg_path.is_file():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                cfg = {}
        data = cfg.get("data") or {}
        seed = int(cfg.get("seed") or 42)
        expectations = pool_expectations_from_data_config(data, seed=seed)
        counts = eval_pool_counts_from_results_dir(results_dir)
        rows.append(
            {
                "run_id": run_id,
                "partition_mode": data.get("partition_mode", "legacy"),
                "noise_percent": data.get("aleatoric_noise_percentage"),
                "under_train_per_class": data.get("under_train_per_class"),
                "epistemic_expected": expectations.epistemic_pool_expected,
                "aleatoric_expected": expectations.aleatoric_pool_expected,
                "ood_expected": expectations.ood_pool_expected,
                **{
                    f"n_{k}": counts.get(k, 0)
                    for k in ("clean", "aleatoric_like", "epistemic_like", "ood_like")
                },
            }
        )
    return rows
