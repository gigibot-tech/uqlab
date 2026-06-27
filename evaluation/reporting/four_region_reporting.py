"""Notebook helpers for four-region benchmark plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

FOUR_REGION_GROUP_ORDER: tuple[str, ...] = (
    "clean",
    "aleatoric_like",
    "epistemic_like",
    "ood_like",
)

_NON_SIGNAL_COLS = frozenset(
    {"group", "dataset_index", "clean_label", "noisy_label", "is_noisy"}
)


def four_region_signals_dataframe(results_dir: Path) -> pd.DataFrame:
    """Load ``per_sample_signals.csv`` from a four-region run directory."""
    csv_path = Path(results_dir) / "per_sample_signals.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"No per_sample_signals.csv under {results_dir}")
    df = pd.read_csv(csv_path)
    if "group" not in df.columns:
        raise ValueError(f"{csv_path} has no 'group' column")
    return df


def list_four_region_signal_columns(df: pd.DataFrame) -> list[str]:
    """Numeric signal columns suitable for per-group plots."""
    return [
        c
        for c in df.columns
        if c not in _NON_SIGNAL_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def plot_four_region_metrics_by_group(
    df: pd.DataFrame,
    metrics: Sequence[str],
    out_dir: Path,
    *,
    title_prefix: str = "",
    kind: str = "box",
) -> list[Path]:
    """One figure per metric: distribution of values across the four eval pools."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    order = [g for g in FOUR_REGION_GROUP_ORDER if g in df["group"].unique()]
    order += [g for g in df["group"].unique() if g not in order]
    plot_df = df.copy()
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=order, ordered=True)

    for metric in metrics:
        if metric not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        grouped = [plot_df.loc[plot_df["group"] == g, metric].dropna() for g in order]
        if kind == "violin":
            parts = ax.violinplot(grouped, showmeans=True, showmedians=False)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order, rotation=20, ha="right")
            for body in parts["bodies"]:
                body.set_alpha(0.7)
        else:
            plot_df.boxplot(column=metric, by="group", ax=ax, grid=False)
            fig.suptitle("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation=20)

        prefix = f"{title_prefix} — " if title_prefix else ""
        ax.set_title(f"{prefix}{metric}", fontsize=10)
        ax.set_ylabel(metric)
        fig.tight_layout()
        out_path = out_dir / f"{metric}_by_group.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


def plot_all_four_region_metrics(
    results_dir: Path,
    out_dir: Path | None = None,
    *,
    metrics: Iterable[str] | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Convenience: load CSV from ``results_dir`` and plot every (or selected) metric."""
    df = four_region_signals_dataframe(results_dir)
    cols = list(metrics) if metrics is not None else list_four_region_signal_columns(df)
    target = out_dir or (Path(results_dir) / "analysis" / "four_region_metrics")
    return plot_four_region_metrics_by_group(
        df,
        cols,
        target,
        title_prefix=title_prefix or Path(results_dir).name,
    )


__all__ = [
    "FOUR_REGION_GROUP_ORDER",
    "four_region_signals_dataframe",
    "list_four_region_signal_columns",
    "plot_all_four_region_metrics",
    "plot_four_region_metrics_by_group",
]
