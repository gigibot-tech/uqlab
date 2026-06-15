"""Pluggable metric definitions for the method-uncertainty comparison grid.

Each :class:`MetricSpec` describes what one tile plots (columns, colours, axes).
The figure builders in ``method_comparison_plotly`` and ``method_comparison``
only handle layout; all "what to draw" lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Shared palette (no plotly/matplotlib imports).
EPISTEMIC_COLOR = "#2ecc71"
ALEATORIC_COLOR = "#3498db"
ACCURACY_COLOR = "#e67e22"
PRIMARY_AUROC_COLOR = "#9b59b6"
ORTHOGONAL_AUROC_COLOR = "#7f8c8d"


@dataclass(frozen=True)
class MetricSpec:
    """Defines what one tile of the comparison grid plots."""

    name: str
    columns: tuple[str, ...]
    labels: tuple[str, ...]
    colors: tuple[str, ...]
    markers: tuple[str, ...] = ()
    y_range: tuple[float, float] = (0.0, 2.5)
    y_title: str = "Uncertainty"
    plot_accuracy: bool = True
    plot_auroc_overlay: bool = False  # opt-in; do not mix AUROC on uncertainty tiles


def resolve_columns(
    spec: MetricSpec,
    *,
    signal: str | None = None,
    auroc_type: str = "epistemic",
) -> list[str]:
    """Format column patterns for a tile (Row 1 passes ``signal=None``)."""
    out: list[str] = []
    for col in spec.columns:
        if "{signal}" in col or "{auroc_type}" in col:
            if signal is None:
                continue
            out.append(col.format(signal=signal, auroc_type=auroc_type))
        else:
            out.append(col)
    return out


# Row 1: BMA-style epistemic / aleatoric decomposition (no per-signal placeholders).
ARCHITECTURE_ROW = MetricSpec(
    name="architecture_row",
    columns=("mean_epistemic_uncertainty", "mean_aleatoric_uncertainty"),
    labels=("Epistemic (mutual info, all eval)", "Aleatoric (entropy − MI, all eval)"),
    colors=(EPISTEMIC_COLOR, ALEATORIC_COLOR),
    markers=("circle", "square"),
    y_range=(0.0, 2.5),
    y_title="Uncertainty",
    plot_accuracy=True,
    plot_auroc_overlay=False,
)

# Signal rows: per-signal mean on epistemic vs aleatoric eval pools (left axis only).
# Use AUROC_ONLY for discrimination; mixing AUROC on this tile was confusing (different scale/meaning).
UNCERTAINTY_DECOMPOSITION = MetricSpec(
    name="uncertainty_decomposition",
    columns=("{signal}_mean_epistemic", "{signal}_mean_aleatoric"),
    labels=("Mean on epistemic eval pack", "Mean on aleatoric eval pack"),
    colors=(EPISTEMIC_COLOR, ALEATORIC_COLOR),
    markers=("circle", "square"),
    y_range=(0.0, 2.5),
    y_title="Uncertainty",
    plot_accuracy=True,
    plot_auroc_overlay=False,
)

# Signal rows: AUROC as the primary left-axis metric.
AUROC_ONLY = MetricSpec(
    name="auroc_only",
    columns=("{signal}_{auroc_type}_auroc",),
    labels=("AUROC",),
    colors=(PRIMARY_AUROC_COLOR,),
    markers=("circle",),
    y_range=(0.0, 1.0),
    y_title="AUROC",
    plot_accuracy=True,
    plot_auroc_overlay=False,
)
