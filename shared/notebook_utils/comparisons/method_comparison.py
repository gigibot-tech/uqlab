"""Matplotlib method-uncertainty comparison for Jupyter validation notebooks.

Mirrors ``method_comparison_plotly``.  Row 1 uses :data:`ARCHITECTURE_ROW`;
signal rows use a pluggable :class:`~metric_specs.MetricSpec`.
"""

from __future__ import annotations

import pandas as pd

from .metric_specs import (
    ACCURACY_COLOR,
    ARCHITECTURE_ROW,
    MetricSpec,
    ORTHOGONAL_AUROC_COLOR,
    PRIMARY_AUROC_COLOR,
    UNCERTAINTY_DECOMPOSITION,
    resolve_columns,
)
from ..signals import (
    SIGNAL_LABELS,
    disentanglement_label,
    format_method_architecture_mapping,
    get_method_architecture_mapping,
    get_top_n_signals,
    present_architectures,
    present_disentanglements,
    resolve_x_col,
    sweep_to_auroc_type,
)

DEFAULT_TOP_N_SIGNALS = 4

_MPL_MARKERS = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
}


def _add_metric_tile_mpl(
    ax,
    data: pd.DataFrame,
    x_col: str,
    spec: MetricSpec,
    *,
    signal: str | None = None,
    auroc_type: str = "epistemic",
) -> object | None:
    """Draw one tile from *spec*. Returns twin axis for right-side overlays, or None."""
    if data.empty or x_col not in data.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        return None

    data = data.sort_values(x_col)
    cols = resolve_columns(spec, signal=signal, auroc_type=auroc_type)
    if not cols:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")
        return None

    plotted_left = False
    markers = spec.markers if spec.markers else ("circle",) * len(cols)
    if len(markers) < len(cols):
        markers = tuple(list(markers) + ["circle"] * (len(cols) - len(markers)))

    for col_name, label, color, marker_key in zip(cols, spec.labels, spec.colors, markers):
        if col_name not in data.columns:
            continue
        valid = data[[x_col, col_name]].dropna()
        if valid.empty:
            continue
        ax.plot(
            valid[x_col],
            valid[col_name],
            color=color,
            marker=_MPL_MARKERS.get(marker_key, "o"),
            linewidth=2,
            label=label,
        )
        plotted_left = True

    if plotted_left:
        ax.set_ylim(*spec.y_range)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray")

    twin = None
    need_twin = (
        spec.plot_accuracy
        or (spec.plot_auroc_overlay and signal is not None)
    )
    if need_twin:
        twin = ax.twinx()
        if spec.plot_accuracy and "accuracy" in data.columns:
            valid_acc = data[[x_col, "accuracy"]].dropna()
            if not valid_acc.empty:
                twin.plot(
                    valid_acc[x_col],
                    valid_acc["accuracy"],
                    color=ACCURACY_COLOR,
                    marker="D",
                    linewidth=2,
                    linestyle="--",
                    label="Accuracy",
                )
        if spec.plot_auroc_overlay and signal is not None:
            other_type = "aleatoric" if auroc_type == "epistemic" else "epistemic"
            for auc_col, auc_label, auc_color, alpha in (
                (f"{signal}_{auroc_type}_auroc", f"{auroc_type.title()} AUROC", PRIMARY_AUROC_COLOR, 0.7),
                (f"{signal}_{other_type}_auroc", f"{other_type.title()} AUROC (orth.)", ORTHOGONAL_AUROC_COLOR, 0.45),
            ):
                if auc_col not in data.columns:
                    continue
                valid_auc = data[[x_col, auc_col]].dropna()
                if not valid_auc.empty:
                    twin.plot(
                        valid_auc[x_col],
                        valid_auc[auc_col],
                        color=auc_color,
                        linestyle=":",
                        linewidth=1.2,
                        alpha=alpha,
                        label=auc_label,
                    )
        if twin.get_lines():
            twin.set_ylim(0, 1)
            if spec.y_range == (0.0, 1.0):
                twin.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.3)
            twin.tick_params(axis="y", labelsize=8)
        else:
            twin = None

    return twin


def plot_method_uncertainty_comparison(
    df: pd.DataFrame,
    x_col: str | None = None,
    sweep_type: str = "dataset_size",
    *,
    architectures: list[str] | None = None,
    top_n_signals: int = DEFAULT_TOP_N_SIGNALS,
    signal_metric: MetricSpec = UNCERTAINTY_DECOMPOSITION,
    show: bool = True,
    print_mapping: bool = True,
):
    """Build the comparison figure (architectures = columns, signals = rows)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if df.empty:
        print("Warning: empty dataframe, cannot plot method comparison.")
        return None

    if x_col is None:
        x_col = resolve_x_col(df, sweep_type)

    col_archs: list[str] = list(architectures) if architectures else present_architectures(df)
    if not col_archs:
        print("Warning: no architectures found in dataframe.")
        return None

    if print_mapping:
        mapping = get_method_architecture_mapping(df)
        print(format_method_architecture_mapping(mapping))

    dis_rows = present_disentanglements(df)
    if not dis_rows:
        dis_rows = [None]

    n_arch_rows = len(dis_rows)
    n_cols = len(col_archs)
    auroc_type = sweep_to_auroc_type(sweep_type)
    row_signals = get_top_n_signals(df, n=max(top_n_signals, 1), signal_type=auroc_type)
    n_signal_rows = len(row_signals)
    n_rows = n_arch_rows + n_signal_rows

    x_label = (
        "Dataset Size (samples/class)"
        if sweep_type == "dataset_size"
        else "Noise Rate (%)"
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex="col",
        squeeze=False,
    )
    accuracy_axes: list = []

    for row_idx, dis in enumerate(dis_rows):
        row_df = (
            df[df["disentanglement"] == dis].copy()
            if dis and "disentanglement" in df.columns
            else df
        )
        dis_lbl = disentanglement_label(dis) if dis else "Uncertainty"
        axes[row_idx, 0].set_ylabel(f"{dis_lbl}\n{ARCHITECTURE_ROW.y_title}", fontsize=9)

        for col_idx, arch in enumerate(col_archs):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(arch, fontsize=10)
            arch_df = (
                row_df[row_df["architecture"] == arch]
                if "architecture" in row_df.columns
                else pd.DataFrame()
            )
            acc_ax = _add_metric_tile_mpl(ax, arch_df, x_col, ARCHITECTURE_ROW, auroc_type=auroc_type)
            if acc_ax is not None:
                accuracy_axes.append(acc_ax)

    for sig_row_offset, (signal, mean_auroc) in enumerate(row_signals):
        sig_row_idx = n_arch_rows + sig_row_offset
        sig_lbl = SIGNAL_LABELS.get(signal, signal)
        axes[sig_row_idx, 0].set_ylabel(
            f"{sig_lbl}\n(mean AUROC {mean_auroc:.3f})", fontsize=9
        )
        for col_idx, arch in enumerate(col_archs):
            ax = axes[sig_row_idx, col_idx]
            arch_df = df[df["architecture"] == arch] if "architecture" in df.columns else pd.DataFrame()
            sig_acc_ax = _add_metric_tile_mpl(
                ax,
                arch_df,
                x_col,
                signal_metric,
                signal=signal,
                auroc_type=auroc_type,
            )
            if sig_acc_ax is not None:
                accuracy_axes.append(sig_acc_ax)
            if sig_row_idx == n_rows - 1:
                ax.set_xlabel(x_label, fontsize=8)

    if accuracy_axes:
        for acc_ax in accuracy_axes[1:]:
            acc_ax.sharey(accuracy_axes[0])
        accuracy_axes[-1].set_ylabel("Accuracy", color=ACCURACY_COLOR)

    legend_handles = [
        Line2D([0], [0], color=ARCHITECTURE_ROW.colors[0], marker="o", linewidth=2,
               label=ARCHITECTURE_ROW.labels[0]),
        Line2D([0], [0], color=ARCHITECTURE_ROW.colors[1], marker="s", linewidth=2,
               label=ARCHITECTURE_ROW.labels[1]),
        Line2D([0], [0], color=ACCURACY_COLOR, marker="D", linewidth=2,
               linestyle="--", label="Accuracy (right axis)"),
    ]
    if signal_metric.plot_auroc_overlay:
        legend_handles.extend([
            Line2D([0], [0], color=PRIMARY_AUROC_COLOR, marker="o", linewidth=2,
                   label=f"{auroc_type.title()} AUROC"),
            Line2D([0], [0], color=ORTHOGONAL_AUROC_COLOR, linestyle=":", linewidth=1,
                   label="Orthogonal AUROC"),
        ])
    elif signal_metric.name == "auroc_only":
        legend_handles.append(
            Line2D([0], [0], color=PRIMARY_AUROC_COLOR, marker="o", linewidth=2, label="AUROC"),
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(len(legend_handles), 5),
        bbox_to_anchor=(0.5, 1.01),
        fontsize=9,
        frameon=True,
    )

    view_suffix = "" if signal_metric.name == "uncertainty_decomposition" else f" ({signal_metric.name})"
    fig.suptitle(
        f"Method Uncertainty Comparison — {sweep_type.replace('_', ' ').title()} Sweep{view_suffix}",
        fontsize=13,
        fontweight="bold",
        y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    fig._method_architecture_mapping = {  # type: ignore[attr-defined]
        "architectures": col_archs,
        "disentanglements_rendered": dis_rows,
        "signals_rendered": [s for s, _ in row_signals],
        "signal_metric": signal_metric.name,
    }

    if show:
        plt.show()
    return fig
