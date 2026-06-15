"""Plotly method-uncertainty comparison (Streamlit and optional notebook export).

Layout: every row shares the SAME columns = architectures.
Row 1 always uses :data:`ARCHITECTURE_ROW` (epistemic + aleatoric + accuracy).
Signal rows use a pluggable :class:`~metric_specs.MetricSpec` (default
:data:`UNCERTAINTY_DECOMPOSITION`; Streamlit "Signal AUROC" view uses
:data:`AUROC_ONLY`).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    get_top_n_signals,
    present_architectures,
    present_disentanglements,
    resolve_x_col,
    sweep_to_auroc_type,
)

DEFAULT_TOP_N_SIGNALS = 4

_PLOTLY_MARKERS = {
    "circle": dict(size=6),
    "square": dict(size=6, symbol="square"),
    "diamond": dict(size=6, symbol="diamond"),
}


# ------------------------------------------------------------------
# Low-level trace helpers
# ------------------------------------------------------------------

def _annotate_empty_subplot(fig: go.Figure, row: int, col: int, message: str = "No data") -> None:
    fig.add_annotation(
        text=message,
        xref="x domain",
        yref="y domain",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=12, color="gray"),
        row=row,
        col=col,
    )


def _filter_by_disentanglement(df: pd.DataFrame, disentanglement: str | None) -> pd.DataFrame:
    if disentanglement is None or "disentanglement" not in df.columns:
        return df
    return df[df["disentanglement"] == disentanglement].copy()


def _slice_architecture(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    if "architecture" not in df.columns:
        return pd.DataFrame()
    return df[df["architecture"] == arch]


def _add_metric_tile_plotly(
    fig: go.Figure,
    data: pd.DataFrame,
    x_col: str,
    row: int,
    col: int,
    spec: MetricSpec,
    *,
    signal: str | None = None,
    arch: str = "",
    auroc_type: str = "epistemic",
    show_legend: bool = False,
) -> bool:
    """Draw one tile from *spec* (left-axis metrics + optional right-axis overlays)."""
    if data.empty or x_col not in data.columns:
        return False

    data = data.sort_values(x_col)
    cols = resolve_columns(spec, signal=signal, auroc_type=auroc_type)
    if not cols:
        return False

    added = False
    markers = spec.markers if spec.markers else ("circle",) * len(cols)
    if len(markers) < len(cols):
        markers = tuple(list(markers) + ["circle"] * (len(cols) - len(markers)))

    for col_name, label, color, marker_key in zip(cols, spec.labels, spec.colors, markers):
        if col_name not in data.columns:
            continue
        valid = data[[x_col, col_name]].dropna()
        if valid.empty:
            continue
        marker = _PLOTLY_MARKERS.get(marker_key, _PLOTLY_MARKERS["circle"])
        hover = f"<b>{signal or label}</b><br>{arch}<br>{label}: %{{y:.3f}}<extra></extra>"
        fig.add_trace(
            go.Scatter(
                x=valid[x_col],
                y=valid[col_name],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=marker,
                showlegend=show_legend,
                legendgroup=label,
                hovertemplate=hover,
            ),
            row=row,
            col=col,
            secondary_y=False,
        )
        added = True

    if spec.plot_accuracy and "accuracy" in data.columns:
        valid_acc = data[[x_col, "accuracy"]].dropna()
        if not valid_acc.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_acc[x_col],
                    y=valid_acc["accuracy"],
                    mode="lines+markers",
                    name="Accuracy",
                    line=dict(color=ACCURACY_COLOR, width=2, dash="dot"),
                    marker=dict(size=6, symbol="diamond"),
                    showlegend=show_legend,
                    legendgroup="accuracy",
                    hovertemplate=(
                        f"<b>{signal or 'Accuracy'}</b><br>{arch}<br>"
                        "Accuracy: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
            added = True

    if spec.plot_auroc_overlay and signal is not None:
        other_type = "aleatoric" if auroc_type == "epistemic" else "epistemic"
        for auc_col, auc_label, auc_color, opacity, lg in (
            (f"{signal}_{auroc_type}_auroc", f"{auroc_type.title()} AUROC", PRIMARY_AUROC_COLOR, 0.7, "auroc_primary"),
            (f"{signal}_{other_type}_auroc", f"{other_type.title()} AUROC (orthogonal)", ORTHOGONAL_AUROC_COLOR, 0.45, "auroc_orthogonal"),
        ):
            if auc_col not in data.columns:
                continue
            valid_auc = data[[x_col, auc_col]].dropna()
            if valid_auc.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=valid_auc[x_col],
                    y=valid_auc[auc_col],
                    mode="lines",
                    name=auc_label,
                    line=dict(color=auc_color, width=1.5, dash="dot"),
                    opacity=opacity,
                    showlegend=show_legend,
                    legendgroup=lg,
                    hovertemplate=(
                        f"<b>{signal}</b><br>{arch}<br>"
                        f"{auc_label}: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
            added = True

    return added


def _resolve_columns(
    df: pd.DataFrame,
    architectures: list[str] | None,
) -> list[str]:
    if architectures:
        return list(architectures)
    return present_architectures(df)


# ------------------------------------------------------------------
# Main figure builder
# ------------------------------------------------------------------

def create_method_uncertainty_comparison_figure(
    df: pd.DataFrame,
    x_col: str | None = None,
    sweep_type: str = "dataset_size",
    disentanglement: str | None = None,
    *,
    architectures: list[str] | None = None,
    top_n_signals: int = DEFAULT_TOP_N_SIGNALS,
    signal_metric: MetricSpec = UNCERTAINTY_DECOMPOSITION,
) -> go.Figure | None:
    """
    Build the adaptive method-uncertainty comparison figure.

    Row 1 always uses :data:`ARCHITECTURE_ROW`.  Signal rows use *signal_metric*.
    """
    if df.empty:
        return None

    if x_col is None:
        x_col = resolve_x_col(df, sweep_type)
    if x_col not in df.columns:
        return None

    col_archs = _resolve_columns(df, architectures)
    if not col_archs:
        return None
    n_cols = len(col_archs)

    plot_df = _filter_by_disentanglement(df, disentanglement)
    if plot_df.empty:
        return None

    dis_rows = present_disentanglements(plot_df)
    if not dis_rows:
        dis_rows = [None]

    n_arch_rows = len(dis_rows)
    auroc_type = sweep_to_auroc_type(sweep_type)
    row_signals = get_top_n_signals(plot_df, n=max(top_n_signals, 1), signal_type=auroc_type)
    n_signal_rows = len(row_signals)
    n_rows = n_arch_rows + n_signal_rows

    subplot_titles: list[str] = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            subplot_titles.append(col_archs[col_idx] if row_idx == 0 else "")

    specs = [[{"secondary_y": True}] * n_cols] * n_rows
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.07 if n_rows <= 4 else 0.05,
        horizontal_spacing=0.06,
    )

    x_label = (
        "Dataset Size (samples/class)"
        if sweep_type == "dataset_size"
        else "Noise Rate (%)"
    )

    # Architecture rows (always ARCHITECTURE_ROW).
    for row_idx, dis in enumerate(dis_rows, start=1):
        row_df = _filter_by_disentanglement(plot_df, dis)
        for col_idx, arch in enumerate(col_archs, start=1):
            arch_df = _slice_architecture(row_df, arch)
            show_legend = row_idx == 1 and col_idx == 1
            if not _add_metric_tile_plotly(
                fig,
                arch_df,
                x_col,
                row_idx,
                col_idx,
                ARCHITECTURE_ROW,
                arch=arch,
                auroc_type=auroc_type,
                show_legend=show_legend,
            ):
                _annotate_empty_subplot(fig, row_idx, col_idx)

    # Signal rows (pluggable metric).
    legend_shown = False
    for sig_row_offset, (signal, _mean_auroc) in enumerate(row_signals):
        row_idx = n_arch_rows + 1 + sig_row_offset
        for col_idx, arch in enumerate(col_archs, start=1):
            arch_df = _slice_architecture(plot_df, arch)
            show_legend = not legend_shown
            drew = _add_metric_tile_plotly(
                fig,
                arch_df,
                x_col,
                row_idx,
                col_idx,
                signal_metric,
                signal=signal,
                arch=arch,
                auroc_type=auroc_type,
                show_legend=show_legend,
            )
            if drew and show_legend:
                legend_shown = True
            elif not drew:
                _annotate_empty_subplot(fig, row_idx, col_idx, "No data")

    y_lo, y_hi = signal_metric.y_range
    right_title = "Accuracy / AUROC" if signal_metric.plot_auroc_overlay else "Accuracy"

    for col_idx in range(1, n_cols + 1):
        for row_idx in range(1, n_arch_rows + 1):
            fig.update_yaxes(
                title_text=ARCHITECTURE_ROW.y_title if col_idx == 1 else "",
                range=list(ARCHITECTURE_ROW.y_range),
                row=row_idx,
                col=col_idx,
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text="Accuracy" if col_idx == n_cols else "",
                range=[0, 1],
                row=row_idx,
                col=col_idx,
                secondary_y=True,
            )
        for sig_row_offset in range(n_signal_rows):
            row_idx = n_arch_rows + 1 + sig_row_offset
            fig.update_yaxes(
                title_text=signal_metric.y_title if col_idx == 1 else "",
                range=[y_lo, y_hi],
                row=row_idx,
                col=col_idx,
                secondary_y=False,
            )
            fig.update_yaxes(
                title_text=right_title if col_idx == n_cols else "",
                range=[0, 1],
                row=row_idx,
                col=col_idx,
                secondary_y=True,
            )
        fig.update_xaxes(title_text=x_label, row=n_rows, col=col_idx)

    if n_rows > 0:
        step = 1.0 / n_rows
        row_centers = [1.0 - step * (i + 0.5) for i in range(n_rows)]
    else:
        row_centers = []

    for row_idx, dis in enumerate(dis_rows):
        if row_idx >= len(row_centers):
            break
        label = disentanglement_label(dis) if dis else "Uncertainty"
        fig.add_annotation(
            text=label,
            xref="paper",
            yref="paper",
            x=-0.05,
            y=row_centers[row_idx],
            showarrow=False,
            font=dict(size=11, color="#555"),
            textangle=-90,
            xanchor="right",
        )

    for sig_row_offset, (signal, mean_auroc) in enumerate(row_signals):
        row_idx = n_arch_rows + sig_row_offset
        if row_idx >= len(row_centers):
            break
        lbl = SIGNAL_LABELS.get(signal, signal)
        fig.add_annotation(
            text=(
                f"{lbl}<br>"
                f"<span style='font-size:9px;color:#888'>"
                f"mean {auroc_type[:4]}. AUROC: {mean_auroc:.3f}</span>"
            ),
            xref="paper",
            yref="paper",
            x=-0.05,
            y=row_centers[row_idx],
            showarrow=False,
            font=dict(size=11, color="#555"),
            textangle=-90,
            xanchor="right",
            align="center",
        )

    view_suffix = "" if signal_metric.name == "uncertainty_decomposition" else f" ({signal_metric.name})"
    fig.update_layout(
        height=max(280 * n_rows, 700),
        title_text=(
            f"Method Uncertainty Comparison — "
            f"{sweep_type.replace('_', ' ').title()} Sweep{view_suffix}"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.04,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=100, t=80, b=120),
        meta={
            "architectures": col_archs,
            "disentanglements_rendered": dis_rows,
            "signals_rendered": [s for s, _ in row_signals],
            "signal_metric": signal_metric.name,
            "x_col": x_col,
            "sweep_type": sweep_type,
        },
    )
    return fig


def display_plotly_figure(fig: go.Figure) -> None:
    """Streamlit if active, otherwise Plotly default viewer."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            import streamlit as st
            st.plotly_chart(fig, use_container_width=True)
            return
    except Exception:
        pass
    fig.show()


def plot_method_uncertainty_comparison(
    df: pd.DataFrame,
    x_col: str | None = None,
    sweep_type: str = "dataset_size",
    disentanglement: str | None = None,
    *,
    architectures: list[str] | None = None,
    top_n_signals: int = DEFAULT_TOP_N_SIGNALS,
    signal_metric: MetricSpec = UNCERTAINTY_DECOMPOSITION,
) -> None:
    """Build and display the interactive comparison figure."""
    if df.empty:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                import streamlit as st
                st.warning("No data available for method uncertainty comparison.")
                return
        except Exception:
            pass
        print("Warning: No data available for method uncertainty comparison.")
        return

    fig = create_method_uncertainty_comparison_figure(
        df,
        x_col=x_col,
        sweep_type=sweep_type,
        disentanglement=disentanglement,
        architectures=architectures,
        top_n_signals=top_n_signals,
        signal_metric=signal_metric,
    )
    if fig is not None:
        display_plotly_figure(fig)
