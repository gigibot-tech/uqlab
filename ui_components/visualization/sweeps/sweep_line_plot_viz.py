"""
Streamlit 3-line sweep plot — thin UI over :mod:`uqlab.evaluation.pipeline.sweep_line_plot`.

Left Y: primary pool mean (swept axis) + optional dashed mirror; right Y: accuracy.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from uqlab.evaluation.pipeline.sweep_line_plot import (
    FACET_PARAM_LABELS,
    build_sweep_line_plot,
    run_ids_for_experiments,
)
from uqlab.runtime_paths import experiments_root
from uqlab.shared.notebook_utils.signals import SIGNAL_LABELS
from uqlab.ui_components.visualization.plot_export import render_plot_png_download


def figure_from_payload(data: dict[str, Any]) -> go.Figure:
    """Build dual-axis Plotly figure from :meth:`SweepLinePlotPayload.to_dict`."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    traces = data["traces"]
    shared_x = next((t.get("x") for t in traces if t.get("x")), None)
    for trace in traces:
        secondary = trace.get("yaxis") == "right"
        fig.add_trace(
            go.Scatter(
                x=trace.get("x") or shared_x or [],
                y=trace["y"],
                name=trace["name"],
                mode="lines+markers",
                line=dict(
                    color=trace.get("color", "#333"),
                    dash=trace.get("dash", "solid"),
                    width=2,
                ),
                marker=dict(size=7),
            ),
            secondary_y=secondary,
        )
    fig.update_layout(
        height=400,
        margin=dict(t=40, r=60, b=48, l=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text=data["x_label"])
    fig.update_yaxes(title_text=data["y_left_title"], secondary_y=False)
    fig.update_yaxes(
        title_text=data["y_right_title"],
        range=[0, 1],
        secondary_y=True,
    )
    return fig


def _safe_key(prefix: str) -> str:
    return re.sub(r"[^\w]", "_", prefix)[:48]


def _format_facet_caption(facet_filters: dict[str, Any]) -> str:
    if not facet_filters:
        return ""
    parts = []
    for col, val in facet_filters.items():
        label = FACET_PARAM_LABELS.get(col, col)
        parts.append(f"{label}={val}")
    return " · ".join(parts)


def render_plot_config_panel(plot_data: dict[str, Any]) -> None:
    """Show plot spec above the chart: summary always visible, JSON in nested expander."""
    cfg = plot_data.get("plot_config") or {}
    if not cfg:
        return

    x_axis = cfg.get("x_axis") or {}
    signal = cfg.get("signal") or {}
    pools = cfg.get("pools") or {}
    facet_filters = cfg.get("facet_filters") or {}

    summary_parts = [
        f"**X** `{x_axis.get('column', plot_data.get('x_col'))}` ({x_axis.get('label', plot_data.get('x_label'))})",
        f"**signal** `{signal.get('id', plot_data.get('signal'))}`",
        f"**primary pool** `{pools.get('primary', plot_data.get('primary_pool'))}`",
        f"**points** {cfg.get('points', plot_data.get('points'))}",
    ]
    if facet_filters:
        summary_parts.append(f"**facet** {_format_facet_caption(facet_filters)}")
    if cfg.get("architecture_filter"):
        summary_parts.append(f"**architecture** `{cfg['architecture_filter']}`")
    if pools.get("has_mirror_line"):
        summary_parts.append("**mirror** dashed")
    elif pools.get("mirror_note") or plot_data.get("mirror_note"):
        summary_parts.append("**mirror** omitted")

    st.markdown("**Plot configuration** · " + " · ".join(summary_parts))
    run_ids = cfg.get("run_ids_plotted") or cfg.get("run_ids_requested") or []
    if run_ids:
        st.caption(f"Runs: `{', '.join(str(r) for r in run_ids)}`")

    with st.expander("Full spec (JSON)", expanded=False):
        st.json(cfg)


def render_sweep_line_plot(
    experiments: list[dict[str, Any]],
    *,
    key_prefix: str = "sweep_line",
    experiments_dir: Path | None = None,
    architecture: str | None = None,
    facet_filters: dict[str, Any] | None = None,
    show_signal_picker: bool = True,
    sweep_kind: str | None = None,
    quiet: bool = False,
) -> bool:
    """
    Render modular 3-line sweep plot with signal picker.

    Returns True if a plot was shown, False if skipped (missing data).
    """
    exp_dir = experiments_dir or experiments_root()
    run_ids = run_ids_for_experiments(experiments, experiments_dir=exp_dir)
    if len(run_ids) < 2:
        if not quiet:
            st.info(
                "Need at least **2 completed runs** with `results.pt` or `summary.json` on disk "
                "to draw the 3-line sweep plot."
            )
        return False

    safe = _safe_key(key_prefix)

    try:
        initial = build_sweep_line_plot(
            run_ids,
            exp_dir,
            signal=None,
            architecture=architecture,
            facet_filters=facet_filters,
            sweep_kind=sweep_kind,
        )
    except (ValueError, TypeError) as exc:
        if not quiet:
            st.warning(str(exc))
        return False
    except Exception as exc:
        if not quiet:
            st.warning(f"Could not build sweep plot: {exc}")
        return False

    init_d = initial.to_dict()
    available = init_d["available_signals"]
    default_idx = available.index(init_d["signal"]) if init_d["signal"] in available else 0

    facet_note = _format_facet_caption(init_d.get("facet_filters") or {})
    pool_caption = init_d.get("pool_caption") or (
        "Left: **pool means** (primary = swept axis; dashed = mirror when present)"
    )
    st.caption(
        f"X: **{init_d['x_label']}** · {pool_caption} · Right: **accuracy**"
        + (f" · Slice: {facet_note}" if facet_note else "")
    )

    signal = init_d["signal"]
    if show_signal_picker:
        signal = st.selectbox(
            "Uncertainty signal (4th dimension)",
            available,
            index=default_idx,
            format_func=lambda s: f"{SIGNAL_LABELS.get(s, s)} (`{s}`)",
            key=f"{safe}_signal",
        )

    if signal != init_d["signal"]:
        try:
            payload = build_sweep_line_plot(
                run_ids,
                exp_dir,
                signal=signal,
                architecture=architecture,
                facet_filters=facet_filters,
                sweep_kind=sweep_kind,
            )
            plot_data = payload.to_dict()
        except (ValueError, TypeError) as exc:
            if not quiet:
                st.warning(str(exc))
            return False
        except Exception as exc:
            if not quiet:
                st.warning(f"Could not build sweep plot: {exc}")
            return False
    else:
        plot_data = init_d

    render_plot_config_panel(plot_data)

    fig = figure_from_payload(plot_data)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"{safe}_chart",
    )
    render_plot_png_download(
        fig,
        plot_data,
        key_prefix=f"{safe}_export",
        filename_prefix="sweep_line_plot",
    )
    mirror_note = plot_data.get("mirror_note")
    tail = f" · default signal: `{plot_data['default_signal']}`"
    if mirror_note:
        tail = f" · {mirror_note}{tail}"
    st.caption(
        f"{plot_data['signal_label']} · {plot_data['points']} point(s) on **{plot_data['x_label']}** "
        f"({plot_data['sweep_kind'].replace('_', ' ')}){tail}"
    )
    return True
