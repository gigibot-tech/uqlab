"""
Streamlit paper-style benchmark plot — three curves vs Percentage (0–1).

Primary visualization for disentanglement sweeps; pool-filtered uqlab diagnostic
plot lives in :mod:`sweep_line_plot_viz`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from uqlab.evaluation.reporting.paper_benchmark_plot import (
    build_paper_benchmark_plot,
)
from uqlab.runtime_paths import experiments_root
from uqlab.ui_components.visualization.plot_export import render_plot_png_download
from uqlab.ui_components.visualization.sweeps.sweep_line_plot_viz import run_ids_for_experiments


def figure_from_payload(data: dict[str, Any]) -> go.Figure:
    """Single-axis Plotly figure from :meth:`PaperBenchmarkPlotPayload.to_dict`."""
    fig = go.Figure()
    traces = data.get("traces") or []
    shared_x = next((t.get("x") for t in traces if t.get("x")), None) or []
    for trace in traces:
        fig.add_trace(
            go.Scatter(
                x=trace.get("x") or shared_x,
                y=trace["y"],
                name=trace["name"],
                mode="lines+markers",
                line=dict(
                    color=trace.get("color", "#333"),
                    dash=trace.get("dash", "solid"),
                    width=2,
                ),
                marker=dict(size=7),
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(t=48, r=40, b=48, l=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        yaxis=dict(range=[0, 1]),
    )
    fig.update_xaxes(title_text=data.get("x_label", "Percentage (0–1)"))
    fig.update_yaxes(title_text=data.get("y_label", "Score / global uncertainty mean"))
    return fig


def _safe_key(prefix: str) -> str:
    return re.sub(r"[^\w]", "_", prefix)[:48]


def _format_correlation_caption(corr: dict[str, Any]) -> str:
    method = "Spearman" if corr.get("rank") else "Pearson"
    primary = corr.get("primary_metric", "?")
    primary_val = corr.get("primary_correlation")
    if primary_val is None:
        return f"{method} ρ({primary}, scores): n/a"
    return f"{method} ρ({primary}, scores) = {primary_val:.3f} (paper arm)"


def render_paper_benchmark_plot(
    experiments: list[dict[str, Any]],
    *,
    sweep_kind: str,
    key_prefix: str = "paper_plot",
    experiments_dir: Path | None = None,
    quiet: bool = False,
) -> bool:
    """
    Render paper-style three-line plot for one sweep arm.

    Returns True when a chart was shown.
    """
    exp_dir = experiments_dir or experiments_root()
    run_ids = run_ids_for_experiments(experiments, experiments_dir=exp_dir)
    if len(run_ids) < 2:
        if not quiet:
            st.info(
                "Paper plot needs **≥2 completed runs** with global signal means in `results.pt`."
            )
        return False

    safe = _safe_key(key_prefix)
    try:
        payload = build_paper_benchmark_plot(run_ids, exp_dir, sweep_kind=sweep_kind)
        plot_data = payload.to_dict()
    except (ValueError, TypeError) as exc:
        if not quiet:
            st.warning(str(exc))
        return False
    except Exception as exc:
        if not quiet:
            st.warning(f"Could not build paper plot: {exc}")
        return False

    st.markdown(f"**Paper plot — {plot_data['experiment']}**")
    st.caption(
        f"X: **Percentage (0–1)** · Y: **accuracy + global {plot_data['aleatoric_signal']} / "
        f"{plot_data['epistemic_signal']} means** (all eval samples, not pool-filtered). "
        + _format_correlation_caption(plot_data.get("correlations") or {})
    )

    fig = figure_from_payload(plot_data)
    st.plotly_chart(fig, use_container_width=True, key=f"{safe}_paper_chart")
    render_plot_png_download(
        fig,
        plot_data,
        key_prefix=f"{safe}_paper_export",
        filename_prefix="paper_benchmark_plot",
    )

    corr = plot_data.get("correlations") or {}
    secondary = corr.get("secondary_metric")
    secondary_val = corr.get("secondary_correlation")
    tail = ""
    if secondary is not None and secondary_val is not None:
        tail = f" · secondary ρ({secondary}, scores) = {secondary_val:.3f}"
    st.caption(
        f"{plot_data['points']} point(s) · runs: `{', '.join(plot_data.get('run_ids') or [])}`{tail}"
    )
    return True
