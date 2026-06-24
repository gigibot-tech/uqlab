"""PNG export for Plotly figures shown in Streamlit (kaleido first, matplotlib fallback)."""

from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Any

import plotly.graph_objects as go
import streamlit as st

_DEFAULT_WIDTH = 1200
_DEFAULT_HEIGHT = 500
_DEFAULT_SCALE = 2


def _sanitize_filename_part(value: str, *, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", value.strip().lower())
    return cleaned[:max_len] or "plot"


def sweep_plot_filename(plot_data: dict[str, Any], *, prefix: str = "sweep_plot") -> str:
    """Stable download name from sweep line payload metadata."""
    kind = _sanitize_filename_part(str(plot_data.get("sweep_kind", "sweep")))
    signal = _sanitize_filename_part(str(plot_data.get("signal", "signal")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{kind}_{signal}_{stamp}.png"


def plotly_figure_to_png_bytes(
    fig: go.Figure,
    *,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    scale: int = _DEFAULT_SCALE,
) -> tuple[bytes | None, str | None]:
    """Render a Plotly figure to PNG bytes (requires kaleido when available)."""
    try:
        png = fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
        )
        return png, None
    except Exception as exc:
        return None, str(exc)


def matplotlib_png_from_line_payload(
    plot_data: dict[str, Any],
    *,
    dpi: int = 150,
) -> bytes:
    """Matplotlib fallback for dual-axis sweep line payloads."""
    import matplotlib.pyplot as plt

    from uqlab.evaluation.pipeline.sweep_line_plot import resolve_sweep_trace_xy

    fig, ax_left = plt.subplots(figsize=(8.5, 4.2))
    ax_right = ax_left.twinx()
    traces = plot_data.get("traces") or []

    for trace in traces:
        axis = ax_right if trace.get("yaxis") == "right" else ax_left
        linestyle = ":" if trace.get("dash") == "dot" else "-"
        xs, ys = resolve_sweep_trace_xy(traces, trace)
        if not xs:
            continue
        axis.plot(
            xs,
            ys,
            marker="o",
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            color=trace.get("color", "#333333"),
            label=trace.get("name", "series"),
        )

    ax_left.set_xlabel(plot_data.get("x_label", "X"))
    ax_left.set_ylabel(plot_data.get("y_left_title", "Signal"))
    ax_right.set_ylabel(plot_data.get("y_right_title", "Accuracy"))
    ax_right.set_ylim(0, 1)

    handles: list[Any] = []
    labels: list[str] = []
    for axis in (ax_left, ax_right):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(3, len(handles)),
            frameon=False,
        )

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def render_plot_png_download(
    fig: go.Figure,
    plot_data: dict[str, Any],
    *,
    key_prefix: str,
    filename_prefix: str = "sweep_plot",
) -> None:
    """
    Show a download button for the current figure.

    Uses Plotly/kaleido when available; otherwise offers matplotlib export.
    """
    filename = sweep_plot_filename(plot_data, prefix=filename_prefix)
    png_bytes, plotly_error = plotly_figure_to_png_bytes(fig)

    if png_bytes:
        st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=filename,
            mime="image/png",
            key=f"{key_prefix}_dl_plotly_png",
            use_container_width=True,
            help="Static PNG export via Plotly (kaleido).",
        )
        return

    st.caption(
        "Plotly PNG export unavailable"
        + (f" ({plotly_error})" if plotly_error else "")
        + " — use matplotlib export below."
    )
    mpl_bytes = matplotlib_png_from_line_payload(plot_data)
    st.download_button(
        label="Download PNG (matplotlib)",
        data=mpl_bytes,
        file_name=filename,
        mime="image/png",
        key=f"{key_prefix}_dl_mpl_png",
        use_container_width=True,
        help="Fallback raster export when kaleido is not installed.",
    )


__all__ = [
    "matplotlib_png_from_line_payload",
    "plotly_figure_to_png_bytes",
    "render_plot_png_download",
    "sweep_plot_filename",
]
