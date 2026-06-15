"""Single architecture uncertainty comparison plot - clearer view without mixing architectures."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..signals import (
    SIGNAL_LABELS,
    get_row3_signals,
    resolve_x_col,
)


def plot_single_architecture_uncertainty(
    df: pd.DataFrame,
    architecture: str,
    x_col: str | None = None,
    sweep_type: str = "dataset_size",
) -> go.Figure | None:
    """
    Create a simple 1-row × 3-column plot for a single architecture.
    
    Columns:
    1. Epistemic Uncertainty over sweep
    2. Aleatoric Uncertainty over sweep  
    3. Accuracy over sweep
    
    All with clear line styles and colors.
    """
    if df.empty:
        return None
    
    # Filter to single architecture
    arch_df = df[df['architecture'] == architecture].copy()
    if arch_df.empty:
        return None
    
    if x_col is None:
        x_col = resolve_x_col(arch_df, sweep_type)
    if x_col not in arch_df.columns:
        return None
    
    arch_df = arch_df.sort_values(x_col)
    
    # Create 1×3 subplot
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Epistemic Uncertainty",
            "Aleatoric Uncertainty",
            "Accuracy"
        ],
        horizontal_spacing=0.12,
    )
    
    epistemic_color = "#2ecc71"
    aleatoric_color = "#3498db"
    accuracy_color = "#e67e22"
    
    # Column 1: Epistemic Uncertainty
    if "mean_epistemic_uncertainty" in arch_df.columns:
        fig.add_trace(
            go.Scatter(
                x=arch_df[x_col],
                y=arch_df["mean_epistemic_uncertainty"],
                mode="lines+markers",
                name="Epistemic",
                line=dict(color=epistemic_color, width=4, dash="solid"),
                marker=dict(size=10, symbol="circle"),
                hovertemplate="<b>Epistemic</b><br>%{x}: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    
    # Column 2: Aleatoric Uncertainty
    if "mean_aleatoric_uncertainty" in arch_df.columns:
        fig.add_trace(
            go.Scatter(
                x=arch_df[x_col],
                y=arch_df["mean_aleatoric_uncertainty"],
                mode="lines+markers",
                name="Aleatoric",
                line=dict(color=aleatoric_color, width=4, dash="dash"),
                marker=dict(size=10, symbol="square"),
                hovertemplate="<b>Aleatoric</b><br>%{x}: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    
    # Column 3: Accuracy
    if "accuracy" in arch_df.columns:
        fig.add_trace(
            go.Scatter(
                x=arch_df[x_col],
                y=arch_df["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color=accuracy_color, width=4, dash="dot"),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="<b>Accuracy</b><br>%{x}: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=3,
        )
    
    x_label = (
        "Dataset Size (samples/class)"
        if sweep_type == "dataset_size"
        else "Noise Rate (%)"
    )
    
    # Update axes
    for col in range(1, 4):
        fig.update_xaxes(title_text=x_label, row=1, col=col)
    
    fig.update_yaxes(title_text="Uncertainty", range=[0, 2], row=1, col=1)
    fig.update_yaxes(title_text="Uncertainty", range=[0, 2], row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=3)
    
    fig.update_layout(
        height=400,
        title_text=f"{architecture} — {sweep_type.replace('_', ' ').title()} Sweep",
        showlegend=False,  # No legend needed - titles are clear
    )
    
    return fig


def display_single_architecture_plot(
    df: pd.DataFrame,
    architecture: str,
    x_col: str | None = None,
    sweep_type: str = "dataset_size",
) -> None:
    """Build and display the single architecture plot."""
    if df.empty:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                import streamlit as st
                st.warning(f"No data available for {architecture}")
                return
        except Exception:
            pass
        print(f"Warning: No data available for {architecture}")
        return
    
    fig = plot_single_architecture_uncertainty(df, architecture, x_col=x_col, sweep_type=sweep_type)
    if fig is not None:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                import streamlit as st
                st.plotly_chart(fig, use_container_width=True)
                return
        except Exception:
            pass
        fig.show()

# Made with Bob
