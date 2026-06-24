"""
Streamlit visualizations for ``per_sample_signals.csv`` (one fast-pilot run).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from uqlab.shared.types import (
    ALEATORIC_SIGNALS,
    EPISTEMIC_SIGNALS,
    LEGACY_SIGNAL_NAMES,
    SIGNAL_LABELS,
    SIGNAL_NAMES,
    signal_choices,
)

# Metadata + canonical signal columns (matches ``save_per_sample_csv`` header).
PER_SAMPLE_META_COLUMNS = (
    "group",
    "dataset_index",
    "clean_label",
    "noisy_label",
    "is_noisy",
)
PER_SAMPLE_SIGNAL_COLUMNS = tuple(SIGNAL_NAMES)
GROUP_ORDER = ("clean", "aleatoric_like", "epistemic_like")


def _present_signal_columns(df: pd.DataFrame, extra: Iterable[str] = ()) -> list[str]:
    wanted = (
        list(PER_SAMPLE_SIGNAL_COLUMNS)
        + [c for c in LEGACY_SIGNAL_NAMES if c not in PER_SAMPLE_SIGNAL_COLUMNS]
        + list(extra)
    )
    return [c for c in wanted if c in df.columns]


def render_per_sample_signal_visualizations(df: pd.DataFrame) -> None:
    """
    Explore all per-sample uncertainty signals for one run folder.

    Expects columns from ``evaluation.save_per_sample_csv`` (group, dataset_index,
    clean/noisy labels, is_noisy, plus every key in ``signal_table``).
    """
    if df.empty:
        st.warning("Per-sample table is empty.")
        return

    signal_cols = _present_signal_columns(df)
    if not signal_cols:
        st.warning(
            "No signal columns found. Expected at least one of: "
            + ", ".join(PER_SAMPLE_SIGNAL_COLUMNS)
        )
        return

    plot_df = df.copy()
    if "group" in plot_df.columns:
        groups = [g for g in GROUP_ORDER if g in plot_df["group"].unique()]
        extras = [g for g in plot_df["group"].unique() if g not in groups]
        plot_df["group"] = pd.Categorical(
            plot_df["group"],
            categories=groups + sorted(extras),
            ordered=True,
        )

    st.markdown("### Per-sample signals")
    st.caption(
        "One row per evaluated sample: **group** = eval pack (clean / aleatoric_like / "
        "epistemic_like). Signal columns match ``build_experiment_signal_table``."
    )

    tab_table, tab_summary, tab_dist, tab_corr, tab_ude = st.tabs(
        ["Table", "Summary by group", "Distributions", "Correlation", "UDE Analysis"]
    )

    with tab_table:
        show_cols = [c for c in PER_SAMPLE_META_COLUMNS if c in plot_df.columns] + signal_cols
        table_df = plot_df
        if "is_noisy" in plot_df.columns:
            only_noisy = st.checkbox("Show only is_noisy rows", value=False, key="per_sample_only_noisy")
            if only_noisy:
                table_df = plot_df[plot_df["is_noisy"].astype(bool)]
        st.dataframe(
            table_df[show_cols],
            use_container_width=True,
            hide_index=True,
        )

    with tab_summary:
        if "group" not in plot_df.columns:
            st.info("No `group` column — showing global stats only.")
            st.dataframe(plot_df[signal_cols].describe().T, use_container_width=True)
        else:
            summary = (
                plot_df.groupby("group", observed=True)[signal_cols]
                .agg(["mean", "std", "min", "max"])
                .round(6)
            )
            st.dataframe(summary, use_container_width=True)

    with tab_dist:
        default_signals = signal_cols[: min(4, len(signal_cols))]
        selected = st.multiselect(
            "Signals to plot",
            signal_cols,
            default=default_signals,
            key="per_sample_viz_signals",
        )
        if not selected:
            st.info("Select at least one signal.")
        elif "group" not in plot_df.columns:
            for col in selected:
                fig = px.histogram(plot_df, x=col, nbins=40, title=col)
                st.plotly_chart(fig, use_container_width=True)
        else:
            n = len(selected)
            cols = 2
            rows = (n + cols - 1) // cols
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=selected,
                vertical_spacing=0.12,
                horizontal_spacing=0.08,
            )
            for i, col in enumerate(selected):
                r, c = i // cols + 1, i % cols + 1
                for group in plot_df["group"].dropna().unique():
                    sub = plot_df[plot_df["group"] == group]
                    fig.add_trace(
                        go.Box(
                            y=sub[col],
                            name=str(group),
                            legendgroup=str(group),
                            showlegend=(i == 0),
                        ),
                        row=r,
                        col=c,
                    )
            fig.update_layout(
                height=max(280, 220 * rows),
                boxmode="group",
                margin=dict(t=48, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        if "is_noisy" in plot_df.columns and selected:
            with st.expander("Split by is_noisy", expanded=False):
                for col in selected:
                    fig = px.box(
                        plot_df,
                        x="is_noisy",
                        y=col,
                        color="group" if "group" in plot_df.columns else None,
                        points="outliers",
                        title=f"{col} vs is_noisy",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        if len(signal_cols) < 2:
            st.info("Need at least two signal columns for a correlation matrix.")
        else:
            corr = plot_df[signal_cols].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                title="Signal correlation (all samples)",
            )
            st.plotly_chart(fig, use_container_width=True)
            if "group" in plot_df.columns:
                group_choice = st.selectbox(
                    "Correlation for one group",
                    options=list(plot_df["group"].dropna().unique()),
                    key="per_sample_corr_group",
                )
                sub = plot_df[plot_df["group"] == group_choice]
                if len(sub) >= 2:
                    corr_g = sub[signal_cols].corr(numeric_only=True)
                    fig_g = px.imshow(
                        corr_g,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu",
                        zmin=-1,
                        zmax=1,
                        title=f"Signal correlation — {group_choice}",
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

    with tab_ude:
        st.markdown("#### 🔬 Uncertainty Disentanglement Error (UDE) Analysis")
        st.caption(
            "UDE measures how well signals distinguish epistemic from aleatoric uncertainty. "
            "Lower scores are better (perfect = 0)."
        )
        
        # Try to load UDE scores
        ude_path = Path(__file__).resolve().parent.parent.parent.parent / "results" / "validation" / "ude_scores.json"
        
        if not ude_path.exists():
            st.warning(
                f"⚠️ UDE scores not found. Run the analysis script first:\n\n"
                f"```bash\ncd uqlab-streamlit && source .venv/bin/activate && "
                f"python scripts/calculate_ude_scores.py\n```"
            )
            st.info(
                "💡 The UDE analysis calculates Pearson correlations between signals and swept parameters "
                "to validate uncertainty disentanglement according to 4 conditions:\n\n"
                "- **C1**: Aleatoric signal should correlate with noise level (ρ > 0.7)\n"
                "- **C2**: Epistemic signal should correlate with dataset size (ρ > 0.7)\n"
                "- **O1**: Aleatoric signal should be independent of dataset size (|ρ| < 0.3)\n"
                "- **O2**: Epistemic signal should be independent of noise level (|ρ| < 0.3)"
            )
        else:
            try:
                with open(ude_path, 'r') as f:
                    ude_results = json.load(f)
                
                # Separate by signal type
                epistemic_results = [r for r in ude_results if r.get("type") == "epistemic"]
                aleatoric_results = [r for r in ude_results if r.get("type") == "aleatoric"]
                other_results = [r for r in ude_results if r.get("type") == "other"]
                
                # Display epistemic signals
                if epistemic_results:
                    st.markdown("##### 📊 Epistemic Signals")
                    st.caption("Should respond to dataset size, stable to noise")
                    
                    for result in epistemic_results:
                        signal = result["signal"]
                        ude_score = result.get("ude_score")
                        c2 = result.get("c2")
                        o2 = result.get("o2")
                        
                        if ude_score is not None:
                            # Color-coded badge
                            if ude_score < 0.1:
                                badge_color = "green"
                                status = "✅ Excellent"
                            elif ude_score < 0.3:
                                badge_color = "orange"
                                status = "⚠️ Good"
                            else:
                                badge_color = "red"
                                status = "❌ Poor"
                            
                            with st.expander(f"**{signal}** — UDE: {ude_score:.3f} {status}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**C2: Correlation with dataset size**")
                                    if c2:
                                        corr = c2["correlation"]
                                        p_val = c2["p_value"]
                                        st.metric(
                                            "Pearson ρ",
                                            f"{corr:+.3f}",
                                            delta="✓ Pass" if corr > 0.7 else "✗ Fail",
                                            delta_color="normal" if corr > 0.7 else "inverse"
                                        )
                                        st.caption(f"p-value: {p_val:.4f}, n={c2['n']}")
                                
                                with col2:
                                    st.markdown("**O2: Independence from noise**")
                                    if o2:
                                        corr = o2["correlation"]
                                        p_val = o2["p_value"]
                                        st.metric(
                                            "Pearson ρ",
                                            f"{corr:+.3f}",
                                            delta="✓ Pass" if abs(corr) < 0.3 else "✗ Fail",
                                            delta_color="normal" if abs(corr) < 0.3 else "inverse"
                                        )
                                        st.caption(f"p-value: {p_val:.4f}, n={o2['n']}")
                
                # Display aleatoric signals
                if aleatoric_results:
                    st.markdown("##### 📊 Aleatoric Signals")
                    st.caption("Should respond to noise, stable to dataset size")
                    
                    for result in aleatoric_results:
                        signal = result["signal"]
                        ude_score = result.get("ude_score")
                        c1 = result.get("c1")
                        o1 = result.get("o1")
                        
                        if ude_score is not None:
                            # Color-coded badge
                            if ude_score < 0.1:
                                badge_color = "green"
                                status = "✅ Excellent"
                            elif ude_score < 0.3:
                                badge_color = "orange"
                                status = "⚠️ Good"
                            else:
                                badge_color = "red"
                                status = "❌ Poor"
                            
                            with st.expander(f"**{signal}** — UDE: {ude_score:.3f} {status}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**C1: Correlation with noise**")
                                    if c1:
                                        corr = c1["correlation"]
                                        p_val = c1["p_value"]
                                        st.metric(
                                            "Pearson ρ",
                                            f"{corr:+.3f}",
                                            delta="✓ Pass" if corr > 0.7 else "✗ Fail",
                                            delta_color="normal" if corr > 0.7 else "inverse"
                                        )
                                        st.caption(f"p-value: {p_val:.4f}, n={c1['n']}")
                                
                                with col2:
                                    st.markdown("**O1: Independence from dataset size**")
                                    if o1:
                                        corr = o1["correlation"]
                                        p_val = o1["p_value"]
                                        st.metric(
                                            "Pearson ρ",
                                            f"{corr:+.3f}",
                                            delta="✓ Pass" if abs(corr) < 0.3 else "✗ Fail",
                                            delta_color="normal" if abs(corr) < 0.3 else "inverse"
                                        )
                                        st.caption(f"p-value: {p_val:.4f}, n={o1['n']}")
                
                # Display other signals
                if other_results:
                    st.markdown("##### 📊 Other Signals (Baseline/Hybrid)")
                    
                    for result in other_results:
                        signal = result["signal"]
                        corr_epis = result.get("corr_epistemic")
                        corr_alea = result.get("corr_aleatoric")
                        
                        with st.expander(f"**{signal}**"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Correlation with dataset size**")
                                if corr_epis:
                                    st.metric("Pearson ρ", f"{corr_epis['correlation']:+.3f}")
                                    st.caption(f"p-value: {corr_epis['p_value']:.4f}")
                            
                            with col2:
                                st.markdown("**Correlation with noise**")
                                if corr_alea:
                                    st.metric("Pearson ρ", f"{corr_alea['correlation']:+.3f}")
                                    st.caption(f"p-value: {corr_alea['p_value']:.4f}")
                
                # Add legend
                st.markdown("---")
                st.markdown("##### 📖 Legend")
                st.markdown("""
                - **C1**: Aleatoric signal should correlate with noise level (ρ > 0.7)
                - **C2**: Epistemic signal should correlate with dataset size (ρ > 0.7)
                - **O1**: Aleatoric signal should be independent of dataset size (|ρ| < 0.3)
                - **O2**: Epistemic signal should be independent of noise level (|ρ| < 0.3)
                - **UDE**: Uncertainty Disentanglement Error (lower is better, perfect = 0)
                """)
                
            except Exception as e:
                st.error(f"❌ Error loading UDE scores: {e}")
                st.exception(e)
