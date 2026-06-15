"""
2D Grid Heatmap Visualization for Uncertainty Quantification

Provides interactive heatmaps showing how epistemic and aleatoric parameters
jointly affect uncertainty signals (epistemic AUROC, aleatoric AUROC, etc.).

Phase 4 Enhancement: Adds validation-aware visualization with C1/C2/O1/O2 compliance overlay.
"""

import re
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime


# Canonical parameter names for the 2D grid / 1D sweeps.
EPISTEMIC_PARAM = "under_train_per_class"
ALEATORIC_PARAM = "aleatoric_noise_percentage"

# Long-form name pattern: ``..._<param_name>_<int_value>`` (current convention).
# Matched anchored to a ``_`` to avoid stealing characters from the timestamp.
_LONG_NAME_RE = {
    EPISTEMIC_PARAM: re.compile(rf"(?:^|_){EPISTEMIC_PARAM}_(\d+)"),
    ALEATORIC_PARAM: re.compile(rf"(?:^|_){ALEATORIC_PARAM}_(\d+)"),
}

# Legacy short-form pattern: ``_e200_a60`` (older batches).  We only treat ``_e``
# / ``_a`` as a value token when it is *immediately* followed by digits and
# terminated by ``_`` or end-of-string — that way ``_exp_11_...`` (which used to
# trigger false positives) is rejected.
_SHORT_NAME_RE = {
    EPISTEMIC_PARAM: re.compile(r"_e(\d+)(?=_|$)"),
    ALEATORIC_PARAM: re.compile(r"_a(\d+)(?=_|$)"),
}


def _parse_param_from_name(name: str, param: str) -> Optional[int]:
    """Pull an integer value for *param* out of an experiment name.

    Supports both the modern ``_under_train_per_class_200`` form and the legacy
    ``_e200`` / ``_a60`` shorthand.  Returns ``None`` if neither matches.
    """
    if not name:
        return None
    long_match = _LONG_NAME_RE[param].search(name)
    if long_match:
        return int(long_match.group(1))
    short_match = _SHORT_NAME_RE[param].search(name)
    if short_match:
        return int(short_match.group(1))
    return None


def _extract_params(
    exp: Dict, epistemic_param: str, aleatoric_param: str
) -> Tuple[Optional[float], Optional[float], str, str]:
    """Resolve (epis_val, alea_val, epis_source, alea_source) for one experiment.

    Prefers ``config[param]`` and falls back to parsing the experiment name with
    both the long-form (``_under_train_per_class_200``) and short-form
    (``_e200``) conventions.
    """
    config = exp.get("config") or {}
    name = exp.get("name", "") or ""

    epis_val = config.get(epistemic_param)
    alea_val = config.get(aleatoric_param)
    epis_source = "config" if epis_val is not None else "name"
    alea_source = "config" if alea_val is not None else "name"

    if epis_val is None and epistemic_param in _LONG_NAME_RE:
        epis_val = _parse_param_from_name(name, epistemic_param)
    if alea_val is None and aleatoric_param in _LONG_NAME_RE:
        alea_val = _parse_param_from_name(name, aleatoric_param)
    return epis_val, alea_val, epis_source, alea_source


def create_2d_heatmap(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    signal_name: str,
    signal_key: str
) -> go.Figure:
    """
    Create 2D heatmap showing how two parameters affect a signal.
    
    Args:
        experiments: List of experiment results
        epistemic_param: Name of epistemic parameter (x-axis)
        aleatoric_param: Name of aleatoric parameter (y-axis)
        signal_name: Display name for the signal
        signal_key: Key to extract signal value from experiment
    
    Returns:
        Plotly figure object
    """
    # Extract data
    data_points = []
    debug_info = []

    for i, exp in enumerate(experiments):
        epis_val, alea_val, epis_src, alea_src = _extract_params(
            exp, epistemic_param, aleatoric_param
        )
        signal_val = exp.get(signal_key)

        # Debug: collect info about first few experiments
        if i < 3:
            debug_info.append({
                'name': exp.get('name', ''),
                'status': exp.get('status', 'unknown'),
                'epis_param': epistemic_param,
                'epis_val': epis_val,
                'epis_source': epis_src,
                'alea_param': aleatoric_param,
                'alea_val': alea_val,
                'alea_source': alea_src,
                'signal_key': signal_key,
                'signal_val': signal_val,
                'available_signals': [
                    k for k in ('epistemic_auroc', 'aleatoric_auroc', 'accuracy')
                    if exp.get(k) is not None
                ],
            })

        if all(v is not None for v in [epis_val, alea_val, signal_val]):
            data_points.append({
                'epistemic': float(epis_val),
                'aleatoric': float(alea_val),
                'signal': float(signal_val),
            })
    
    if not data_points:
        # Return empty figure with detailed debug message
        fig = go.Figure()
        debug_text = f"<b>⚠️ No data available for {signal_name} heatmap</b><br><br>"
        debug_text += f"Checked {len(experiments)} experiments<br><br>"
        debug_text += f"<b>📋 What we need:</b><br>"
        debug_text += f"  1. Epistemic parameter: config['{epistemic_param}'] OR from name (_eXXX)<br>"
        debug_text += f"  2. Aleatoric parameter: config['{aleatoric_param}'] OR from name (_aXXX)<br>"
        debug_text += f"  3. Signal value: exp['{signal_key}']<br><br>"
        debug_text += f"<b>🔍 Debug info (first 3 experiments):</b><br>"
        
        for info in debug_info:
            debug_text += f"<br>━━━━━━━━━━━━━━━━━━━━━━<br>"
            debug_text += f"<b>{info['name']}</b> ({info['status']})<br>"
            debug_text += f"<br><b>Found values:</b><br>"
            debug_text += f"  • {info['epis_param']} = {info['epis_val']} (from {info['epis_source']})<br>"
            debug_text += f"  • {info['alea_param']} = {info['alea_val']} (from {info['alea_source']})<br>"
            debug_text += f"  • {info['signal_key']} = {info['signal_val']}<br>"
            debug_text += f"<br><b>Available signals:</b> {', '.join(info['available_signals']) if info['available_signals'] else 'None'}<br>"
        
        fig.add_annotation(
            text=debug_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=10),
            align='left'
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data_points)
    
    # Get unique values for each axis
    epis_values = sorted(df['epistemic'].unique())
    alea_values = sorted(df['aleatoric'].unique())
    
    # Create pivot table for heatmap
    pivot = df.pivot_table(
        values='signal',
        index='aleatoric',
        columns='epistemic',
        aggfunc='mean'  # Average if multiple experiments with same params
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=epis_values,
        y=alea_values,
        colorscale='RdYlGn',  # Red (low) to Green (high)
        text=np.round(pivot.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(
            title=dict(text=signal_name, side='right')
        ),
        hoverongaps=False,
        hovertemplate=(
            f'{epistemic_param}: %{{x}}<br>'
            f'{aleatoric_param}: %{{y}}<br>'
            f'{signal_name}: %{{z:.3f}}<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=f'{signal_name} vs {epistemic_param} & {aleatoric_param}',
        xaxis_title=epistemic_param,
        yaxis_title=aleatoric_param,
        height=500,
        font=dict(size=12)
    )
    
    return fig


def render_2d_grid_heatmaps(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    key_prefix: str = ""
) -> None:
    """
    Render all relevant heatmaps for a 2D grid experiment.
    
    Shows:
    - Epistemic AUROC heatmap
    - Aleatoric AUROC heatmap
    - Accuracy heatmap
    - Optional: Other signals
    
    Args:
        experiments: List of experiment results
        epistemic_param: Name of epistemic parameter
        aleatoric_param: Name of aleatoric parameter
        key_prefix: Unique prefix for Streamlit keys (e.g., batch timestamp)
    """
    st.markdown("## 🔥 2D Grid Heatmap Analysis")
    st.markdown("Visualize how epistemic and aleatoric parameters jointly affect uncertainty signals.")
    
    # Check if we have enough data
    if len(experiments) < 4:
        st.warning(f"⚠️ Need at least 4 experiments for meaningful heatmap. Found {len(experiments)}.")
        return
    
    # Create tabs for different signals
    tab1, tab2, tab3 = st.tabs([
        "📊 Epistemic AUROC",
        "📊 Aleatoric AUROC",
        "🎯 Accuracy"
    ])
    
    with tab1:
        st.markdown("### Epistemic Uncertainty Signal")
        st.caption("Higher values indicate better detection of epistemic uncertainty (model uncertainty)")
        
        fig = create_2d_heatmap(
            experiments,
            epistemic_param,
            aleatoric_param,
            "Epistemic AUROC",
            "epistemic_auroc"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_heatmap_epistemic")
        
        # Interpretation
        with st.expander("📖 How to interpret this heatmap"):
            st.markdown("""
            **What to look for:**
            - **Vertical gradient**: Epistemic AUROC should change with epistemic parameter (training size)
            - **Horizontal stability**: Should remain relatively constant across aleatoric parameter (noise level)
            - **Green regions**: High AUROC = good epistemic uncertainty detection
            - **Red regions**: Low AUROC = poor epistemic uncertainty detection
            
            **Ideal pattern**: Vertical gradient (changes with training size), horizontal stability (independent of noise)
            """)
    
    with tab2:
        st.markdown("### Aleatoric Uncertainty Signal")
        st.caption("Higher values indicate better detection of aleatoric uncertainty (data uncertainty)")
        
        fig = create_2d_heatmap(
            experiments,
            epistemic_param,
            aleatoric_param,
            "Aleatoric AUROC",
            "aleatoric_auroc"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_heatmap_aleatoric")
        
        # Interpretation
        with st.expander("📖 How to interpret this heatmap"):
            st.markdown("""
            **What to look for:**
            - **Horizontal gradient**: Aleatoric AUROC should change with aleatoric parameter (noise level)
            - **Vertical stability**: Should remain relatively constant across epistemic parameter (training size)
            - **Green regions**: High AUROC = good aleatoric uncertainty detection
            - **Red regions**: Low AUROC = poor aleatoric uncertainty detection
            
            **Ideal pattern**: Horizontal gradient (changes with noise), vertical stability (independent of training size)
            """)
    
    with tab3:
        st.markdown("### Model Accuracy")
        st.caption("Overall classification accuracy on test set")
        
        fig = create_2d_heatmap(
            experiments,
            epistemic_param,
            aleatoric_param,
            "Accuracy",
            "accuracy"
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_heatmap_accuracy")
        
        # Interpretation
        with st.expander("📖 How to interpret this heatmap"):
            st.markdown("""
            **What to look for:**
            - **Vertical gradient**: Accuracy typically increases with more training data
            - **Horizontal gradient**: Accuracy typically decreases with more noise
            - **Green regions**: High accuracy
            - **Red regions**: Low accuracy
            
            **Trade-offs**: Higher accuracy doesn't always mean better uncertainty quantification!
            """)

def extract_signal_from_best_signals(exp: Dict, signal_name: str, uncertainty_type: str) -> Optional[float]:
    """
    Extract a specific signal value from best_signals_json.
    
    Args:
        exp: Experiment dict
        signal_name: Name of signal (e.g., 'inverse_mass', 'msp_uncertainty')
        uncertainty_type: 'aleatoric' or 'epistemic'
    
    Returns:
        Signal AUROC value or None
    """
    best_signals_json = exp.get('best_signals_json')
    if not best_signals_json:
        return None
    
    try:
        best_signals = json.loads(best_signals_json) if isinstance(best_signals_json, str) else best_signals_json
        one_vs_rest = best_signals.get('one_vs_rest_auroc', [])
        
        for signal_data in one_vs_rest:
            if signal_data.get('signal') == signal_name:
                key = f"{uncertainty_type}_like_auroc"
                return signal_data.get(key)
        
        return None
    except (json.JSONDecodeError, AttributeError, KeyError):
        return None


def create_multi_signal_subplot_grid(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    uncertainty_type: str = "epistemic",
    key_prefix: str = ""
) -> go.Figure:
    """
    Create 3×3 subplot grid showing all 7 uncertainty signals.
    
    Args:
        experiments: List of experiment results
        epistemic_param: Name of epistemic parameter (x-axis)
        aleatoric_param: Name of aleatoric parameter (y-axis)
        uncertainty_type: 'epistemic' or 'aleatoric' - which AUROC to show
        key_prefix: Unique prefix for keys
    
    Returns:
        Plotly figure with 3×3 subplots
    """
    from plotly.subplots import make_subplots
    
    # Define the 7 signals
    signals = [
        ('msp_uncertainty', 'MSP Uncertainty'),
        ('predictive_entropy', 'Predictive Entropy'),
        ('mutual_info', 'Mutual Info'),
        ('inverse_coherence', 'Inverse Coherence'),
        ('dominance', 'Dominance'),
        ('inverse_mass', 'Inverse Mass'),
        ('inverse_logit_magnitude', 'Inverse Logit Mag')
    ]
    
    # Create 3×3 subplot grid
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[name for _, name in signals] + ['', ''],  # 7 signals + 2 empty
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # Extract data for each signal
    for idx, (signal_key, signal_name) in enumerate(signals):
        row = (idx // 3) + 1
        col = (idx % 3) + 1
        
        data_points = []
        for exp in experiments:
            epis_val, alea_val, _, _ = _extract_params(
                exp, epistemic_param, aleatoric_param
            )
            signal_val = extract_signal_from_best_signals(exp, signal_key, uncertainty_type)

            if all(v is not None for v in [epis_val, alea_val, signal_val]):
                data_points.append({
                    'epistemic': float(epis_val),
                    'aleatoric': float(alea_val),
                    'signal': float(signal_val),
                })
        
        if data_points:
            # Create pivot table for heatmap
            df = pd.DataFrame(data_points)
            pivot = df.pivot_table(
                values='signal',
                index='aleatoric',
                columns='epistemic',
                aggfunc='mean'
            )
            
            # Add heatmap to subplot
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale='RdYlGn',
                    showscale=(idx == 0),  # Only show colorbar for first subplot
                    colorbar=dict(
                        title=dict(text="AUROC", side='right'),
                        x=1.02,
                        len=0.3,
                        y=0.85
                    ) if idx == 0 else None,
                    hovertemplate=f'{signal_name}<br>Epistemic: %{{x}}<br>Aleatoric: %{{y}}<br>AUROC: %{{z:.3f}}<extra></extra>'
                ),
                row=row, col=col
            )
        else:
            # Add empty placeholder
            fig.add_trace(
                go.Heatmap(
                    z=[[0]],
                    showscale=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"🔬 All 7 Uncertainty Signals - {uncertainty_type.capitalize()} AUROC",
            x=0.5,
            xanchor='center'
        ),
        height=900,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text=epistemic_param if i == 3 else "", row=i, col=j)
            fig.update_yaxes(title_text=aleatoric_param if j == 1 else "", row=i, col=j)
    
    return fig


def render_multi_signal_heatmaps(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    key_prefix: str = ""
) -> None:
    """
    Render multi-signal visualization with all 7 uncertainty signals.
    
    Shows 3×3 subplot grid for both epistemic and aleatoric AUROC.
    """
    st.markdown("## 🔬 Multi-Signal Analysis (All 7 Signals)")
    st.markdown("Comprehensive view of all uncertainty quantification signals in 3×3 grid layout.")
    
    # Check if experiments have best_signals_json
    has_signals = any(exp.get('best_signals_json') for exp in experiments)
    
    if not has_signals:
        st.warning("⚠️ No multi-signal data available. These experiments were run before the multi-signal update.")
        st.info("💡 Run new experiments to see all 7 signals. Old experiments only show aggregated metrics.")
        return
    
    # Create tabs for epistemic and aleatoric
    tab_epis, tab_alea = st.tabs([
        "📊 Epistemic AUROC (All Signals)",
        "📊 Aleatoric AUROC (All Signals)"
    ])
    
    with tab_epis:
        st.markdown("### All 7 Signals - Epistemic Uncertainty Detection")
        st.caption("Shows how well each signal detects epistemic uncertainty (model uncertainty from insufficient data)")
        
        fig = create_multi_signal_subplot_grid(
            experiments,
            epistemic_param,
            aleatoric_param,
            uncertainty_type="epistemic",
            key_prefix=key_prefix
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_multi_epistemic")
        
        with st.expander("📖 Signal Descriptions"):
            st.markdown("""
            **Baseline Signals:**
            - **MSP Uncertainty**: 1 - max(softmax probabilities)
            - **Predictive Entropy**: Entropy of predictive distribution
            - **Mutual Info**: Mutual information between predictions
            
            **Attribution-Based Signals (DualXDA):**
            - **Inverse Coherence**: Best aleatoric indicator (0.73 AUROC)
            - **Dominance**: Good epistemic indicator (0.76 AUROC)
            - **Inverse Mass**: **Best epistemic indicator (0.94 AUROC)**
            
            **Logit-Based Signal:**
            - **Inverse Logit Magnitude**: Baseline comparison via Representer Theorem
            """)
    
    with tab_alea:
        st.markdown("### All 7 Signals - Aleatoric Uncertainty Detection")
        st.caption("Shows how well each signal detects aleatoric uncertainty (data uncertainty from label noise)")
        
        fig = create_multi_signal_subplot_grid(
            experiments,
            epistemic_param,
            aleatoric_param,
            uncertainty_type="aleatoric",
            key_prefix=key_prefix
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_multi_aleatoric")
        
        with st.expander("📖 Best Performers"):
            st.markdown("""
            **Top Signals for Aleatoric Uncertainty:**
            1. **Inverse Coherence**: 0.73 AUROC (best)
            2. **Inverse Mass**: 0.70 AUROC
            3. **Predictive Entropy**: 0.68 AUROC
            
            **Note**: Aleatoric uncertainty is harder to detect than epistemic!
            """)



def render_signal_comparison_heatmap(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str
) -> None:
    """
    Render side-by-side comparison of epistemic and aleatoric signals.
    
    This helps visualize the orthogonality conditions (O1, O2).
    """
    st.markdown("### 🔀 Signal Comparison")
    st.caption("Side-by-side view to assess orthogonality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_epis = create_2d_heatmap(
            experiments,
            epistemic_param,
            aleatoric_param,
            "Epistemic AUROC",
            "epistemic_auroc"
        )
        fig_epis.update_layout(height=400)
        st.plotly_chart(fig_epis, use_container_width=True)
    
    with col2:
        fig_alea = create_2d_heatmap(
            experiments,
            epistemic_param,
            aleatoric_param,
            "Aleatoric AUROC",
            "aleatoric_auroc"
        )
        fig_alea.update_layout(height=400)
        st.plotly_chart(fig_alea, use_container_width=True)
    
    st.markdown("""
    **Orthogonality Check:**
    - Left heatmap should show **vertical** gradient (epistemic changes with training size)
    - Right heatmap should show **horizontal** gradient (aleatoric changes with noise)
    - If patterns are similar, signals are **not orthogonal** (contamination detected)
    """)


def render_difference_heatmap(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str
) -> None:
    """
    Render heatmap showing difference between epistemic and aleatoric signals.
    
    Helps identify regions where one type of uncertainty dominates.
    """
    st.markdown("### ⚖️ Signal Difference (Epistemic - Aleatoric)")
    st.caption("Positive values = epistemic dominates, Negative values = aleatoric dominates")
    
    data_points = []
    for exp in experiments:
        epis_val, alea_val, _, _ = _extract_params(
            exp, epistemic_param, aleatoric_param
        )
        epis_auroc = exp.get('epistemic_auroc')
        alea_auroc = exp.get('aleatoric_auroc')

        if all(v is not None for v in [epis_val, alea_val, epis_auroc, alea_auroc]):
            data_points.append({
                'epistemic': float(epis_val),
                'aleatoric': float(alea_val),
                'difference': float(epis_auroc) - float(alea_auroc),
            })
    
    if not data_points:
        st.warning("No data available for difference heatmap")
        return
    
    df = pd.DataFrame(data_points)
    epis_values = sorted(df['epistemic'].unique())
    alea_values = sorted(df['aleatoric'].unique())
    
    pivot = df.pivot_table(
        values='difference',
        index='aleatoric',
        columns='epistemic',
        aggfunc='mean'
    )
    
    # Create diverging colorscale (red-white-blue)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=epis_values,
        y=alea_values,
        colorscale='RdBu',  # Red (negative) to Blue (positive)
        zmid=0,  # Center at zero
        text=np.round(pivot.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(
            title=dict(text="Difference", side='right')
        ),
        hovertemplate=(
            f'{epistemic_param}: %{{x}}<br>'
            f'{aleatoric_param}: %{{y}}<br>'
            'Difference: %{z:.3f}<br>'
            '<extra></extra>'
        )
    ))
    
    fig.update_layout(
        title=f'Epistemic AUROC - Aleatoric AUROC',
        xaxis_title=epistemic_param,
        yaxis_title=aleatoric_param,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📖 Interpretation"):
        st.markdown("""
        **Color meaning:**
        - **Blue regions**: Epistemic uncertainty dominates (epistemic AUROC > aleatoric AUROC)
        - **Red regions**: Aleatoric uncertainty dominates (aleatoric AUROC > epistemic AUROC)
        - **White regions**: Both signals are similar in strength
        
        **What to look for:**
        - Regions where one type of uncertainty is clearly stronger
        - Helps identify which uncertainty source is more important in different scenarios
        """)

# Made with Bob



def render_validation_aware_heatmap(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    validation_results: Optional[Dict] = None
) -> None:
    """
    Render 2D heatmaps with validation compliance overlay.
    
    Phase 4 Feature: Shows which regions of the parameter space pass/fail
    C1, C2, O1, O2 validation conditions.
    
    Args:
        experiments: List of experiment results
        epistemic_param: Name of epistemic parameter
        aleatoric_param: Name of aleatoric parameter
        validation_results: Optional validation results from correlation analysis
    """
    st.markdown("## 🎯 Validation-Aware Heatmap Analysis")
    st.info("""
    **Phase 4 Feature**: Heatmaps now show validation compliance!
    - ✅ Green borders = Passes validation conditions
    - ❌ Red borders = Fails validation conditions
    - 📊 Hover for detailed validation status
    """)
    
    if not validation_results:
        st.warning("⚠️ No validation results available. Run validation analysis first.")
        render_2d_grid_heatmaps(experiments, epistemic_param, aleatoric_param)
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Epistemic + Validation",
        "📊 Aleatoric + Validation",
        "🎯 Compliance Map",
        "📈 Export & Report"
    ])
    
    with tab1:
        st.markdown("### Epistemic AUROC with C2 & O1 Validation")
        render_heatmap_with_validation_overlay(
            experiments, epistemic_param, aleatoric_param,
            "Epistemic AUROC", "epistemic_auroc",
            validation_results, ["C2", "O1"]
        )
    
    with tab2:
        st.markdown("### Aleatoric AUROC with C1 & O2 Validation")
        render_heatmap_with_validation_overlay(
            experiments, epistemic_param, aleatoric_param,
            "Aleatoric AUROC", "aleatoric_auroc",
            validation_results, ["C1", "O2"]
        )
    
    with tab3:
        st.markdown("### Validation Compliance Map")
        render_compliance_heatmap(
            experiments, epistemic_param, aleatoric_param, validation_results
        )
    
    with tab4:
        st.markdown("### Export Heatmaps & Validation Report")
        render_export_options(
            experiments, epistemic_param, aleatoric_param, validation_results
        )


def render_heatmap_with_validation_overlay(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    signal_name: str,
    signal_key: str,
    validation_results: Dict,
    relevant_conditions: List[str]
) -> None:
    """
    Render heatmap with validation condition overlay.
    
    Args:
        experiments: List of experiments
        epistemic_param: Epistemic parameter name
        aleatoric_param: Aleatoric parameter name
        signal_name: Display name for signal
        signal_key: Key to extract signal value
        validation_results: Validation results dict
        relevant_conditions: List of conditions to check (e.g., ["C1", "C2"])
    """
    # Create base heatmap
    fig = create_2d_heatmap(
        experiments, epistemic_param, aleatoric_param,
        signal_name, signal_key
    )
    
    # Add validation status annotations
    compliance_status = validation_results.get('compliance', {})
    
    # Check if relevant conditions pass
    passes_validation = all(
        compliance_status.get(cond, {}).get('pass', False)
        for cond in relevant_conditions
    )
    
    # Add validation badge
    badge_color = "green" if passes_validation else "red"
    badge_text = "✅ PASS" if passes_validation else "❌ FAIL"
    
    fig.add_annotation(
        text=f"<b>{badge_text}</b>",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=14, color="white"),
        bgcolor=badge_color,
        borderpad=8,
        borderwidth=2,
        bordercolor="white"
    )
    
    # Add condition details
    condition_text = "<br>".join([
        f"{cond}: {'✅' if compliance_status.get(cond, {}).get('pass', False) else '❌'}"
        for cond in relevant_conditions
    ])
    
    fig.add_annotation(
        text=condition_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4,
        align="left"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed validation info
    with st.expander("📋 Validation Details"):
        for cond in relevant_conditions:
            cond_result = compliance_status.get(cond, {})
            status_icon = "✅" if cond_result.get('pass', False) else "❌"
            st.markdown(f"**{status_icon} {cond}**: {cond_result.get('description', 'N/A')}")
            if 'correlation' in cond_result:
                st.caption(f"Correlation: {cond_result['correlation']:.3f}")


def render_compliance_heatmap(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    validation_results: Dict
) -> None:
    """
    Render heatmap showing overall validation compliance score.
    
    Each cell shows how many validation conditions (C1, C2, O1, O2) pass.
    """
    st.caption("Shows number of validation conditions passed (0-4) for each parameter combination")
    
    # Extract compliance data
    compliance_status = validation_results.get('compliance', {})
    
    # Calculate compliance score (0-4)
    conditions = ['C1', 'C2', 'O1', 'O2']
    compliance_score = sum(
        1 for cond in conditions
        if compliance_status.get(cond, {}).get('pass', False)
    )
    
    # For now, show overall score (in future, could be per-cell if we have per-experiment validation)
    st.metric(
        "Overall Compliance Score",
        f"{compliance_score}/4",
        delta=f"{(compliance_score/4)*100:.0f}% compliant"
    )
    
    # Show which conditions pass/fail
    col1, col2, col3, col4 = st.columns(4)
    for idx, cond in enumerate(conditions):
        with [col1, col2, col3, col4][idx]:
            cond_result = compliance_status.get(cond, {})
            passes = cond_result.get('pass', False)
            st.metric(
                cond,
                "✅ PASS" if passes else "❌ FAIL",
                delta=cond_result.get('description', '')[:20]
            )
    
    # Create compliance visualization
    st.markdown("#### Validation Condition Status")
    
    compliance_df = pd.DataFrame([
        {
            'Condition': cond,
            'Status': '✅ Pass' if compliance_status.get(cond, {}).get('pass', False) else '❌ Fail',
            'Description': compliance_status.get(cond, {}).get('description', 'N/A'),
            'Correlation': compliance_status.get(cond, {}).get('correlation', 0.0)
        }
        for cond in conditions
    ])
    
    st.dataframe(compliance_df, use_container_width=True, hide_index=True)


def render_export_options(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    validation_results: Dict
) -> None:
    """
    Render export options for heatmaps and validation reports.
    
    Phase 4 Feature: Export functionality for analysis results.
    """
    st.markdown("#### 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export Heatmap Data**")
        
        # Prepare data for export
        export_data = []
        for exp in experiments:
            config = exp.get('config', {})
            export_data.append({
                'experiment_id': exp.get('id'),
                'experiment_name': exp.get('name'),
                epistemic_param: config.get(epistemic_param),
                aleatoric_param: config.get(aleatoric_param),
                'epistemic_auroc': exp.get('epistemic_auroc'),
                'aleatoric_auroc': exp.get('aleatoric_auroc'),
                'accuracy': exp.get('accuracy'),
                'status': exp.get('status')
            })
        
        df_export = pd.DataFrame(export_data)
        
        # CSV download
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv,
            file_name=f"heatmap_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Export Validation Report**")
        
        # Prepare validation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'experiment_count': len(experiments),
            'parameters': {
                'epistemic': epistemic_param,
                'aleatoric': aleatoric_param
            },
            'validation_results': validation_results,
            'summary': {
                'total_conditions': 4,
                'passed_conditions': sum(
                    1 for cond in ['C1', 'C2', 'O1', 'O2']
                    if validation_results.get('compliance', {}).get(cond, {}).get('pass', False)
                ),
                'compliance_percentage': (sum(
                    1 for cond in ['C1', 'C2', 'O1', 'O2']
                    if validation_results.get('compliance', {}).get(cond, {}).get('pass', False)
                ) / 4) * 100
            }
        }
        
        # JSON download
        json_str = json.dumps(report, indent=2)
        st.download_button(
            label="📄 Download Validation Report (JSON)",
            data=json_str,
            file_name=f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Additional export options
    st.markdown("---")
    st.markdown("#### 📊 Generate Summary Report")
    
    if st.button("📝 Generate Markdown Report", use_container_width=True):
        report_md = generate_markdown_report(
            experiments, epistemic_param, aleatoric_param, validation_results
        )
        
        st.download_button(
            label="📥 Download Markdown Report",
            data=report_md,
            file_name=f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        with st.expander("👁️ Preview Report"):
            st.markdown(report_md)


def generate_markdown_report(
    experiments: List[Dict],
    epistemic_param: str,
    aleatoric_param: str,
    validation_results: Dict
) -> str:
    """
    Generate a comprehensive markdown report of the experiment and validation results.
    
    Args:
        experiments: List of experiments
        epistemic_param: Epistemic parameter name
        aleatoric_param: Aleatoric parameter name
        validation_results: Validation results
    
    Returns:
        Markdown-formatted report string
    """
    compliance_status = validation_results.get('compliance', {})
    
    # Calculate statistics
    completed_exps = [e for e in experiments if e.get('status') == 'completed']
    avg_epistemic_auroc = np.mean([e.get('epistemic_auroc', 0) for e in completed_exps]) if completed_exps else 0
    avg_aleatoric_auroc = np.mean([e.get('aleatoric_auroc', 0) for e in completed_exps]) if completed_exps else 0
    avg_accuracy = np.mean([e.get('accuracy', 0) for e in completed_exps]) if completed_exps else 0
    
    report = f"""# Uncertainty Quantification Experiment Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Experiment Overview

- **Total Experiments**: {len(experiments)}
- **Completed**: {len(completed_exps)}
- **Epistemic Parameter**: {epistemic_param}
- **Aleatoric Parameter**: {aleatoric_param}

## Performance Metrics

| Metric | Average Value |
|--------|---------------|
| Epistemic AUROC | {avg_epistemic_auroc:.3f} |
| Aleatoric AUROC | {avg_aleatoric_auroc:.3f} |
| Accuracy | {avg_accuracy:.3f} |

## Validation Results

### Compliance Summary

"""
    
    # Add compliance status for each condition
    conditions = ['C1', 'C2', 'O1', 'O2']
    passed = sum(1 for cond in conditions if compliance_status.get(cond, {}).get('pass', False))
    
    report += f"**Overall Compliance**: {passed}/4 conditions passed ({(passed/4)*100:.0f}%)\n\n"
    
    for cond in conditions:
        cond_result = compliance_status.get(cond, {})
        status = "✅ PASS" if cond_result.get('pass', False) else "❌ FAIL"
        desc = cond_result.get('description', 'N/A')
        corr = cond_result.get('correlation', 0.0)
        
        report += f"### {cond}: {status}\n\n"
        report += f"- **Description**: {desc}\n"
        report += f"- **Correlation**: {corr:.3f}\n\n"
    
    report += """## Interpretation

### What These Results Mean

"""
    
    if passed == 4:
        report += "🎉 **Excellent!** All validation conditions passed. Your uncertainty quantification method successfully disentangles epistemic and aleatoric uncertainty.\n\n"
    elif passed >= 2:
        report += "⚠️ **Partial Success**: Some validation conditions passed. Review failed conditions to improve uncertainty disentanglement.\n\n"
    else:
        report += "❌ **Needs Improvement**: Most validation conditions failed. Consider adjusting your uncertainty quantification approach.\n\n"
    
    report += """### Next Steps

1. Review the heatmap visualizations to understand parameter interactions
2. Analyze failed validation conditions to identify improvement areas
3. Consider adjusting model architecture or training strategy
4. Run additional experiments to validate findings

---

*Generated by Uncertainty Quantification Dashboard - Phase 4*
"""
    
    return report


# Made with Bob



def render_enhanced_1d_sweep_plot(
    experiments: List[Dict],
    sweep_param: str,
    sweep_type: str,
    key_prefix: str = ""
) -> None:
    """
    Enhanced 1D line plot with multi-signal support and persistent filtering.
    
    Features:
    - Supports all 7 uncertainty signals from best_signals_json
    - Persistent signal selection via session state
    - Separate filters for epistemic and aleatoric sweeps
    - Modular design with helper functions
    
    Args:
        experiments: List of experiment results
        sweep_param: Parameter being swept
        sweep_type: "epistemic" or "aleatoric"
        key_prefix: Unique prefix for session state keys
    """
    st.markdown(f"### 📈 Enhanced 1D {sweep_type.title()} Sweep")
    st.caption(f"Multi-signal analysis: How all 7 signals change with {sweep_param}")
    
    # Define all 7 signals
    all_signals = [
        ('msp_uncertainty', 'MSP Uncertainty'),
        ('predictive_entropy', 'Predictive Entropy'),
        ('mutual_info', 'Mutual Info'),
        ('inverse_coherence', 'Inverse Coherence ⭐'),
        ('dominance', 'Dominance'),
        ('inverse_mass', 'Inverse Mass 🏆'),
        ('inverse_logit_magnitude', 'Inverse Logit Mag')
    ]
    
    # Session state key for this sweep type
    filter_key = f"signal_filter_{sweep_type}_{key_prefix}"
    
    # Initialize session state if not exists
    if filter_key not in st.session_state:
        # Default: show best performers
        if sweep_type == "epistemic":
            st.session_state[filter_key] = ['inverse_mass', 'dominance', 'inverse_logit_magnitude']
        else:  # aleatoric
            st.session_state[filter_key] = ['inverse_coherence', 'inverse_mass', 'predictive_entropy']
    
    # Sidebar filter
    with st.sidebar:
        st.markdown(f"#### 🎛️ {sweep_type.title()} Signal Filter")
        st.caption(f"Select signals to display in {sweep_type} 1D plot")
        
        selected_signals = st.multiselect(
            f"Signals ({sweep_type})",
            options=[sig[0] for sig in all_signals],
            default=st.session_state[filter_key],
            format_func=lambda x: dict(all_signals)[x],
            key=f"multiselect_{filter_key}",
            help=f"⭐ = Best aleatoric | 🏆 = Best epistemic"
        )
        
        # Update session state
        st.session_state[filter_key] = selected_signals
        
        if not selected_signals:
            st.warning("⚠️ Select at least one signal")
            return
    
    # Extract data for selected signals
    data_points = extract_1d_sweep_data(
        experiments, 
        sweep_param, 
        selected_signals,
        sweep_type
    )
    
    if not data_points:
        st.warning(f"⚠️ No data available for {sweep_type} 1D plot")
        st.info("💡 Make sure experiments have `best_signals_json` data")
        return
    
    # Create and display plot
    fig = create_1d_line_plot(
        data_points,
        sweep_param,
        selected_signals,
        sweep_type,
        all_signals
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_1d_{sweep_type}")
    
    # Show data table
    with st.expander("📊 View Data Table"):
        df = pd.DataFrame(data_points)
        if 'param' in df.columns:
            df = df.sort_values('param')
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Interpretation guide
    render_1d_interpretation_guide(sweep_type, selected_signals, all_signals)


def extract_1d_sweep_data(
    experiments: List[Dict],
    sweep_param: str,
    selected_signals: List[str],
    sweep_type: str
) -> List[Dict]:
    """
    Extract data points for 1D sweep plot from experiments.
    
    Args:
        experiments: List of experiment dicts
        sweep_param: Parameter being swept
        selected_signals: List of signal names to extract
        sweep_type: "epistemic" or "aleatoric"
    
    Returns:
        List of data point dicts with param value and signal values
    """
    data_points = []
    uncertainty_key = f"{sweep_type}_like_auroc"

    for exp in experiments:
        config = exp.get('config', {}) or {}
        name = exp.get('name', '') or ''

        param_val = config.get(sweep_param)
        if param_val is None and sweep_param in _LONG_NAME_RE:
            param_val = _parse_param_from_name(name, sweep_param)

        if param_val is None:
            continue
        
        point = {'param': float(param_val)}
        
        # Extract signal values from best_signals_json
        best_signals_json = exp.get('best_signals_json')
        if best_signals_json:
            try:
                best_signals = json.loads(best_signals_json) if isinstance(best_signals_json, str) else best_signals_json
                one_vs_rest = best_signals.get('one_vs_rest_auroc', [])
                
                for signal_name in selected_signals:
                    for signal_data in one_vs_rest:
                        if signal_data.get('signal') == signal_name:
                            signal_val = signal_data.get(uncertainty_key)
                            if signal_val is not None:
                                point[signal_name] = float(signal_val)
                            break
            except (json.JSONDecodeError, AttributeError, KeyError):
                pass
        
        # Only add point if it has at least one signal value
        if len(point) > 1:  # More than just 'param'
            data_points.append(point)
    
    return data_points


def create_1d_line_plot(
    data_points: List[Dict],
    sweep_param: str,
    selected_signals: List[str],
    sweep_type: str,
    all_signals: List[Tuple[str, str]]
) -> go.Figure:
    """
    Create Plotly line plot for 1D sweep.
    
    Args:
        data_points: List of data point dicts
        sweep_param: Parameter name
        selected_signals: Signal names to plot
        sweep_type: "epistemic" or "aleatoric"
        all_signals: List of (signal_key, signal_label) tuples
    
    Returns:
        Plotly Figure object
    """
    # Convert to DataFrame and sort
    df = pd.DataFrame(data_points)
    df = df.sort_values('param')
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for 7 signals
    colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE'   # Purple
    ]
    
    signal_dict = dict(all_signals)
    
    for idx, signal in enumerate(selected_signals):
        if signal in df.columns:
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=df['param'],
                y=df[signal],
                mode='lines+markers',
                name=signal_dict.get(signal, signal),
                line=dict(color=color, width=3),
                marker=dict(size=10, line=dict(width=2, color='white')),
                hovertemplate=f'<b>{signal_dict.get(signal, signal)}</b><br>' +
                             f'{sweep_param}: %{{x}}<br>' +
                             'AUROC: %{y:.3f}<extra></extra>'
            ))
    
    # Update layout
    param_label = sweep_param.replace('_', ' ').title()
    fig.update_layout(
        title=dict(
            text=f'{sweep_type.title()} Sweep: {len(selected_signals)} Signals vs {param_label}',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=param_label,
        yaxis_title='AUROC Score',
        yaxis=dict(range=[0, 1], gridcolor='rgba(128,128,128,0.2)'),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def render_1d_interpretation_guide(
    sweep_type: str,
    selected_signals: List[str],
    all_signals: List[Tuple[str, str]]
) -> None:
    """
    Render interpretation guide for 1D sweep plot.
    
    Args:
        sweep_type: "epistemic" or "aleatoric"
        selected_signals: Currently selected signals
        all_signals: All available signals
    """
    with st.expander("📖 How to interpret this plot"):
        signal_dict = dict(all_signals)
        
        st.markdown(f"**{sweep_type.title()} Sweep Analysis**")
        st.markdown(f"Showing {len(selected_signals)} signal(s): {', '.join([signal_dict.get(s, s) for s in selected_signals])}")
        
        if sweep_type == "epistemic":
            st.markdown("""
            **Expected Behavior (Epistemic Sweep)**:
            - 🏆 **Inverse Mass** should **increase strongly** (best epistemic signal, 0.94 AUROC)
            - **Dominance** should **increase** (good epistemic signal, 0.76 AUROC)
            - **Inverse Logit Magnitude** should **increase** (logit-based signal)
            - Other signals should show **moderate increase** or stay relatively flat
            
            **Why?** More training data → Better model → Better epistemic uncertainty detection
            
            **Orthogonality Check (O1)**: Aleatoric-focused signals (Inverse Coherence) should remain relatively constant
            """)
        else:  # aleatoric
            st.markdown("""
            **Expected Behavior (Aleatoric Sweep)**:
            - ⭐ **Inverse Coherence** should **increase** (best aleatoric signal, 0.73 AUROC)
            - **Inverse Mass** should show **moderate increase** (0.70 AUROC for aleatoric)
            - **Predictive Entropy** should **increase** (0.68 AUROC for aleatoric)
            - Epistemic-focused signals should show **less variation**
            
            **Why?** More label noise → More data uncertainty → Better aleatoric uncertainty detection
            
            **Orthogonality Check (O2)**: Epistemic-focused signals (Inverse Mass, Dominance) should remain relatively constant
            """)
        
        st.markdown("---")
        st.markdown("**Signal Performance Summary:**")
        st.markdown("""
        - 🏆 **Best Epistemic**: Inverse Mass (0.94 AUROC)
        - ⭐ **Best Aleatoric**: Inverse Coherence (0.73 AUROC)
        - 📊 **Baseline**: MSP Uncertainty, Predictive Entropy, Mutual Info
        - 🔬 **Attribution-Based**: Inverse Coherence, Dominance, Inverse Mass
        - 📐 **Logit-Based**: Inverse Logit Magnitude
        """)


def render_1d_sweep_plot(
    experiments: List[Dict],
    sweep_param: str,
    sweep_type: str,
    signal_names: List[str] = None
) -> None:
    """
    Render 1D line plot showing how signals change with swept parameter.
    
    Args:
        experiments: List of experiment results
        sweep_param: Parameter being swept (e.g., "under_train_per_class", "aleatoric_noise_percentage")
        sweep_type: Type of sweep ("epistemic" or "aleatoric")
        signal_names: List of signal names to plot (default: epistemic_auroc, aleatoric_auroc, accuracy)
    """
    import plotly.graph_objects as go
    
    if signal_names is None:
        signal_names = ['epistemic_auroc', 'aleatoric_auroc', 'accuracy']
    
    st.markdown(f"### 📈 1D {sweep_type.title()} Sweep Analysis")
    st.caption(f"How uncertainty signals change with {sweep_param}")
    
    # Extract data
    data_points = []
    for exp in experiments:
        config = exp.get('config', {})
        param_val = config.get(sweep_param)
        
        if param_val is not None:
            point = {'param': float(param_val)}
            for signal in signal_names:
                signal_val = exp.get(signal)
                if signal_val is not None:
                    point[signal] = float(signal_val)
            data_points.append(point)
    
    if not data_points:
        st.warning("No data available for 1D plot")
        return
    
    # Convert to DataFrame and sort by parameter
    df = pd.DataFrame(data_points)
    df = df.sort_values('param')
    
    # Create line plot
    fig = go.Figure()
    
    # Color mapping for signals
    colors = {
        'epistemic_auroc': '#FF6B6B',  # Red
        'aleatoric_auroc': '#4ECDC4',  # Teal
        'accuracy': '#95E1D3'  # Light green
    }
    
    signal_labels = {
        'epistemic_auroc': 'Epistemic AUROC',
        'aleatoric_auroc': 'Aleatoric AUROC',
        'accuracy': 'Accuracy'
    }
    
    for signal in signal_names:
        if signal in df.columns:
            fig.add_trace(go.Scatter(
                x=df['param'],
                y=df[signal],
                mode='lines+markers',
                name=signal_labels.get(signal, signal),
                line=dict(color=colors.get(signal, '#999'), width=3),
                marker=dict(size=10)
            ))
    
    # Update layout
    param_label = sweep_param.replace('_', ' ').title()
    fig.update_layout(
        title=f'{sweep_type.title()} Sweep: Signals vs {param_label}',
        xaxis_title=param_label,
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation guide
    with st.expander("📖 How to interpret this plot"):
        if sweep_type == "epistemic":
            st.markdown("""
            **Epistemic Sweep** (varying training data size):
            - **Epistemic AUROC** (red) should **increase** with more training data
              - More data → better model → better epistemic uncertainty detection
            - **Aleatoric AUROC** (teal) should remain **relatively constant**
              - Data noise is independent of training size (orthogonality O1)
            - **Accuracy** (green) typically increases with more data
            
            **Ideal pattern**: Red line goes up, teal line stays flat
            """)
        else:  # aleatoric
            st.markdown("""
            **Aleatoric Sweep** (varying label noise):
            - **Aleatoric AUROC** (teal) should **increase** with more noise
              - More noise → more aleatoric uncertainty to detect
            - **Epistemic AUROC** (red) should remain **relatively constant**
              - Model uncertainty is independent of data noise (orthogonality O2)
            - **Accuracy** (green) typically decreases with more noise
            
            **Ideal pattern**: Teal line goes up, red line stays flat
            """)


def detect_sweep_type(experiments: List[Dict]) -> Tuple[str, Optional[str]]:
    """
    Detect whether experiments are an epistemic sweep, aleatoric sweep, or 2D grid.

    Returns
    -------
    (sweep_type, swept_parameter)
        ``sweep_type`` is one of ``"epistemic"``, ``"aleatoric_custom"``,
        ``"aleatoric_cifar10n"``, ``"2d_grid"``, ``"single_point"``, or
        ``"unknown"``.  ``swept_parameter`` is the parameter name, or ``None``
        for ``2d_grid`` and ``unknown``.

    Notes
    -----
    The previous implementation matched the substrings ``"_e"`` and ``"_a"`` in
    experiment names, which falsely triggered for names like
    ``..._exp_11_aleatoric_noise_percentage_100`` (containing both ``_e`` from
    ``_exp_`` *and* ``_a`` from ``_aleatoric_``).  We now classify based on
    whether each parameter has **more than one distinct parsed value** across
    the batch.
    """
    if not experiments:
        return "unknown", None

    epis_values: set = set()
    alea_values: set = set()
    for exp in experiments:
        epis_val, alea_val, _, _ = _extract_params(exp, EPISTEMIC_PARAM, ALEATORIC_PARAM)
        if epis_val is not None:
            epis_values.add(epis_val)
        if alea_val is not None:
            alea_values.add(alea_val)

    epis_varies = len(epis_values) > 1
    alea_varies = len(alea_values) > 1

    if epis_varies and alea_varies:
        return "2d_grid", None
    if epis_varies:
        return "epistemic", EPISTEMIC_PARAM
    if alea_varies:
        first_exp = experiments[0]
        config = first_exp.get("config", {}) or {}
        noise_source = str(config.get("noise_source", "custom")).lower()
        if "cifar10n" in noise_source:
            return "aleatoric_cifar10n", ALEATORIC_PARAM
        return "aleatoric_custom", ALEATORIC_PARAM

    # Exactly one (or zero) distinct value per axis — not really a sweep.
    if epis_values or alea_values:
        return "single_point", None
    return "unknown", None


# Made with Bob
