"""
2D Parameter Sweep Configuration for Batch Experiments

Enables epistemic × aleatoric grid sweeps with heatmap visualization,
similar to the watsonx_deployment_experiment.ipynb notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def render_2d_sweep_config() -> Tuple[str, str, List[float], List[float], bool]:
    """
    Render controls for 2D parameter sweep (epistemic × aleatoric).
    
    Returns:
        Tuple of (param1, param2, values1, values2, is_valid)
        Note: values are returned as List[float] for consistency
    """
    st.markdown("### 🔁 2D Parameter Sweep Configuration")
    st.info("""
    💡 **Grid Search**: Sweep two parameters simultaneously to explore their interaction.
    Recommended: Epistemic (under_train_per_class) × Aleatoric (aleatoric_noise_percentage)
    """)
    
    # Parameter 1: Epistemic
    st.markdown("#### 📊 Parameter 1: Epistemic Uncertainty")
    param1 = "under_train_per_class"
    
    col1, col2 = st.columns(2)
    with col1:
        epis_preset = st.selectbox(
            "Epistemic Preset",
            options=["Quick (3 values)", "Standard (5 values)", "Comprehensive (7 values)", "Custom"],
            help="Pre-configured epistemic sweep ranges"
        )
    
    if epis_preset == "Quick (3 values)":
        epis_values = [1.0, 101.0, 301.0]
    elif epis_preset == "Standard (5 values)":
        epis_values = [1.0, 51.0, 101.0, 201.0, 301.0]
    elif epis_preset == "Comprehensive (7 values)":
        epis_values = [1.0, 51.0, 101.0, 151.0, 201.0, 251.0, 301.0]
    else:  # Custom
        with col2:
            epis_custom = st.text_input(
                "Custom Values (comma-separated)",
                value="1,51,101,151,201,251,301",
                help="Enter epistemic values separated by commas"
            )
        try:
            epis_values = [float(int(x.strip())) for x in epis_custom.split(",") if x.strip()]
        except ValueError:
            st.error("❌ Invalid epistemic values. Please enter integers separated by commas.")
            epis_values = []
    
    if epis_values:
        st.caption(f"✅ Epistemic sweep: {epis_values} ({len(epis_values)} values)")
    
    st.markdown("---")
    
    # Parameter 2: Aleatoric
    st.markdown("#### 📊 Parameter 2: Aleatoric Uncertainty")
    param2 = "aleatoric_noise_percentage"
    
    col1, col2 = st.columns(2)
    with col1:
        alea_preset = st.selectbox(
            "Aleatoric Preset",
            options=["Quick (3 values)", "Standard (5 values)", "Comprehensive (6 values)", "Custom"],
            help="Pre-configured aleatoric sweep ranges"
        )
    
    if alea_preset == "Quick (3 values)":
        alea_values = [0.0, 25.0, 50.0]
    elif alea_preset == "Standard (5 values)":
        alea_values = [0.0, 20.0, 40.0, 60.0, 80.0]
    elif alea_preset == "Comprehensive (6 values)":
        alea_values = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    else:  # Custom
        with col2:
            alea_custom = st.text_input(
                "Custom Values (comma-separated)",
                value="0,20,40,60,80,100",
                help="Enter aleatoric noise percentages (0-100)"
            )
        try:
            alea_values = [float(x.strip()) for x in alea_custom.split(",") if x.strip()]
            # Validate range
            if any(v < 0 or v > 100 for v in alea_values):
                st.error("❌ Aleatoric values must be between 0 and 100")
                alea_values = []
        except ValueError:
            st.error("❌ Invalid aleatoric values. Please enter numbers separated by commas.")
            alea_values = []
    
    if alea_values:
        st.caption(f"✅ Aleatoric sweep: {alea_values} ({len(alea_values)} values)")
    
    st.markdown("---")
    
    # Summary
    is_valid = len(epis_values) > 0 and len(alea_values) > 0
    
    if is_valid:
        total_experiments = len(epis_values) * len(alea_values)
        st.success(f"""
        ✅ **2D Sweep Configuration Valid**
        - Epistemic: {len(epis_values)} values
        - Aleatoric: {len(alea_values)} values
        - **Total experiments: {total_experiments}**
        """)
        
        # Show preview grid
        with st.expander("📋 Preview Experiment Grid"):
            preview_df = pd.DataFrame(
                index=[f"Epis={e}" for e in epis_values],
                columns=[f"Alea={a}%" for a in alea_values],
                data="✓"
            )
            st.dataframe(preview_df, use_container_width=True)
    else:
        st.error("❌ Invalid configuration. Please configure both parameters.")
    
    return param1, param2, epis_values, alea_values, is_valid


def render_2d_heatmap(
    results: List[Dict],
    signal_name: str,
    metric_type: str,
    epis_values: List[float],
    alea_values: List[float],
    save_path: Optional[Path] = None
) -> None:
    """
    Render interactive heatmap for 2D sweep results.
    
    Args:
        results: List of experiment results with AUROC data
        signal_name: Name of the uncertainty signal
        metric_type: Either "epistemic_like_auroc" or "aleatoric_like_auroc"
        epis_values: Epistemic parameter values
        alea_values: Aleatoric parameter values
        save_path: Optional path to save the heatmap image
    """
    # Build matrix
    matrix = np.full((len(epis_values), len(alea_values)), np.nan)
    
    for result in results:
        if result.get("status") != "completed":
            continue
            
        # Find indices
        epis_val = result.get("epistemic_value")
        alea_val = result.get("aleatoric_value")
        
        if epis_val is None or alea_val is None:
            continue
            
        try:
            epis_idx = epis_values.index(epis_val)
            alea_idx = alea_values.index(alea_val)
        except ValueError:
            continue
        
        # Extract AUROC for this signal
        auroc_data = result.get("auroc_results", [])
        for auroc in auroc_data:
            if auroc.get("signal") == signal_name:
                matrix[epis_idx, alea_idx] = auroc.get(metric_type, np.nan)
                break
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"{a}%" for a in alea_values],
        y=[f"{e}" for e in epis_values],
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        text=np.round(matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="AUROC"),
        hoverongaps=False,
    ))
    
    metric_label = "Epistemic" if "epistemic" in metric_type else "Aleatoric"
    
    fig.update_layout(
        title=f"{signal_name} - {metric_label} Detection",
        xaxis_title="Aleatoric Noise (%)",
        yaxis_title="Epistemic (samples/class)",
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Save to file if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(save_path), width=1200, height=800)
        fig.write_html(str(save_path.with_suffix('.html')))


def render_2d_results_analysis(
    batch_id: str,
    results: List[Dict],
    epis_values: List[float],
    alea_values: List[float],
    output_dir: Optional[Path] = None
) -> None:
    """
    Render comprehensive analysis of 2D sweep results with heatmaps.
    
    Args:
        batch_id: Batch experiment ID
        results: List of experiment results
        epis_values: Epistemic parameter values
        alea_values: Aleatoric parameter values
        output_dir: Optional directory to save heatmap images
    """
    st.markdown(f"### 📊 2D Sweep Analysis: Batch {batch_id}")
    
    # Get available signals from first completed result
    available_signals = []
    for result in results:
        if result.get("status") == "completed" and result.get("auroc_results"):
            available_signals = [auroc["signal"] for auroc in result["auroc_results"]]
            break
    
    if not available_signals:
        st.warning("⚠️ No completed experiments with AUROC results yet.")
        return
    
    # Signal selector
    selected_signal = st.selectbox(
        "Select Uncertainty Signal",
        options=available_signals,
        help="Choose which uncertainty signal to visualize"
    ) or available_signals[0]  # Provide default
    
    # Metric type selector
    metric_type = st.radio(
        "Detection Type",
        options=["epistemic_like_auroc", "aleatoric_like_auroc"],
        format_func=lambda x: "Epistemic Detection" if "epistemic" in x else "Aleatoric Detection",
        horizontal=True
    )
    
    # Render heatmap with optional save
    save_path = None
    if output_dir:
        metric_label = "epistemic" if "epistemic" in metric_type else "aleatoric"
        save_path = output_dir / f"{selected_signal}_{metric_label}.png"
    
    render_2d_heatmap(results, selected_signal, metric_type, epis_values, alea_values, save_path)
    
    # Summary statistics
    with st.expander("📈 Summary Statistics"):
        completed = [r for r in results if r.get("status") == "completed"]
        st.metric("Completed Experiments", f"{len(completed)}/{len(results)}")
        
        if completed:
            # Extract AUROC values for selected signal
            auroc_values = []
            for result in completed:
                for auroc in result.get("auroc_results", []):
                    if auroc.get("signal") == selected_signal:
                        auroc_values.append(auroc.get(metric_type, np.nan))
                        break
            
            if auroc_values:
                auroc_values = [v for v in auroc_values if not np.isnan(v)]
                if auroc_values:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean AUROC", f"{np.mean(auroc_values):.3f}")
                    with col2:
                        st.metric("Max AUROC", f"{np.max(auroc_values):.3f}")
                    with col3:
                        st.metric("Min AUROC", f"{np.min(auroc_values):.3f}")

# Made with Bob
