"""
Signal Visualization UI Components

This module contains UI components for per-signal AUROC visualization
in batch experiments, reading data directly from experiment output files.
"""

import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List, Callable


def render_batch_results(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
) -> None:
    """
    Render batch experiment status, charts, and aggregated results.
    """
    import requests
    
    st.subheader("📈 Batch Experiment Results")

    try:
        response = requests.get(
            f"{api_base_url}/api/v1/batch-experiments",
            headers=get_headers_func(),
            timeout=10,
        )
        response.raise_for_status()
        batches = response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch batch experiments: {str(exc)}")
        if hasattr(exc, "response") and exc.response is not None:
            st.error(f"Response: {exc.response.text}")
        return

    if not batches:
        st.info("No batch experiments found. Create one using the batch form above.")
        return

    # Filter out invalid completed batches (completed with 0 runs = database inconsistency)
    valid_batches = [
        b for b in batches
        if not (b["status"] == "completed" and b.get("completed_runs", 0) == 0)
    ]
    
    if not valid_batches:
        st.info("No valid batch experiments found. Create one using the batch form above.")
        return

    batch_df = pd.DataFrame(
        [
            {
                "ID": batch["id"],
                "Name": batch["name"],
                "Status": batch["status"],
                "Progress": f"{batch.get('progress', 0):.1%}",
                "Runs": batch.get("total_runs", 0),
                "Completed": batch.get("completed_runs", 0),
                "Failed": batch.get("failed_runs", 0),
            }
            for batch in valid_batches
        ]
    )
    st.dataframe(batch_df, use_container_width=True, hide_index=True)

    selected_batch_id = st.selectbox(
        "Inspect Batch",
        options=[batch["id"] for batch in valid_batches],
        format_func=lambda batch_id: next(
            (f"{batch['name']} ({batch['status']})" for batch in valid_batches if batch["id"] == batch_id),
            batch_id,
        ),
    )

    # Add start/retry buttons based on batch status
    selected_batch = next((b for b in valid_batches if b["id"] == selected_batch_id), None)
    if selected_batch:
        if selected_batch["status"] == "queued":
            if st.button("▶️ Start Batch Execution", type="primary", use_container_width=True):
                try:
                    start_response = requests.post(
                        f"{api_base_url}/api/v1/batch-experiments/{selected_batch_id}/start",
                        headers=get_headers_func(),
                        timeout=10,
                    )
                    start_response.raise_for_status()
                    st.success(f"✅ Batch '{selected_batch['name']}' started successfully!")
                    st.rerun()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Failed to start batch: {str(exc)}")
                    if hasattr(exc, "response") and exc.response is not None:
                        st.error(f"Response: {exc.response.text}")
        
        elif selected_batch["status"] in ["failed", "completed_with_errors"]:
            if st.button("🔄 Retry Failed Runs", type="secondary", use_container_width=True):
                try:
                    retry_response = requests.post(
                        f"{api_base_url}/api/v1/batch-experiments/{selected_batch_id}/retry",
                        headers=get_headers_func(),
                        timeout=10,
                    )
                    retry_response.raise_for_status()
                    result = retry_response.json()
                    st.success(f"✅ {result.get('message', 'Retry started')}")
                    st.rerun()
                except requests.exceptions.RequestException as exc:
                    st.error(f"Failed to retry batch: {str(exc)}")
                    if hasattr(exc, "response") and exc.response is not None:
                        st.error(f"Response: {exc.response.text}")
        
        # Delete batch experiment button with confirmation
        st.markdown("---")
        st.markdown("### 🗑️ Delete Batch Experiment")
        
        # Use session state to track confirmation
        confirm_key = f"confirm_delete_batch_{selected_batch_id}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False
        
        if not st.session_state[confirm_key]:
            # Show delete button
            col_del1, col_del2 = st.columns([1, 3])
            with col_del1:
                if st.button(f"🗑️ Delete Batch", key=f"delete_batch_{selected_batch_id}", type="secondary", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
            with col_del2:
                st.caption("⚠️ This will permanently delete the batch and all its experiment runs")
        else:
            # Show confirmation
            st.warning(f"⚠️ **Are you sure you want to delete batch '{selected_batch['name']}'?**\n\nThis will delete:\n- The batch experiment\n- All {selected_batch.get('total_runs', 0)} experiment runs\n- All associated data\n\nThis action cannot be undone.")
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                if st.button("✅ Yes, Delete Batch", key=f"confirm_yes_batch_{selected_batch_id}", type="primary", use_container_width=True):
                    try:
                        delete_response = requests.delete(
                            f"{api_base_url}/api/v1/batch-experiments/{selected_batch_id}",
                            headers=get_headers_func(),
                            timeout=10,
                        )
                        delete_response.raise_for_status()
                        st.success(f"✅ Batch experiment '{selected_batch['name']}' deleted successfully!")
                        # Clear confirmation state
                        del st.session_state[confirm_key]
                        st.rerun()
                    except requests.exceptions.RequestException as exc:
                        st.error(f"Failed to delete batch: {str(exc)}")
                        if hasattr(exc, "response") and exc.response is not None:
                            st.error(f"Response: {exc.response.text}")
                        st.session_state[confirm_key] = False
            with col_conf2:
                if st.button("❌ Cancel", key=f"confirm_no_batch_{selected_batch_id}", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()

    try:
        result_response = requests.get(
            f"{api_base_url}/api/v1/batch-experiments/{selected_batch_id}/results",
            headers=get_headers_func(),
            timeout=10,
        )
        result_response.raise_for_status()
        results = result_response.json()
        # Add batch_id to results for file-based visualization
        results["batch_id"] = selected_batch_id
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch batch results: {str(exc)}")
        if hasattr(exc, "response") and exc.response is not None:
            st.error(f"Response: {exc.response.text}")
        return

    comparison_table = results.get("comparison_table", [])
    summary = results.get("summary", {})
    
    # Debug info - show what data we received
    with st.expander("🔍 Debug Info - Data Received", expanded=False):
        st.write(f"**Series count:** {len(results.get('series', []))}")
        st.write(f"**Comparison table rows:** {len(comparison_table)}")
        st.write(f"**Status:** {results.get('status', 'unknown')}")
        if results.get('series'):
            st.write("**Series metrics:**")
            for s in results.get('series', [])[:5]:  # Show first 5
                st.write(f"  - {s.get('display_name', 'unknown')}: {len(s.get('points', []))} points")
            if len(results.get('series', [])) > 5:
                st.write(f"  ... and {len(results.get('series', [])) - 5} more")

    from ...results.results import _format_best_metric
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = results.get("status", "unknown")
        st.metric("Status", status)
    with col2:
        st.metric("Runs", len(comparison_table))
    with col3:
        # Show "In Progress" for running batches, actual metric for completed
        if status in ["running", "queued"]:
            st.metric("Best Epistemic", "In Progress...")
        else:
            st.metric("Best Epistemic", _format_best_metric(summary, "best_epistemic_run", "epistemic_auroc"))
    with col4:
        if status in ["running", "queued"]:
            st.metric("Best Aleatoric", "In Progress...")
        else:
            st.metric("Best Aleatoric", _format_best_metric(summary, "best_aleatoric_run", "aleatoric_auroc"))
    # ========== VALIDATION ANALYSIS (PHASE 2) ==========
    st.markdown("---")
    st.markdown("### 🔬 Validation Analysis")
    
    # Check if batch has validation metadata
    batch_config = selected_batch.get("base_config", {})
    validation_metadata = batch_config.get("validation_metadata", {})
    validation_enabled = validation_metadata.get("validation_enabled", False)
    
    if validation_enabled and status == "completed" and comparison_table:
        st.info("✅ This batch is configured for validation analysis")
        
        # Determine experiment type
        is_epistemic = validation_metadata.get("is_epistemic_sweep", False)
        is_aleatoric = validation_metadata.get("is_aleatoric_sweep", False)
        
        if is_epistemic:
            st.caption("📊 **Type:** Epistemic Sweep - Validates C2 (ue ∼∝ Ue) and O1 (ua ⊥ Ue)")
        elif is_aleatoric:
            st.caption("📊 **Type:** Aleatoric Sweep - Validates C1 (ua ∼∝ Ua) and O2 (ue ⊥ Ua)")
        
        # Button to run validation analysis
        if st.button("📊 Run Validation Analysis", type="primary", key=f"validate_{selected_batch_id}"):
            with st.spinner("Analyzing correlations..."):
                try:
                    # Import validation functions
                    from ..analysis.correlation_analysis import (
                        analyze_epistemic_sweep,
                        analyze_aleatoric_sweep,
                        analyze_2d_grid,
                    )
                    from ..validation.validation_visualization import render_full_validation_report
                    
                    # Prepare experiments data from comparison table
                    experiments = []
                    for row in comparison_table:
                        exp_data = {
                            'config': row.get('config', {}),
                            'epistemic_auroc': row.get('epistemic_auroc'),
                            'aleatoric_auroc': row.get('aleatoric_auroc'),
                        }
                        experiments.append(exp_data)
                    
                    # Run appropriate analysis based on experiment type
                    epistemic_param = None
                    aleatoric_param = None
                    
                    if is_epistemic:
                        epistemic_param = validation_metadata.get("epistemic_parameter", "under_train_per_class")
                        validation_result = analyze_epistemic_sweep(experiments, epistemic_param)
                    elif is_aleatoric:
                        aleatoric_param = validation_metadata.get("aleatoric_parameter", "aleatoric_noise_percentage")
                        validation_result = analyze_aleatoric_sweep(experiments, aleatoric_param)
                    else:
                        st.warning("⚠️ Unknown experiment type for validation")
                        validation_result = None
                    
                    # Display validation report with new compliance dashboard
                    if validation_result:
                        render_full_validation_report(
                            validation_result=validation_result,
                            experiments=experiments,
                            epistemic_param=epistemic_param,
                            aleatoric_param=aleatoric_param
                        )
                    
                except Exception as e:
                    st.error(f"❌ Validation analysis failed: {str(e)}")
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
    
    elif validation_enabled and status != "completed":
        st.info("⏳ Validation analysis will be available once the batch completes")
    elif not validation_enabled:
        st.info("ℹ️ This batch was not configured for validation analysis")
        st.caption("To enable validation, create a batch that sweeps epistemic or aleatoric parameters")
    else:
        st.warning("⚠️ No results available for validation analysis")
    
    st.markdown("---")


    series = results.get("series", [])
    if series:
        import plotly.graph_objects as go

        # Group series by signal type for better organization
        signal_categories = {
            "Predictive Uncertainty": ["msp_uncertainty", "predictive_entropy", "mutual_info"],
            "Attribution-Based (DualXDA)": ["inverse_coherence", "dominance"],
            "Logit-Based (Representer)": ["inverse_mass", "inverse_logit_magnitude"],
            "Aggregated": ["epistemic_auroc", "aleatoric_auroc"]
        }
        
        # Categorize series
        categorized_series = {cat: [] for cat in signal_categories.keys()}
        for item in series:
            metric = item.get("metric", "")
            display_name = item.get("display_name", metric)
            
            # Find which category this metric belongs to
            found = False
            for category, signals in signal_categories.items():
                if metric in signals or any(sig in metric for sig in signals):
                    categorized_series[category].append(item)
                    found = True
                    break
            
            # If not categorized, add to aggregated
            if not found:
                categorized_series["Aggregated"].append(item)
        
        # Create tabs for different views - put Aggregated first as default
        view_tabs = st.tabs(["📈 Aggregated (Original)", "📊 All Signals", "🎯 By Category"])
        
        # Tab 1: Aggregated only (original view) - NOW FIRST/DEFAULT
        with view_tabs[0]:
            figure_agg = go.Figure()
            aggregated_items = categorized_series.get("Aggregated", [])
            
            if not aggregated_items:
                st.warning("⚠️ No aggregated metrics available. This view shows epistemic_auroc and aleatoric_auroc when available.")
                st.info(f"📊 Found {len(series)} total series. Check other tabs for per-signal data.")
            else:
                for item in aggregated_items:
                    points = item.get("points", [])
                    if not points:
                        continue
                    figure_agg.add_trace(
                        go.Scatter(
                            x=[point["x"] for point in points],
                            y=[point["y"] for point in points],
                            mode="lines+markers",
                            name=item.get("display_name", item.get("metric", "metric")),
                            line=dict(width=3),  # Make lines thicker for visibility
                            marker=dict(size=8),  # Make markers bigger
                        )
                    )

                figure_agg.update_layout(
                    title=f"Aggregated AUROC vs {results.get('x_axis_label', 'parameter')}",
                    xaxis_title=results.get('x_axis_label', 'parameter'),
                    yaxis_title="AUROC",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Metric",
                    height=500,
                    template="plotly_white",  # Use white background for better visibility
                    legend=dict(
                        font=dict(color="black"),  # Ensure legend text is black
                        bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                        bordercolor="black",
                        borderwidth=1
                    )
                )
                st.plotly_chart(figure_agg, use_container_width=True)
                st.caption(f"✅ Showing {len(aggregated_items)} aggregated metrics")
        
        # Tab 2: All signals in one chart
        with view_tabs[1]:
            st.info(f"📊 Displaying {len(series)} total signals (including per-signal AUROC)")
            figure_all = go.Figure()
            for item in series:
                points = item.get("points", [])
                if not points:
                    continue
                figure_all.add_trace(
                    go.Scatter(
                        x=[point["x"] for point in points],
                        y=[point["y"] for point in points],
                        mode="lines+markers",
                        name=item.get("display_name", item.get("metric", "metric")),
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )

            figure_all.update_layout(
                title=f"All AUROC Signals vs {results.get('x_axis_label', 'parameter')}",
                xaxis_title=results.get("x_axis_label", "parameter"),
                yaxis_title="AUROC",
                yaxis=dict(range=[0, 1]),
                legend_title="Signal",
                height=600,
                template="plotly_white",
                legend=dict(
                    font=dict(color="black"),  # Ensure legend text is black
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                    bordercolor="black",
                    borderwidth=1
                )
            )
            st.plotly_chart(figure_all, use_container_width=True)
            st.caption(f"✅ Showing all {len(series)} signals")
        
        # Tab 3: Separate charts by category
        with view_tabs[2]:
            for category, items in categorized_series.items():
                if not items:
                    continue
                
                st.markdown(f"#### {category}")
                figure_cat = go.Figure()
                
                for item in items:
                    points = item.get("points", [])
                    if not points:
                        continue
                    figure_cat.add_trace(
                        go.Scatter(
                            x=[point["x"] for point in points],
                            y=[point["y"] for point in points],
                            mode="lines+markers",
                            name=item.get("display_name", item.get("metric", "metric")),
                            line=dict(width=2),
                            marker=dict(size=6),
                        )
                    )
                
                figure_cat.update_layout(
                    title=f"{category} - AUROC vs {results.get('x_axis_label', 'parameter')}",
                    xaxis_title=results.get("x_axis_label", "parameter"),
                    yaxis_title="AUROC",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Signal",
                    height=400,
                    template="plotly_white",
                    legend=dict(
                        font=dict(color="black"),  # Ensure legend text is black
                        bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                        bordercolor="black",
                        borderwidth=1
                    )
                )
                st.plotly_chart(figure_cat, use_container_width=True)
                st.caption(f"✅ {category}: {len(items)} signals")
    
    # ========================================================================
    # NEW SECTION: Per-Signal AUROC Visualization (Direct from Files)
    # ========================================================================
    st.markdown("---")
    st.markdown("### 📊 Per-Signal AUROC Analysis")
    st.info("💡 This section reads signal data directly from experiment output files")
    
    # Try to load per-signal data from experiment directories
    batch_id_str = results.get("batch_id")
    if not batch_id_str:
        st.warning("⚠️ Batch ID not available")
    else:
        batch_dir = f"/tmp/walaris_experiments/batch_{batch_id_str}"
        
        # Debug output
        with st.expander("🔍 Debug Information", expanded=False):
            st.code(f"Batch ID: {batch_id_str}")
            st.code(f"Batch directory: {batch_dir}")
            st.code(f"Directory exists: {os.path.exists(batch_dir)}")
            
            if os.path.exists(batch_dir):
                exp_base = Path(batch_dir) / "experiments"
                st.code(f"Experiments directory: {exp_base}")
                st.code(f"Experiments dir exists: {exp_base.exists()}")
                
                if exp_base.exists():
                    exp_dirs_list = list(exp_base.glob("exp_*"))
                    st.code(f"Found {len(exp_dirs_list)} experiment directories")
                    if exp_dirs_list:
                        st.code(f"First few: {[d.name for d in exp_dirs_list[:3]]}")
        
        if os.path.exists(batch_dir):
            _render_per_signal_visualization(batch_dir, results)
        else:
            st.warning("⚠️ Batch directory not found or not accessible")
    
    # ========================================================================
    # WATSONX EXPORT SECTION - Export batch experiments to watsonx.ai
    # ========================================================================
    st.markdown("---")
    st.markdown("### 🚀 Export to watsonx.ai")
    
    # Only show export options for completed batches
    if selected_batch and selected_batch["status"] == "completed":
        st.info("📦 Export trained models from this batch for deployment to IBM watsonx.ai cloud platform.")
        
        # Get list of completed experiments in this batch
        batch_id_str = results.get("batch_id")
        if batch_id_str:
            batch_dir = Path(f"/tmp/walaris_experiments/batch_{batch_id_str}")
            experiments_dir = batch_dir / "experiments"
            
            if experiments_dir.exists():
                exp_dirs = sorted([d for d in experiments_dir.glob("exp_*") if d.is_dir()])
                
                if exp_dirs:
                    # Create tabs for different export options
                    export_tabs = st.tabs(["📦 Export Individual Experiment", "📦 Export All Experiments"])
                    
                    # Tab 1: Export individual experiment
                    with export_tabs[0]:
                        st.markdown("#### Export Single Experiment")
                        st.caption("Select an experiment from this batch to export")
                        
                        # Create selection options with experiment info
                        exp_options = {}
                        for exp_dir in exp_dirs:
                            summary_file = exp_dir / "summary.json"
                            if summary_file.exists():
                                try:
                                    with open(summary_file, 'r') as f:
                                        summary = json.load(f)
                                    
                                    # Get best AUROC scores
                                    aurocs = summary.get("one_vs_rest_auroc", [])
                                    if aurocs:
                                        best_alea = max(aurocs, key=lambda x: x.get("aleatoric_like_auroc", 0))
                                        best_epis = max(aurocs, key=lambda x: x.get("epistemic_like_auroc", 0))
                                        
                                        label = (f"{exp_dir.name} - "
                                                f"Alea: {best_alea['aleatoric_like_auroc']:.3f}, "
                                                f"Epis: {best_epis['epistemic_like_auroc']:.3f}")
                                        exp_options[label] = exp_dir
                                except Exception:
                                    exp_options[exp_dir.name] = exp_dir
                            else:
                                exp_options[exp_dir.name] = exp_dir
                        
                        if exp_options:
                            selected_exp_label = st.selectbox(
                                "Select Experiment",
                                options=list(exp_options.keys()),
                                key="batch_export_select"
                            )
                            
                            selected_exp_dir = exp_options[selected_exp_label]
                            
                            if st.button("📦 Export Selected Experiment", type="primary", key="export_single"):
                                _export_batch_experiment(selected_exp_dir, selected_batch)
                        else:
                            st.warning("⚠️ No valid experiments found in batch")
                    
                    # Tab 2: Export all experiments
                    with export_tabs[1]:
                        st.markdown("#### Export All Experiments")
                        st.caption(f"Export all {len(exp_dirs)} experiments from this batch as separate packages")
                        
                        st.warning("⚠️ This will create multiple export packages. Make sure you have sufficient disk space.")
                        
                        if st.button("📦 Export All Experiments", type="secondary", key="export_all"):
                            _export_all_batch_experiments(exp_dirs, selected_batch, batch_id_str)
                else:
                    st.warning("⚠️ No experiment directories found in batch")
            else:
                st.warning(f"⚠️ Batch experiments directory not found: {experiments_dir}")
        else:
            st.error("❌ Batch ID not available")
    elif selected_batch and selected_batch["status"] in ["running", "queued"]:
        st.info("⏳ Export will be available once the batch completes")
    else:
        st.warning("⚠️ Export is only available for completed batches")
    
    st.markdown("---")

    if comparison_table:
        st.markdown("#### Comparison Table")
        st.dataframe(pd.DataFrame(comparison_table), use_container_width=True, hide_index=True)

    artifacts = results.get("artifacts", {})
    if artifacts:
        st.markdown("#### Artifacts")
        st.json(artifacts)


def _render_per_signal_visualization(batch_dir: str, results: Dict) -> None:
    """
    Render per-signal AUROC visualization from experiment files.
    
    Args:
        batch_dir: Path to batch experiment directory
        results: Results dictionary containing batch metadata
    """
    import plotly.graph_objects as go
    import yaml
    
    # Collect all signal data from experiments
    signal_data = {}  # {signal_name: {uncertainty_type: [(x, y), ...]}}
    x_values = []
    x_param = results.get("x_axis_label", "parameter")
    
    # Find all experiment directories
    experiments_dir = Path(batch_dir) / "experiments"
    exp_dirs = []
    
    if not experiments_dir.exists():
        st.error(f"❌ Experiments directory not found: {experiments_dir}")
        return
    
    exp_dirs = sorted([d for d in experiments_dir.glob("exp_*") if d.is_dir()])
    
    if not exp_dirs:
        st.warning(f"⚠️ No experiment directories found in {experiments_dir}")
        return
    
    st.info(f"🔍 Found {len(exp_dirs)} experiment directories")
    
    for exp_dir in exp_dirs:
        summary_file = exp_dir / "summary.json"
        if not summary_file.exists():
            st.warning(f"⚠️ No summary.json in {exp_dir.name}")
            continue
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Get x-axis value (parameter being swept)
            config_file = exp_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Extract the swept parameter value from config.data
                x_val = None
                data_config = config.get("data", {})
                for key in ["under_train_per_class", "regular_train_per_class", "eval_per_group"]:
                    if key in data_config and data_config[key] is not None:
                        x_val = data_config[key]
                        break
                
                if x_val is None:
                    st.warning(f"⚠️ Could not extract parameter value from {exp_dir.name}")
                    continue
                x_values.append(x_val)
            else:
                st.warning(f"⚠️ No config.yaml in {exp_dir.name}")
                continue
            
            # Extract per-signal AUROC
            one_vs_rest = summary.get("one_vs_rest_auroc", [])
            if not one_vs_rest:
                st.warning(f"⚠️ No one_vs_rest_auroc in {exp_dir.name}")
                continue
            
            for signal_item in one_vs_rest:
                signal_name = signal_item.get("signal", "unknown")
                alea_auroc = signal_item.get("aleatoric_like_auroc", 0)
                epis_auroc = signal_item.get("epistemic_like_auroc", 0)
                
                if signal_name not in signal_data:
                    signal_data[signal_name] = {"aleatoric": [], "epistemic": []}
                
                signal_data[signal_name]["aleatoric"].append((x_val, alea_auroc))
                signal_data[signal_name]["epistemic"].append((x_val, epis_auroc))
        
        except Exception as e:
            st.error(f"❌ Error reading {exp_dir.name}: {e}")
            import traceback
            st.code(traceback.format_exc())
            continue
    
    if not signal_data:
        st.warning("⚠️ No per-signal data found in experiment directories")
        return
    
    st.success(f"✅ Loaded {len(signal_data)} signals from {len(exp_dirs)} experiments")
    
    # Create visualization tabs - COMPLETE separation of epistemic and aleatoric
    signal_tabs = st.tabs([
        "🔴 Aleatoric - All Signals",
        "🔵 Epistemic - All Signals",
        "🔴 Aleatoric - By Signal Type",
        "🔵 Epistemic - By Signal Type",
        "📋 Data Table"
    ])
    
    # Tab 1: Aleatoric Uncertainty - All signals
    with signal_tabs[0]:
        st.markdown("#### Aleatoric Uncertainty (Label Noise)")
        st.caption("Measures uncertainty from noisy/ambiguous labels in the training data")
        
        fig_alea = go.Figure()
        
        # Add only aleatoric traces
        for signal_name, data in sorted(signal_data.items()):
            alea_points = sorted(data["aleatoric"])
            if alea_points:
                fig_alea.add_trace(go.Scatter(
                    x=[p[0] for p in alea_points],
                    y=[p[1] for p in alea_points],
                    mode="lines+markers",
                    name=signal_name,
                    line=dict(width=3),
                    marker=dict(size=8),
                ))
        
        fig_alea.update_layout(
            title=f"Aleatoric AUROC: All Signals vs {x_param}",
            xaxis_title=x_param,
            yaxis_title="AUROC",
            yaxis=dict(range=[0, 1]),
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(color="black"),  # Ensure legend text is black
                bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                bordercolor="black",
                borderwidth=1
            )
        )
        st.plotly_chart(fig_alea, use_container_width=True)
        st.caption(f"✅ Showing {len(signal_data)} signals")
    
    # Tab 2: Epistemic Uncertainty - All signals
    with signal_tabs[1]:
        st.markdown("#### Epistemic Uncertainty (Under-supported Classes)")
        st.caption("Measures uncertainty from lack of training data for certain classes")
        
        fig_epis = go.Figure()
        
        # Add only epistemic traces
        for signal_name, data in sorted(signal_data.items()):
            epis_points = sorted(data["epistemic"])
            if epis_points:
                fig_epis.add_trace(go.Scatter(
                    x=[p[0] for p in epis_points],
                    y=[p[1] for p in epis_points],
                    mode="lines+markers",
                    name=signal_name,
                    line=dict(width=3),
                    marker=dict(size=8),
                ))
        
        fig_epis.update_layout(
            title=f"Epistemic AUROC: All Signals vs {x_param}",
            xaxis_title=x_param,
            yaxis_title="AUROC",
            yaxis=dict(range=[0, 1]),
            height=600,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(color="black"),  # Ensure legend text is black
                bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                bordercolor="black",
                borderwidth=1
            )
        )
        st.plotly_chart(fig_epis, use_container_width=True)
        st.caption(f"✅ Showing {len(signal_data)} signals")
    
    # Tab 3: Aleatoric - By Signal Type (NO epistemic)
    with signal_tabs[2]:
        st.markdown("#### Aleatoric Uncertainty - By Signal Type")
        st.caption("Measures uncertainty from noisy/ambiguous labels - organized by signal category")
        
        signal_categories = {
            "Predictive Uncertainty": ["msp_uncertainty", "predictive_entropy", "mutual_info"],
            "Attribution-Based (DualXDA)": ["inverse_coherence", "dominance"],
            "Logit-Based (Representer)": ["inverse_mass", "inverse_logit_magnitude"],
        }
        
        for category, signal_names in signal_categories.items():
            category_signals = {k: v for k, v in signal_data.items() if k in signal_names}
            if not category_signals:
                continue
            
            st.markdown(f"##### {category}")
            
            fig_cat_alea = go.Figure()
            for signal_name, data in sorted(category_signals.items()):
                alea_points = sorted(data["aleatoric"])
                if alea_points:
                    fig_cat_alea.add_trace(go.Scatter(
                        x=[p[0] for p in alea_points],
                        y=[p[1] for p in alea_points],
                        mode="lines+markers",
                        name=signal_name,
                        line=dict(width=3),
                        marker=dict(size=8),
                    ))
            
            fig_cat_alea.update_layout(
                title=f"{category} - Aleatoric AUROC vs {x_param}",
                xaxis_title=x_param,
                yaxis_title="AUROC",
                yaxis=dict(range=[0, 1]),
                height=450,
                template="plotly_white",
                legend=dict(
                    font=dict(color="black"),  # Ensure legend text is black
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                    bordercolor="black",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_cat_alea, use_container_width=True)
    
    # Tab 4: Epistemic - By Signal Type (NO aleatoric)
    with signal_tabs[3]:
        st.markdown("#### Epistemic Uncertainty - By Signal Type")
        st.caption("Measures uncertainty from lack of training data - organized by signal category")
        
        signal_categories = {
            "Predictive Uncertainty": ["msp_uncertainty", "predictive_entropy", "mutual_info"],
            "Attribution-Based (DualXDA)": ["inverse_coherence", "dominance"],
            "Logit-Based (Representer)": ["inverse_mass", "inverse_logit_magnitude"],
        }
        
        for category, signal_names in signal_categories.items():
            category_signals = {k: v for k, v in signal_data.items() if k in signal_names}
            if not category_signals:
                continue
            
            st.markdown(f"##### {category}")
            
            fig_cat_epis = go.Figure()
            for signal_name, data in sorted(category_signals.items()):
                epis_points = sorted(data["epistemic"])
                if epis_points:
                    fig_cat_epis.add_trace(go.Scatter(
                        x=[p[0] for p in epis_points],
                        y=[p[1] for p in epis_points],
                        mode="lines+markers",
                        name=signal_name,
                        line=dict(width=3),
                        marker=dict(size=8),
                    ))
            
            fig_cat_epis.update_layout(
                title=f"{category} - Epistemic AUROC vs {x_param}",
                xaxis_title=x_param,
                yaxis_title="AUROC",
                yaxis=dict(range=[0, 1]),
                height=450,
                template="plotly_white",
                legend=dict(
                    font=dict(color="black"),  # Ensure legend text is black
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
                    bordercolor="black",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig_cat_epis, use_container_width=True)
    
    # Tab 5: Data table
    with signal_tabs[4]:
        st.markdown("#### Raw Signal Data")
        
        # Create a table with all data
        table_data = []
        for signal_name, data in sorted(signal_data.items()):
            for x_val in sorted(set(x_values)):
                alea_val = next((y for x, y in data["aleatoric"] if x == x_val), None)
                epis_val = next((y for x, y in data["epistemic"] if x == x_val), None)
                if alea_val is not None or epis_val is not None:
                    table_data.append({
                        x_param: x_val,
                        "Signal": signal_name,
                        "Aleatoric AUROC": f"{alea_val:.4f}" if alea_val else "N/A",
                        "Epistemic AUROC": f"{epis_val:.4f}" if epis_val else "N/A",
                    })
        
        if table_data:
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            
            # Download button
            csv = pd.DataFrame(table_data).to_csv(index=False)
            batch_name = results.get("batch_name", "batch")
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"per_signal_auroc_{batch_name}.csv",
                mime="text/csv",
            )

def _export_batch_experiment(exp_dir: Path, batch_info: Dict) -> None:
    """
    Export a single experiment from a batch to watsonx.ai format.
    
    Args:
        exp_dir: Path to experiment directory
        batch_info: Batch metadata dictionary
    """
    try:
        from uq_classification.watsonx_export import export_all_for_watsonx
        import torch
        import yaml
        
        with st.spinner(f"Exporting {exp_dir.name}..."):
            # Load checkpoint
            checkpoint_path = exp_dir / "checkpoint.pt"
            if not checkpoint_path.exists():
                st.error(f"❌ Checkpoint not found: {checkpoint_path}")
                return
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Load results
            results_file = exp_dir / "results.pt"
            if not results_file.exists():
                st.error(f"❌ Results file not found: {results_file}")
                return
            
            results = torch.load(results_file, map_location='cpu', weights_only=False)
            
            # Load config
            config_file = exp_dir / "config.yaml"
            if not config_file.exists():
                st.error(f"❌ Config file not found: {config_file}")
                return
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Export
            output_dir = exp_dir / "watsonx_exports"
            export_dir, zip_path = export_all_for_watsonx(
                model=checkpoint['model'],
                optimizer=None,  # Not needed for deployment
                epoch=checkpoint.get('epoch', 0),
                loss=checkpoint.get('loss', 0.0),
                train_embeddings=results['train_embeddings'],
                train_labels=results['train_labels'],
                train_noisy_labels=results['train_noisy_labels'],
                train_is_noisy=results['train_is_noisy'],
                train_indices=results['train_indices'],
                eval_embeddings=results['eval_embeddings'],
                eval_clean_labels=results['eval_clean_labels'],
                eval_noisy_labels=results['eval_noisy_labels'],
                eval_is_noisy=results['eval_is_noisy'],
                eval_group_labels=results['eval_group_labels'],
                eval_indices=results['eval_indices'],
                signal_table=results['signal_table'],
                predictions=results['predictions'],
                confidences=results['confidences'],
                auroc_rows=results.get('auroc_rows', []),
                config=config,
                output_base_dir=output_dir,
            )
            
            st.success(f"✅ Export complete for {exp_dir.name}!")
            st.markdown(f"**Export directory:** `{export_dir}`")
            st.markdown(f"**ZIP package:** `{zip_path}`")
            
            # Provide download link
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label=f"⬇️ Download {exp_dir.name} Package",
                    data=f.read(),
                    file_name=zip_path.name,
                    mime="application/zip",
                    key=f"download_{exp_dir.name}"
                )
            
            st.info("📖 See `WATSONX_DEPLOYMENT_GUIDE.md` for deployment instructions.")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def _export_all_batch_experiments(exp_dirs: List[Path], batch_info: Dict, batch_id: str) -> None:
    """
    Export all experiments from a batch to watsonx.ai format.
    
    Args:
        exp_dirs: List of experiment directory paths
        batch_info: Batch metadata dictionary
        batch_id: Batch identifier
    """
    try:
        from uq_classification.watsonx_export import export_all_for_watsonx
        import torch
        import yaml
        import zipfile
        from datetime import datetime
        
        st.info(f"🔄 Exporting {len(exp_dirs)} experiments...")
        
        # Create a master export directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_export_dir = Path(f"/tmp/walaris_experiments/batch_{batch_id}/watsonx_batch_export_{timestamp}")
        master_export_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        exported_count = 0
        failed_count = 0
        export_paths = []
        
        for idx, exp_dir in enumerate(exp_dirs):
            status_text.text(f"Exporting {exp_dir.name} ({idx + 1}/{len(exp_dirs)})...")
            
            try:
                # Load checkpoint
                checkpoint_path = exp_dir / "checkpoint.pt"
                if not checkpoint_path.exists():
                    st.warning(f"⚠️ Skipping {exp_dir.name}: checkpoint not found")
                    failed_count += 1
                    continue
                
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Load results
                results_file = exp_dir / "results.pt"
                if not results_file.exists():
                    st.warning(f"⚠️ Skipping {exp_dir.name}: results not found")
                    failed_count += 1
                    continue
                
                results = torch.load(results_file, map_location='cpu', weights_only=False)
                
                # Load config
                config_file = exp_dir / "config.yaml"
                if not config_file.exists():
                    st.warning(f"⚠️ Skipping {exp_dir.name}: config not found")
                    failed_count += 1
                    continue
                
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Export to master directory
                output_dir = master_export_dir / exp_dir.name
                export_dir, zip_path = export_all_for_watsonx(
                    model=checkpoint['model'],
                    optimizer=None,
                    epoch=checkpoint.get('epoch', 0),
                    loss=checkpoint.get('loss', 0.0),
                    train_embeddings=results['train_embeddings'],
                    train_labels=results['train_labels'],
                    train_noisy_labels=results['train_noisy_labels'],
                    train_is_noisy=results['train_is_noisy'],
                    train_indices=results['train_indices'],
                    eval_embeddings=results['eval_embeddings'],
                    eval_clean_labels=results['eval_clean_labels'],
                    eval_noisy_labels=results['eval_noisy_labels'],
                    eval_is_noisy=results['eval_is_noisy'],
                    eval_group_labels=results['eval_group_labels'],
                    eval_indices=results['eval_indices'],
                    signal_table=results['signal_table'],
                    predictions=results['predictions'],
                    confidences=results['confidences'],
                    auroc_rows=results.get('auroc_rows', []),
                    config=config,
                    output_base_dir=output_dir,
                )
                
                export_paths.append((exp_dir.name, zip_path))
                exported_count += 1
                
            except Exception as e:
                st.warning(f"⚠️ Failed to export {exp_dir.name}: {str(e)}")
                failed_count += 1
            
            # Update progress
            progress_bar.progress((idx + 1) / len(exp_dirs))
        
        progress_bar.empty()
        status_text.empty()
        
        # Summary
        st.success(f"✅ Batch export complete!")
        st.markdown(f"**Successfully exported:** {exported_count} experiments")
        if failed_count > 0:
            st.warning(f"**Failed:** {failed_count} experiments")
        
        st.markdown(f"**Master export directory:** `{master_export_dir}`")
        
        # Create a combined ZIP with all exports
        combined_zip_path = master_export_dir.parent / f"batch_{batch_id}_all_exports_{timestamp}.zip"
        with zipfile.ZipFile(combined_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(master_export_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(master_export_dir.parent)
                    zipf.write(file_path, arcname)
        
        st.markdown(f"**Combined ZIP:** `{combined_zip_path}`")
        
        # Provide download link for combined package
        with open(combined_zip_path, 'rb') as f:
            st.download_button(
                label=f"⬇️ Download All Exports (Combined ZIP)",
                data=f.read(),
                file_name=combined_zip_path.name,
                mime="application/zip",
                key=f"download_batch_all_{batch_id}"
            )
        
        # Show individual download links
        with st.expander("📦 Individual Export Packages", expanded=False):
            for exp_name, zip_path in export_paths:
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label=f"⬇️ {exp_name}",
                        data=f.read(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        key=f"download_individual_{exp_name}"
                    )
        
        st.info("📖 See `WATSONX_DEPLOYMENT_GUIDE.md` for deployment instructions.")
    
    except Exception as e:
        st.error(f"❌ Batch export failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# Made with Bob
