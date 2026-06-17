"""
Results Visualization UI Components

This module contains UI components for displaying experiment results,
including single experiment details and batch experiment comparisons.
"""

import streamlit as st
import pandas as pd
import requests
import json
from pathlib import Path
from typing import Dict, List, Callable, Any

# Import training data inspection
from .training_data_inspection import render_training_data_stats


def render_experiment_results(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
    auto_refresh: bool
) -> bool:
    """
    Render experiment results section with auto-refresh.
    
    Args:
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
        auto_refresh: Current auto-refresh state
    
    Returns:
        Updated auto_refresh state
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_refresh = st.checkbox(
            "🔄 Enable Auto-Refresh (5s)",
            value=auto_refresh,
            key="auto_refresh_checkbox",
            help="Automatically refresh experiment status every 5 seconds"
        )
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
            return  # CRITICAL: Stop execution after rerun
    with col3:
        if st.button("🛑 Stop Refresh", use_container_width=True):
            auto_refresh = False
            st.rerun()
            return  # CRITICAL: Stop execution after rerun
    
    try:
        # Fetch experiments using no-auth endpoint
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        experiments = response.json()
        
        if not experiments:
            st.info("No experiments found. Create one using the form above!")
        else:
            # Display experiments table with start button
            exp_data = []
            for exp in experiments:
                exp_data.append({
                    "ID": exp["id"],
                    "Name": exp["name"],
                    "Status": exp["status"],
                    "Progress": f"{exp.get('progress', 0):.1%}",
                    "Created": pd.to_datetime(exp["created_at"]).strftime("%Y-%m-%d %H:%M"),
                    "Aleatoric AUROC": f"{exp['aleatoric_auroc']:.3f}" if exp.get('aleatoric_auroc') else "N/A",
                    "Epistemic AUROC": f"{exp['epistemic_auroc']:.3f}" if exp.get('epistemic_auroc') else "N/A",
                })
            
            df_exp = pd.DataFrame(exp_data)
            st.dataframe(df_exp, use_container_width=True, hide_index=True)
            
            st.caption(f"Total experiments: {len(experiments)}")
            
            # Expandable details for each experiment
            st.markdown("### 📋 Experiment Details")
            for exp in experiments:
                _render_experiment_detail(exp, api_base_url, get_headers_func)
            
            # Start training button for queued experiments
            _render_start_training_buttons(experiments, api_base_url, get_headers_func)
            
            # Enterprise-grade auto-refresh (non-blocking — no time.sleep loop)
            if auto_refresh:
                running_statuses = {"pending", "running", "queued"}
                has_running = any(exp.get("status") in running_statuses for exp in experiments)
                st.session_state._experiments_has_running = has_running

                if not has_running:
                    st.caption("✅ All experiments finished — auto-refresh off.")
                    auto_refresh = False
                else:
                    import streamlit.components.v1 as components

                    components.html(
                        '<script>setTimeout(function(){'
                        'window.parent.postMessage({type:"streamlit:rerun"},"*");'
                        "},5000);</script>",
                        height=0,
                    )
                    st.caption(
                        "🔄 Auto-refresh every 5s while runs are active. Uncheck to stop."
                    )
        
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch experiments: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Response: {e.response.text}")
    
    return auto_refresh


def _render_experiment_detail(
    exp: Dict,
    api_base_url: str,
    get_headers_func: Callable[[], Dict]
) -> None:
    """
    Render detailed view for a single experiment.
    
    Args:
        exp: Experiment data dictionary
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
    """
    # Try to load results from files
    results_data = None
    results_path = f"/tmp/uqlab_experiments/{exp['id']}/results"
    summary_file = f"{results_path}/summary.json"
    
    try:
        if Path(summary_file).exists():
            with open(summary_file) as f:
                results_data = json.load(f)
    except Exception:
        pass  # Results not available yet
    
    status_emoji = {
        "queued": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌"
    }.get(exp['status'], "❓")
    
    with st.expander(f"{status_emoji} {exp['name']} ({exp['status'].upper()})"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Info**")
            st.json({
                "ID": exp["id"],
                "Name": exp["name"],
                "Status": exp["status"],
                "Progress": f"{exp.get('progress', 0):.1%}",
                "Message": exp.get("progress_message", "N/A"),
                "Created": exp["created_at"],
                "Started": exp.get("started_at", "N/A"),
                "Completed": exp.get("completed_at", "N/A"),
            })
        
        with col2:
            st.markdown("**Configuration**")
            config = exp.get("config_yaml", {})
            st.json({
                "Noise Type": config.get("noise_type", "N/A"),
                "Epochs": config.get("epochs", "N/A"),
                "Model": config.get("dinov2_model", "N/A"),
                "Hidden Dim": config.get("hidden_dim", "N/A"),
                "Dropout": config.get("dropout", "N/A"),
            })
        
        # Show results from files if available
        if results_data:
            _render_experiment_results_data(results_data)
        elif exp['status'] == 'completed':
            st.warning("⚠️ Results files not found. Check results_path.")
            st.code(f"Expected: {summary_file}")
        
        # Training data inspection section
        st.markdown("---")
        render_training_data_stats(exp['id'])
        
        # Export to watsonx.ai button (only for completed experiments)
        if exp['status'] == 'completed':
            st.markdown("---")
            st.markdown("### 🚀 Export to watsonx.ai")
            st.info("Export this trained model for deployment to IBM watsonx.ai cloud platform.")
            
            if st.button(f"📦 Export to watsonx.ai", key=f"export_{exp['id']}", type="primary"):
                try:
                    # Import here to avoid circular dependencies
                    from uqlab.evaluation.classification.watsonx_export import export_all_for_watsonx
                    import torch
                    
                    # Load checkpoint and results
                    checkpoint_path = Path(f"{results_path}/checkpoint.pt")
                    if not checkpoint_path.exists():
                        st.error(f"❌ Checkpoint not found: {checkpoint_path}")
                    else:
                        with st.spinner("Exporting model package..."):
                            # Load checkpoint
                            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                            
                            # Load results data
                            results_file = Path(f"{results_path}/results.pt")
                            if not results_file.exists():
                                st.error(f"❌ Results file not found: {results_file}")
                            else:
                                results = torch.load(results_file, map_location='cpu', weights_only=False)
                                
                                # Export
                                output_dir = Path(f"{results_path}/watsonx_exports")
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
                                    config=exp.get('config_yaml', {}),
                                    output_base_dir=output_dir,
                                )
                                
                                st.success(f"✅ Export complete!")
                                st.markdown(f"**Export directory:** `{export_dir}`")
                                st.markdown(f"**ZIP package:** `{zip_path}`")
                                
                                # Provide download link
                                with open(zip_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Download ZIP Package",
                                        data=f.read(),
                                        file_name=zip_path.name,
                                        mime="application/zip",
                                        key=f"download_{exp['id']}"
                                    )
                                
                                st.info("📖 See `WATSONX_EXPORT_GUIDE.md` for deployment instructions.")
                                
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Delete button with confirmation
        st.markdown("---")
        st.markdown("### 🗑️ Delete Experiment")
        
        # Use session state to track confirmation
        confirm_key = f"confirm_delete_{exp['id']}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False
        
        if not st.session_state[confirm_key]:
            # Show delete button
            col_del1, col_del2 = st.columns([1, 3])
            with col_del1:
                if st.button(f"🗑️ Delete", key=f"delete_{exp['id']}", type="secondary", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()
                    return  # CRITICAL: Stop execution after rerun
            with col_del2:
                st.caption("⚠️ This will permanently delete the experiment and all its data")
        else:
            # Show confirmation
            st.warning(f"⚠️ **Are you sure you want to delete '{exp['name']}'?**\n\nThis action cannot be undone.")
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                if st.button("✅ Yes, Delete", key=f"confirm_yes_{exp['id']}", type="primary", use_container_width=True):
                    try:
                        delete_response = requests.delete(
                            f"{api_base_url}/api/v1/experiments/no-auth/{exp['id']}",
                            headers=get_headers_func(),
                            timeout=10
                        )
                        delete_response.raise_for_status()
                        st.success(f"✅ Experiment '{exp['name']}' deleted successfully!")
                        # Clear confirmation state
                        del st.session_state[confirm_key]
                        st.rerun()
                        return  # CRITICAL: Stop execution after rerun
                    except Exception as e:
                        st.error(f"Failed to delete experiment: {str(e)}")
                        st.session_state[confirm_key] = False
            with col_conf2:
                if st.button("❌ Cancel", key=f"confirm_no_{exp['id']}", use_container_width=True):
                    st.session_state[confirm_key] = False
                    st.rerun()
                    return  # CRITICAL: Stop execution after rerun


def _render_experiment_results_data(results_data: Dict) -> None:
    """
    Render experiment results data from summary file.
    
    Args:
        results_data: Results dictionary from summary.json
    """
    st.markdown("---")
    st.markdown("### 📊 Results (from files)")
    
    # Best signals
    aurocs = results_data.get("one_vs_rest_auroc", [])
    if aurocs:
        st.markdown("**🎯 Best Uncertainty Signals:**")
        
        # Find best aleatoric and epistemic signals
        best_aleatoric = max(aurocs, key=lambda x: x.get("aleatoric_like_auroc", 0))
        best_epistemic = max(aurocs, key=lambda x: x.get("epistemic_like_auroc", 0))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Best Aleatoric Signal",
                f"{best_aleatoric['aleatoric_like_auroc']:.4f}",
                delta=best_aleatoric['signal']
            )
        with col2:
            st.metric(
                "Best Epistemic Signal",
                f"{best_epistemic['epistemic_like_auroc']:.4f}",
                delta=best_epistemic['signal']
            )
        
        # All signals table
        st.markdown("**All Signals:**")
        signals_df = pd.DataFrame(aurocs)
        signals_df = signals_df.rename(columns={
            "signal": "Signal",
            "aleatoric_like_auroc": "Aleatoric AUROC",
            "epistemic_like_auroc": "Epistemic AUROC"
        })
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    
    # Macro F1 scores
    macro_f1 = results_data.get("macro_f1", [])
    if macro_f1:
        st.markdown("**🎯 3-Way Classification F1 Scores:**")
        f1_df = pd.DataFrame(macro_f1)
        f1_df = f1_df.rename(columns={
            "signal_set": "Signal Set",
            "macro_f1": "Macro F1"
        })
        st.dataframe(f1_df, use_container_width=True, hide_index=True)
    
    # Dataset info
    st.markdown("**📦 Dataset Info:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Train Size", results_data.get("train_size", "N/A"))
    with col2:
        eval_sizes = results_data.get("eval_sizes", {})
        st.metric("Clean Eval", eval_sizes.get("clean", "N/A"))
    with col3:
        st.metric("Noisy Eval", eval_sizes.get("aleatoric_like", "N/A"))


def _render_start_training_buttons(
    experiments: List[Dict],
    api_base_url: str,
    get_headers_func: Callable[[], Dict]
) -> None:
    """
    Render start training buttons for queued experiments.
    
    Args:
        experiments: List of experiment dictionaries
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
    """
    queued_exps = [e for e in experiments if e["status"] == "queued"]
    if queued_exps:
        st.markdown("### 🚀 Start Training")
        for exp in queued_exps:
            if st.button(f"Start: {exp['name']}", key=f"start_{exp['id']}"):
                try:
                    start_response = requests.post(
                        f"{api_base_url}/api/v1/experiments/no-auth/{exp['id']}/start",
                        headers=get_headers_func(),
                        timeout=30
                    )
                    start_response.raise_for_status()
                    st.success(f"✅ Training started for {exp['name']}")
                    st.rerun()
                    return  # CRITICAL: Stop execution after rerun
                except Exception as e:
                    st.error(f"Failed to start training: {str(e)}")


def _format_best_metric(summary: Dict[str, Any], key: str, metric: str) -> str:
    """
    Format a best-run metric from batch summary data.
    """
    best = summary.get(key)
    if not best or best.get(metric) is None:
        return "N/A"
    return f"{best[metric]:.3f}"

# Made with Bob
