"""
Enhanced Experiment Details Component

Displays comprehensive experiment results including:
- Per-signal metrics breakdown (all 7 signals)
- Uncertainty type explanations
- Best performing signals
- Visual AUROC interpretation
- Training data inspection
"""

import json
import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional

# Import training data inspection
from uqlab.ui_components.results.training_data_inspection import (
    render_training_data_stats,
    parse_training_data_stats
)


def _experiment_details_enabled() -> bool:
    from uqlab.ui_components.ui_debug import ui_on

    return ui_on("results_experiment_details")


def render_experiment_details_with_metrics(
    experiment: Dict[str, Any],
    show_explanation: bool = True
) -> None:
    """
    Render detailed experiment view with comprehensive metrics breakdown.
    
    Args:
        experiment: Experiment data dictionary from API
        show_explanation: Whether to show uncertainty explanation section
    """
    if not _experiment_details_enabled():
        st.caption(f"Details hidden (UI debug): {experiment.get('name', '?')}")
        return
    # Display experiment header
    st.markdown(f"### 🔬 Experiment: {experiment['name']}")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_emoji = {
            "queued": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌"
        }.get(experiment['status'], "❓")
        st.metric("Status", f"{status_emoji} {experiment['status'].upper()}")
    
    with col2:
        progress = experiment.get('progress', 0)
        st.metric("Progress", f"{progress:.1%}")
    
    with col3:
        created = pd.to_datetime(experiment['created_at']).strftime("%Y-%m-%d %H:%M")
        st.metric("Created", created)
    
    with col4:
        if experiment.get('completed_at'):
            completed = pd.to_datetime(experiment['completed_at']).strftime("%Y-%m-%d %H:%M")
            st.metric("Completed", completed)
        else:
            st.metric("Completed", "N/A")
    
    # Show uncertainty explanation
    if show_explanation:
        with st.expander("📚 What do these metrics mean?", expanded=False):
            render_uncertainty_explanation_compact()
    
    # Parse and display metrics
    best_signals_json = experiment.get('best_signals_json')
    
    if not best_signals_json:
        if experiment['status'] == 'completed':
            st.warning("⚠️ No metrics data available. This may be an older experiment.")
        else:
            st.info("💡 Metrics will be available after experiment completes.")
        return
    
    # Parse metrics
    try:
        if isinstance(best_signals_json, str):
            best_signals = json.loads(best_signals_json)
        else:
            best_signals = best_signals_json
        
        metrics = best_signals.get('one_vs_rest_auroc', [])
        
        if not metrics:
            st.warning("⚠️ No signal metrics found in experiment data.")
            return
        
        # Display metrics
        _render_metrics_table(metrics)
        _render_best_signals(metrics)
        _render_signal_comparison_chart(metrics)
        
        # Add training data inspection section
        st.markdown("---")
        _render_training_data_section(experiment)
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse metrics data: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error displaying metrics: {str(e)}")


def _render_training_data_section(experiment: Dict[str, Any]) -> None:
    """
    Render training data inspection section for an experiment.
    
    Args:
        experiment: Experiment data dictionary
    """
    experiment_id = experiment.get('id', '')
    
    # Check if training data is available
    train_stats = parse_training_data_stats(experiment_id)
    
    if train_stats:
        # Show in an expander to keep the UI clean
        with st.expander("📊 Training Data Inspection", expanded=False):
            st.caption("View statistics about the training data used in this experiment")
            render_training_data_stats(experiment_id)
    else:
        # Show a collapsed info message
        with st.expander("📊 Training Data Inspection", expanded=False):
            st.info("Training data statistics not available for this experiment. This feature requires training data files to be saved during experiment execution.")


def _render_metrics_table(metrics: List[Dict[str, Any]]) -> None:
    """Render comprehensive metrics table with all signals."""
    
    st.markdown("### 📊 Uncertainty Signals Performance")
    st.caption("Each signal is evaluated on two tasks: detecting noisy labels (aleatoric) and detecting under-trained classes (epistemic)")
    
    # Create DataFrame
    df_data = []
    for metric in metrics:
        signal_name = metric.get('signal', 'unknown')
        aleatoric = metric.get('aleatoric_like_auroc', 0.0)
        epistemic = metric.get('epistemic_like_auroc', 0.0)
        
        df_data.append({
            'Signal': signal_name,
            'Aleatoric AUROC': aleatoric,
            'Epistemic AUROC': epistemic,
            'Average': (aleatoric + epistemic) / 2
        })
    
    df = pd.DataFrame(df_data)
    
    # Sort by average performance
    df = df.sort_values('Average', ascending=False)
    
    # Format the dataframe
    df_display = df.copy()
    df_display['Aleatoric AUROC'] = df_display['Aleatoric AUROC'].apply(lambda x: f"{x:.3f}")
    df_display['Epistemic AUROC'] = df_display['Epistemic AUROC'].apply(lambda x: f"{x:.3f}")
    df_display['Average'] = df_display['Average'].apply(lambda x: f"{x:.3f}")
    
    # Add color indicators
    def get_performance_emoji(val_str):
        """Get emoji indicator for performance level."""
        val = float(val_str)
        if val >= 0.8:
            return "🟢"  # Excellent
        elif val >= 0.7:
            return "🟡"  # Good
        elif val >= 0.6:
            return "🟠"  # Fair
        else:
            return "🔴"  # Poor
    
    df_display['Aleatoric'] = df_display['Aleatoric AUROC'].apply(get_performance_emoji) + " " + df_display['Aleatoric AUROC']
    df_display['Epistemic'] = df_display['Epistemic AUROC'].apply(get_performance_emoji) + " " + df_display['Epistemic AUROC']
    df_display['Avg'] = df_display['Average'].apply(get_performance_emoji) + " " + df_display['Average']
    
    # Display with emoji indicators
    df_final = df_display[['Signal', 'Aleatoric', 'Epistemic', 'Avg']]
    
    st.dataframe(df_final, use_container_width=True, hide_index=True)
    
    # Add legend
    st.caption("🟢 ≥0.8 Excellent | 🟡 ≥0.7 Good | 🟠 ≥0.6 Fair | 🔴 <0.6 Poor")


def _render_best_signals(metrics: List[Dict[str, Any]]) -> None:
    """Highlight best performing signals for each uncertainty type."""
    
    st.markdown("#### 🏆 Best Performing Signals")
    
    # Find best signals
    best_aleatoric = max(metrics, key=lambda x: x.get('aleatoric_like_auroc', 0))
    best_epistemic = max(metrics, key=lambda x: x.get('epistemic_like_auroc', 0))
    best_overall = max(metrics, key=lambda x: (
        x.get('aleatoric_like_auroc', 0) + x.get('epistemic_like_auroc', 0)
    ) / 2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        **🎲 Best for Aleatoric (Noisy Labels)**
        
        Signal: `{best_aleatoric['signal']}`
        
        AUROC: **{best_aleatoric['aleatoric_like_auroc']:.3f}**
        
        💡 Use this signal for data quality control and identifying mislabeled samples
        """)
    
    with col2:
        st.success(f"""
        **🧠 Best for Epistemic (Under-training)**
        
        Signal: `{best_epistemic['signal']}`
        
        AUROC: **{best_epistemic['epistemic_like_auroc']:.3f}**
        
        💡 Use this signal for active learning and identifying data gaps
        """)
    
    with col3:
        avg_score = (best_overall['aleatoric_like_auroc'] + best_overall['epistemic_like_auroc']) / 2
        st.info(f"""
        **⭐ Best Overall Balance**
        
        Signal: `{best_overall['signal']}`
        
        Average: **{avg_score:.3f}**
        
        💡 Most versatile signal for general uncertainty quantification
        """)


def _render_signal_comparison_chart(metrics: List[Dict[str, Any]]) -> None:
    """Render comparison chart showing aleatoric vs epistemic performance."""
    
    st.markdown("#### 📈 Signal Comparison: Aleatoric vs Epistemic")
    
    # Validate input data
    if not metrics or len(metrics) == 0:
        st.info("No metrics data available for visualization")
        return
    
    # Create comparison dataframe with data validation
    df_data = []
    skipped_signals = []
    
    for metric in metrics:
        signal = metric.get('signal', 'unknown')
        aleatoric = metric.get('aleatoric_like_auroc', None)
        epistemic = metric.get('epistemic_like_auroc', None)
        
        # Validate that values are valid numbers
        # Simple validation - just check if values exist and are numeric
        # No need for strict validation since dataframe handles display gracefully
        try:
            if aleatoric is not None and epistemic is not None:
                df_data.append({
                    'Signal': signal,
                    'Aleatoric (Noisy Labels)': float(aleatoric),
                    'Epistemic (Under-training)': float(epistemic)
                })
            else:
                skipped_signals.append(signal)
        except (ValueError, TypeError):
            skipped_signals.append(signal)
    
    # Check if we have any valid data
    if len(df_data) == 0:
        st.warning("⚠️ No valid metrics data available for visualization. All AUROC values are invalid (NaN, Infinity, or out of range).")
        if skipped_signals:
            st.caption(f"Skipped signals with invalid data: {', '.join(skipped_signals)}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(df_data)
    
    # Sort by average performance
    df['Average'] = (df['Aleatoric (Noisy Labels)'] + df['Epistemic (Under-training)']) / 2
    df = df.sort_values('Average', ascending=False)
    df = df.drop('Average', axis=1)
    
    # Display as table; bar chart only when Vega has a valid numeric domain
    try:
        chart_df = df.set_index('Signal')
        numeric = chart_df.apply(pd.to_numeric, errors='coerce')
        finite = numeric.replace([float('inf'), float('-inf')], pd.NA)

        # Bar chart completely removed - caused infinite rerun loops
        # Always show dataframe instead (simpler, faster, no Vega issues)
        st.dataframe(df, use_container_width=True)
        st.caption(
            "Compare values to see which signals excel at which uncertainty type."
        )
        
        # Show warning if some signals were skipped
        if skipped_signals:
            st.caption(f"⚠️ Note: {len(skipped_signals)} signal(s) skipped due to invalid data: {', '.join(skipped_signals)}")
    
    except Exception as e:
        st.error(f"❌ Could not render chart: {str(e)}")
        st.write("Debug - Valid data available:")
        st.dataframe(df, use_container_width=True)


def render_uncertainty_explanation_compact():
    """Render compact uncertainty explanation."""
    
    st.markdown("""
    ### 📚 Understanding Uncertainty Types
    
    #### 🎲 Aleatoric Uncertainty (Data Uncertainty)
    **What it is:** Uncertainty inherent in the data itself - irreducible noise
    
    **Example:** A mislabeled training sample (cat labeled as "dog"), blurry image, ambiguous case
    
    **Key point:** More data won't help - the uncertainty is IN the data
    
    **How we test:** Inject label noise, measure if model can detect mislabeled samples
    
    **Use for:** Data cleaning, quality control, identifying annotation errors
    
    ---
    
    #### 🧠 Epistemic Uncertainty (Model Uncertainty)
    **What it is:** Uncertainty due to lack of knowledge or insufficient training data
    
    **Example:** Model trained on only 50 samples per class, rare/unseen classes
    
    **Key point:** More training data WOULD help - the model just hasn't learned enough
    
    **How we test:** Train on limited data, measure if model knows what it doesn't know
    
    **Use for:** Active learning, identifying where to collect more data
    
    ---
    
    #### 📊 AUROC Scores Explained
    
    **What is AUROC?** Area Under ROC Curve - measures how well a signal separates two groups
    
    | Score | Interpretation | Meaning |
    |-------|---------------|---------|
    | 0.5 | Random | Coin flip - no discrimination |
    | 0.6-0.7 | Fair | Some useful information |
    | 0.7-0.8 | Good | Reliable signal |
    | 0.8-0.9 | Excellent | Strong discrimination |
    | 0.9-1.0 | Outstanding | Near-perfect separation |
    
    **In our context:**
    - **Aleatoric AUROC:** Can this signal detect mislabeled/noisy samples?
    - **Epistemic AUROC:** Can this signal detect under-trained classes?
    
    ---
    
    #### 💡 Practical Tips
    
    1. **High aleatoric, low epistemic** → Use for data cleaning
    2. **Low aleatoric, high epistemic** → Use for active learning
    3. **Both high** → Versatile signal, good for general UQ
    4. **Both low** → Signal may not be useful for your task
    """)


# Made with Bob