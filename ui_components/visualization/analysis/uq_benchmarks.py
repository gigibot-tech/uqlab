"""
UI components for UQ Benchmarks tab in Streamlit dashboard.
Displays label noise sweep plots comparing research methods with production signals.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple


# ============================================================================
# API FETCH FUNCTIONS
# ============================================================================

def fetch_available_methods(api_base_url: str, get_headers_func) -> Optional[Dict]:
    """Fetch available UQ methods from backend."""
    try:
        response = requests.get(
            f"{api_base_url}/api/v1/uq-benchmarks/methods",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch UQ methods: {str(e)}")
        return None


def fetch_sweep_results(api_base_url: str, get_headers_func, sweep_id: str) -> Optional[Dict]:
    """Fetch sweep results from backend."""
    try:
        response = requests.get(
            f"{api_base_url}/api/v1/uq-benchmarks/sweeps/{sweep_id}",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch sweep results: {str(e)}")
        return None


def fetch_all_sweeps(api_base_url: str, get_headers_func) -> Optional[List[Dict]]:
    """Fetch all benchmark sweeps."""
    try:
        response = requests.get(
            f"{api_base_url}/api/v1/uq-benchmarks/sweeps",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch sweeps: {str(e)}")
        return None


def fetch_production_sweep_results(api_base_url: str, get_headers_func, batch_id: str) -> Optional[Dict]:
    """Fetch production system batch sweep results for comparison."""
    try:
        response = requests.get(
            f"{api_base_url}/api/v1/batch-experiments/{batch_id}",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch production sweep: {str(e)}")
        return None


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_accuracy_comparison(ax: plt.Axes, gaussian_data: Dict, it_data: Dict, prod_data: Dict = None):
    """
    Row 1: Plot accuracy comparison for all methods.
    
    Args:
        ax: Matplotlib axes
        gaussian_data: Gaussian Logits sweep data
        it_data: Information Theoretic sweep data
        prod_data: Production system sweep data (optional)
    """
    ax.plot(gaussian_data['parameter_values'], gaussian_data['accuracy'], 
            label='Gaussian Logits', color='blue', marker='o', linewidth=2)
    ax.plot(it_data['parameter_values'], it_data['accuracy'], 
            label='Information Theoretic', color='orange', marker='s', linewidth=2)
    
    if prod_data:
        ax.plot(prod_data['parameter_values'], prod_data['accuracy'], 
                label='Production System', color='purple', marker='^', linewidth=2)
    
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Noise Rate')
    ax.set_title('Accuracy Comparison')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])


def plot_uncertainty_with_accuracy(
    ax: plt.Axes,
    param_values: List[float],
    epistemic: List[float],
    aleatoric: List[float],
    accuracy: List[float],
    title: str,
    show_ylabel: bool = True
) -> plt.Axes:
    """
    Plot epistemic & aleatoric uncertainty with accuracy on secondary y-axis.
    
    Args:
        ax: Matplotlib axes
        param_values: X-axis values (noise rates)
        epistemic: Epistemic uncertainty values
        aleatoric: Aleatoric uncertainty values
        accuracy: Accuracy values
        title: Plot title
        show_ylabel: Whether to show y-axis label
        
    Returns:
        Secondary y-axis for accuracy
    """
    # Plot uncertainties on primary y-axis
    ax.plot(param_values, epistemic, label='Epistemic', color='blue', marker='o')
    ax.plot(param_values, aleatoric, label='Aleatoric', color='orange', marker='s')
    
    if show_ylabel:
        ax.set_ylabel('Uncertainty')
    ax.tick_params(axis='y', labelcolor='black')
    ax.legend(loc='upper left', fontsize=8)
    
    # Plot accuracy on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(param_values, accuracy, label='Accuracy', color='green', marker='^', linewidth=2)
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    
    # Add title
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Noise Rate')
    
    return ax2


def plot_signal_with_accuracy(
    ax: plt.Axes,
    param_values: List[float],
    signal_values: List[float],
    accuracy: List[float],
    signal_name: str,
    show_ylabel: bool = True
) -> plt.Axes:
    """
    Plot a single production signal with accuracy on secondary y-axis.
    
    Args:
        ax: Matplotlib axes
        param_values: X-axis values (noise rates)
        signal_values: Signal uncertainty values
        accuracy: Accuracy values
        signal_name: Name of the signal
        show_ylabel: Whether to show y-axis label
        
    Returns:
        Secondary y-axis for accuracy
    """
    # Determine color based on signal type
    if 'epistemic' in signal_name.lower() or 'dominance' in signal_name.lower() or 'mass' in signal_name.lower():
        color = 'blue'
    elif 'aleatoric' in signal_name.lower() or 'coherence' in signal_name.lower():
        color = 'orange'
    else:
        color = 'purple'
    
    # Plot signal on primary y-axis
    ax.plot(param_values, signal_values, label=signal_name, color=color, marker='o')
    
    if show_ylabel:
        ax.set_ylabel('Signal Value')
    ax.tick_params(axis='y', labelcolor='black')
    ax.legend(loc='upper left', fontsize=7)
    
    # Plot accuracy on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(param_values, accuracy, label='Acc', color='green', marker='^', linewidth=1.5, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    
    if not show_ylabel:
        ax2.set_yticklabels([])
    else:
        ax2.set_ylabel('Accuracy', color='green', fontsize=8)
    
    # Add title
    ax.set_title(signal_name.replace('_', ' ').title(), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Noise Rate', fontsize=8)
    
    return ax2


def render_comparison_grid(
    gaussian_data: Dict,
    it_data: Dict,
    prod_data: Dict = None
):
    """
    Render the full comparison grid:
    - Row 1: Accuracy comparison
    - Row 2: Gaussian Logits (Epistemic + Aleatoric + Accuracy)
    - Row 3: Information Theoretic (Epistemic + Aleatoric + Accuracy)
    - Row 4: Production signals (7 subplots, each signal + Accuracy)
    """
    # Determine number of rows
    n_rows = 4 if prod_data else 3
    
    # Create figure with subplots
    if prod_data:
        # Row 1: 1 plot (accuracy comparison)
        # Row 2: 1 plot (Gaussian Logits)
        # Row 3: 1 plot (IT)
        # Row 4: 7 plots (production signals)
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 7, hspace=0.4, wspace=0.4)
        
        # Row 1: Accuracy comparison (spans all columns)
        ax_acc = fig.add_subplot(gs[0, :])
        plot_accuracy_comparison(ax_acc, gaussian_data, it_data, prod_data)
        
        # Row 2: Gaussian Logits (spans all columns)
        ax_gl = fig.add_subplot(gs[1, :])
        plot_uncertainty_with_accuracy(
            ax_gl,
            gaussian_data['parameter_values'],
            gaussian_data['epistemic'],
            gaussian_data['aleatoric'],
            gaussian_data['accuracy'],
            'Gaussian Logits',
            show_ylabel=True
        )
        
        # Row 3: Information Theoretic (spans all columns)
        ax_it = fig.add_subplot(gs[2, :])
        plot_uncertainty_with_accuracy(
            ax_it,
            it_data['parameter_values'],
            it_data['epistemic'],
            it_data['aleatoric'],
            it_data['accuracy'],
            'Information Theoretic',
            show_ylabel=True
        )
        
        # Row 4: Production signals (7 subplots)
        signal_names = [
            'msp_uncertainty',
            'predictive_entropy',
            'mutual_info',
            'inverse_coherence',
            'dominance',
            'inverse_mass',
            'inverse_logit_magnitude'
        ]
        
        for idx, signal_name in enumerate(signal_names):
            ax_sig = fig.add_subplot(gs[3, idx])
            plot_signal_with_accuracy(
                ax_sig,
                prod_data['parameter_values'],
                prod_data['signals'][signal_name],
                prod_data['accuracy'],
                signal_name,
                show_ylabel=(idx == 0)  # Only first plot shows y-label
            )
    else:
        # Only research methods (3 rows, 1 column each)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Row 1: Accuracy comparison
        plot_accuracy_comparison(axes[0], gaussian_data, it_data)
        
        # Row 2: Gaussian Logits
        plot_uncertainty_with_accuracy(
            axes[1],
            gaussian_data['parameter_values'],
            gaussian_data['epistemic'],
            gaussian_data['aleatoric'],
            gaussian_data['accuracy'],
            'Gaussian Logits'
        )
        
        # Row 3: Information Theoretic
        plot_uncertainty_with_accuracy(
            axes[2],
            it_data['parameter_values'],
            it_data['epistemic'],
            it_data['aleatoric'],
            it_data['accuracy'],
            'Information Theoretic'
        )
    
    return fig


# ============================================================================
# RENDER FUNCTIONS
# ============================================================================

def render_benchmark_comparison_plots(api_base_url: str, get_headers_func):
    """
    Render label noise sweep plots comparing research methods with production signals.
    """
    st.markdown("### 📊 UQ Methods Comparison: Label Noise Sweep")
    st.caption("Compare research methods (Gaussian Logits, Information Theoretic) with production signals")
    
    # Fetch research sweeps
    sweeps = fetch_all_sweeps(api_base_url, get_headers_func)
    
    if not sweeps:
        st.info("🔬 No benchmark sweeps found. Run parameter sweeps to see comparison plots.")
        st.markdown("""
        **To generate comparison plots:**
        1. Expand "⚙️ Run New Benchmark" below
        2. Run sweeps for both Gaussian Logits and Information Theoretic methods
        3. Optionally run a production system batch sweep with noise rate parameter
        4. Results will appear here automatically
        """)
        return
    
    # Filter for label noise sweeps
    noise_sweeps = [s for s in sweeps if s.get('sweep_parameter') == 'noise_rate']
    
    if len(noise_sweeps) < 2:
        st.warning("⚠️ Need at least 2 label noise sweeps (Gaussian Logits + Information Theoretic) for comparison")
        return
    
    # Let user select sweeps to compare
    st.markdown("#### Select Sweeps to Compare")
    col1, col2, col3 = st.columns(3)
    
    sweep_options = {
        f"{s['id']}: {s.get('method_name', 'Unknown')}": s['id']
        for s in noise_sweeps
    }
    
    with col1:
        gl_sweep_label = st.selectbox(
            "Gaussian Logits Sweep",
            options=list(sweep_options.keys()),
            key="gl_sweep"
        )
    
    with col2:
        it_sweep_label = st.selectbox(
            "Information Theoretic Sweep",
            options=list(sweep_options.keys()),
            key="it_sweep"
        )
    
    with col3:
        # Optional: production system batch
        include_production = st.checkbox("Include Production Signals", value=False)
        prod_batch_id = None
        if include_production:
            prod_batch_id = st.text_input("Production Batch ID", key="prod_batch")
    
    if not gl_sweep_label or not it_sweep_label:
        return
    
    # Fetch sweep data
    with st.spinner("Loading sweep results..."):
        gl_results = fetch_sweep_results(api_base_url, get_headers_func, sweep_options[gl_sweep_label])
        it_results = fetch_sweep_results(api_base_url, get_headers_func, sweep_options[it_sweep_label])
        
        prod_results = None
        if include_production and prod_batch_id:
            prod_results = fetch_production_sweep_results(api_base_url, get_headers_func, prod_batch_id)
    
    if not gl_results or not it_results:
        st.error("Failed to load sweep results")
        return
    
    # Extract and prepare data
    def extract_sweep_data(results):
        """Extract data from sweep results."""
        data_results = results.get('results', [])
        param_values = []
        epistemic_vals = []
        aleatoric_vals = []
        accuracy_vals = []
        
        for result in sorted(data_results, key=lambda x: x.get('parameter_value', 0)):
            param_values.append(result.get('parameter_value', 0))
            epistemic_vals.append(result.get('epistemic_uncertainty', 0))
            aleatoric_vals.append(result.get('aleatoric_uncertainty', 0))
            accuracy_vals.append(result.get('accuracy', 0))
        
        return {
            'parameter_values': param_values,
            'epistemic': epistemic_vals,
            'aleatoric': aleatoric_vals,
            'accuracy': accuracy_vals
        }
    
    gaussian_data = extract_sweep_data(gl_results)
    it_data = extract_sweep_data(it_results)
    
    # Extract production data if available
    prod_data = None
    if prod_results:
        # TODO: Extract production signals from batch results
        # This requires the batch results to include all 7 signals
        st.info("Production signal extraction not yet implemented")
    
    # Create and display comparison grid
    fig = render_comparison_grid(gaussian_data, it_data, prod_data)
    st.pyplot(fig)
    plt.close()
    
    # Show data tables
    with st.expander("📋 View Raw Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gaussian Logits**")
            df_gl = pd.DataFrame(gaussian_data)
            st.dataframe(df_gl, use_container_width=True)
        with col2:
            st.markdown("**Information Theoretic**")
            df_it = pd.DataFrame(it_data)
            st.dataframe(df_it, use_container_width=True)


def render_single_run(api_base_url: str, get_headers_func, methods_info: Dict):
    """Render single benchmark run configuration."""
    st.markdown("#### 🎯 Single Benchmark Run")
    st.info("Run a single UQ method with specific parameters")
    
    with st.form("single_benchmark_form"):
        method_name = st.selectbox(
            "UQ Method",
            options=list(methods_info.keys()),
            help="Select uncertainty quantification method"
        )
        
        st.markdown("**Dataset Configuration**")
        col1, col2 = st.columns(2)
        with col1:
            noise_rate = st.slider("Noise Rate", 0.0, 1.0, 0.1, 0.05)
            train_samples = st.number_input("Training Samples", 1000, 50000, 5000, 1000)
        with col2:
            test_samples = st.number_input("Test Samples", 100, 10000, 1000, 100)
            epochs = st.number_input("Training Epochs", 1, 100, 10, 1)
        
        submitted = st.form_submit_button("🚀 Run Benchmark", type="primary")
        
        if submitted:
            with st.spinner(f"Running {method_name}..."):
                try:
                    payload = {
                        "method_name": method_name,
                        "config": {
                            "noise_rate": noise_rate,
                            "train_samples": train_samples,
                            "test_samples": test_samples,
                            "epochs": epochs
                        }
                    }
                    
                    response = requests.post(
                        f"{api_base_url}/api/v1/uq-benchmarks/run",
                        json=payload,
                        headers=get_headers_func(),
                        timeout=300
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    st.success(f"✅ Benchmark completed!")
                    st.json(result)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Benchmark failed: {str(e)}")


def render_parameter_sweep(api_base_url: str, get_headers_func, methods_info: Dict):
    """Render parameter sweep configuration."""
    st.markdown("#### 📊 Parameter Sweep")
    st.info("Sweep noise rate to compare uncertainty quantification methods")
    
    with st.form("sweep_benchmark_form"):
        method_name = st.selectbox(
            "UQ Method",
            options=list(methods_info.keys()),
            help="Select uncertainty quantification method",
            key="sweep_method"
        )
        
        st.markdown("**Sweep Configuration**")
        col1, col2, col3 = st.columns(3)
        with col1:
            sweep_start = st.number_input("Start Noise Rate", 0.0, 1.0, 0.0, 0.05)
        with col2:
            sweep_end = st.number_input("End Noise Rate", 0.0, 1.0, 0.9, 0.05)
        with col3:
            sweep_step = st.number_input("Step Size", 0.01, 0.2, 0.1, 0.01)
        
        st.markdown("**Fixed Parameters**")
        col1, col2 = st.columns(2)
        with col1:
            train_samples = st.number_input("Training Samples", 1000, 50000, 5000, 1000, key="sweep_train")
            test_samples = st.number_input("Test Samples", 100, 10000, 1000, 100, key="sweep_test")
        with col2:
            epochs = st.number_input("Training Epochs", 1, 100, 10, 1, key="sweep_epochs")
        
        submitted = st.form_submit_button("🚀 Run Sweep", type="primary")
        
        if submitted:
            sweep_values = np.arange(sweep_start, sweep_end + sweep_step/2, sweep_step).tolist()
            st.info(f"Will run {len(sweep_values)} experiments: {sweep_values}")
            
            with st.spinner(f"Running sweep with {method_name}..."):
                try:
                    payload = {
                        "method_name": method_name,
                        "sweep_parameter": "noise_rate",
                        "sweep_values": sweep_values,
                        "base_config": {
                            "train_samples": train_samples,
                            "test_samples": test_samples,
                            "epochs": epochs
                        }
                    }
                    
                    response = requests.post(
                        f"{api_base_url}/api/v1/uq-benchmarks/sweep",
                        json=payload,
                        headers=get_headers_func(),
                        timeout=1800
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    st.success(f"✅ Sweep completed! Sweep ID: {result.get('sweep_id')}")
                    st.info("🔄 Refresh the page to see the new plot above")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Sweep failed: {str(e)}")


def render_uq_benchmarks_tab(api_base_url: str, get_headers_func):
    """
    Render the UQ Benchmarks tab with visualization plots.
    
    Args:
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
    """
    st.markdown("## 🔬 UQ Benchmarks: Research Methods vs Production Signals")
    
    # Render the comparison plots directly
    render_benchmark_comparison_plots(api_base_url, get_headers_func)
    
    st.markdown("---")
    
    # Configuration and controls below the plots
    with st.expander("⚙️ Run New Benchmark", expanded=False):
        # Fetch available methods
        methods_info = fetch_available_methods(api_base_url, get_headers_func)
        
        if not methods_info:
            st.error("❌ Could not fetch available UQ methods from backend")
            return
        
        # Create tabs for single run and parameter sweep
        single_run_tab, sweep_tab = st.tabs([
            "🎯 Single Run",
            "📊 Parameter Sweep"
        ])
        
        with single_run_tab:
            render_single_run(api_base_url, get_headers_func, methods_info)
        
        with sweep_tab:
            render_parameter_sweep(api_base_url, get_headers_func, methods_info)

# Made with Bob
