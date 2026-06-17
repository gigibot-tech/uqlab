"""
Visualization module for UQ benchmarks.

Adapts plotting code from uq_disentanglement_comparison research package
to work with production uncertainty signals.

Creates subplot grids comparing:
- Research methods (Gaussian Logits, Information Theoretic) - each in own subplot
- Production signals (all 7 signals) - each in own subplot
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Dict, Optional, Tuple
import pandas as pd


def plot_benchmark_comparison_grid(
    research_results: Dict[str, Dict[str, List[float]]],
    production_signals: Dict[str, List[float]],
    parameter_values: List[float],
    parameter_name: str = "noise_rate",
    title: str = "UQ Method Comparison"
) -> plt.Figure:
    """
    Create grid of subplots comparing research methods vs all production signals.
    
    Layout:
    Row 1: Research method 1 (Gaussian Logits)
    Row 2: Research method 2 (Information Theoretic) 
    Row 3: Production signal 1 (msp_uncertainty)
    Row 4: Production signal 2 (predictive_entropy)
    ... (one row per signal)
    
    Each subplot shows: epistemic (blue), aleatoric (orange), accuracy (green, secondary axis)
    
    Args:
        research_results: Results from research methods
            Format: {
                'gaussian_logits': {
                    'epistemic': [values],
                    'aleatoric': [values],
                    'accuracy': [values]
                },
                'information_theoretic': {...}
            }
        production_signals: All 7 production signals + accuracy
            Format: {
                'msp_uncertainty': [values],
                'predictive_entropy': [values],
                'mutual_info': [values],
                'inverse_coherence': [values],  # Aleatoric indicator
                'dominance': [values],  # Epistemic indicator
                'inverse_mass': [values],  # Best epistemic
                'inverse_logit_magnitude': [values],
                'accuracy': [values]
            }
        parameter_values: Values of swept parameter
        parameter_name: Name of swept parameter
        title: Overall figure title
    
    Returns:
        matplotlib Figure object
    """
    # Calculate number of rows needed
    n_research = len(research_results)
    n_production = len([k for k in production_signals.keys() if k != 'accuracy'])
    n_rows = n_research + n_production
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12, 3 * n_rows),
        sharex=True
    )
    
    if n_rows == 1:
        axes = [axes]
    
    row_idx = 0
    accuracy_ax = None  # Share accuracy axis across all plots
    
    # Plot research methods
    for method_name, method_results in research_results.items():
        ax = axes[row_idx]
        accuracy_ax = _plot_uncertainty_and_accuracy(
            ax,
            parameter_values,
            method_results.get('epistemic', []),
            method_results.get('aleatoric', []),
            method_results.get('accuracy', []),
            title=f"Research: {_format_method_name(method_name)}",
            is_first=(row_idx == 0),
            is_last=(row_idx == n_rows - 1),
            accuracy_ax_to_share=accuracy_ax
        )
        row_idx += 1
    
    # Define signal categories for labeling
    signal_categories = {
        'msp_uncertainty': 'General',
        'predictive_entropy': 'General',
        'mutual_info': 'General',
        'inverse_coherence': 'Aleatoric Indicator',
        'dominance': 'Epistemic Indicator',
        'inverse_mass': 'Epistemic Indicator (Best)',
        'inverse_logit_magnitude': 'General'
    }
    
    # Plot production signals
    accuracy_values = production_signals.get('accuracy', [])
    for signal_name in ['msp_uncertainty', 'predictive_entropy', 'mutual_info',
                        'inverse_coherence', 'dominance', 'inverse_mass',
                        'inverse_logit_magnitude']:
        if signal_name in production_signals:
            ax = axes[row_idx]
            
            # For production signals, we don't have separate epistemic/aleatoric
            # So we plot the signal value as "uncertainty" and show its category
            signal_values = production_signals[signal_name]
            category = signal_categories.get(signal_name, 'General')
            
            accuracy_ax = _plot_single_signal_and_accuracy(
                ax,
                parameter_values,
                signal_values,
                accuracy_values,
                signal_name=signal_name,
                category=category,
                is_first=(row_idx == 0),
                is_last=(row_idx == n_rows - 1),
                accuracy_ax_to_share=accuracy_ax
            )
            row_idx += 1
    
    # Set overall title and x-label
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    axes[-1].set_xlabel(_format_parameter_name(parameter_name), fontsize=12)
    
    plt.tight_layout()
    return fig


def _plot_uncertainty_and_accuracy(
    ax: plt.Axes,
    parameter_values: List[float],
    epistemic_values: List[float],
    aleatoric_values: List[float],
    accuracy_values: List[float],
    title: str,
    is_first: bool = False,
    is_last: bool = False,
    accuracy_ax_to_share: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot epistemic, aleatoric uncertainties and accuracy on same axes.
    
    Adapted from disentanglement/benchmarks/plotting.py
    """
    # Plot uncertainties on primary y-axis
    ax.plot(
        parameter_values,
        epistemic_values,
        label="Epistemic",
        color='#1f77b4',  # Blue
        marker='o',
        linewidth=2
    )
    
    ax.plot(
        parameter_values,
        aleatoric_values,
        label="Aleatoric",
        color='#ff7f0e',  # Orange
        marker='s',
        linewidth=2
    )
    
    # Create secondary y-axis for accuracy
    ax2 = ax.twinx()
    
    # Share y-axis with other accuracy axes
    if accuracy_ax_to_share is not None:
        ax2.sharey(accuracy_ax_to_share)
    
    ax2.plot(
        parameter_values,
        accuracy_values,
        label="Accuracy",
        color='#2ca02c',  # Green
        marker='^',
        linewidth=2
    )
    
    # Set labels
    ax.set_ylabel(title, fontsize=10, fontweight='bold')
    
    if is_last:
        ax2.set_ylabel("Accuracy", color='#2ca02c', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
    else:
        ax2.tick_params(axis='y', labelleft=False, labelright=False)
    
    # Add legend (only on first subplot)
    if is_first:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_axisbelow(True)
    
    return ax2


def _plot_single_signal_and_accuracy(
    ax: plt.Axes,
    parameter_values: List[float],
    signal_values: List[float],
    accuracy_values: List[float],
    signal_name: str,
    category: str,
    is_first: bool = False,
    is_last: bool = False,
    accuracy_ax_to_share: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot a single production signal and accuracy.
    
    Args:
        ax: Matplotlib axes
        parameter_values: X-axis values
        signal_values: Signal uncertainty values
        accuracy_values: Accuracy values
        signal_name: Name of the signal
        category: Category label (e.g., "Aleatoric Indicator")
        is_first: Is this the first subplot?
        is_last: Is this the last subplot?
        accuracy_ax_to_share: Accuracy axis to share y-scale with
    
    Returns:
        Secondary axes for accuracy
    """
    # Choose color based on category
    if 'Aleatoric' in category:
        color = '#ff7f0e'  # Orange
        marker = 's'
    elif 'Epistemic' in category:
        color = '#1f77b4'  # Blue
        marker = 'o'
    else:
        color = '#9467bd'  # Purple
        marker = 'd'
    
    # Plot signal on primary y-axis
    ax.plot(
        parameter_values,
        signal_values,
        label=signal_name,
        color=color,
        marker=marker,
        linewidth=2
    )
    
    # Create secondary y-axis for accuracy
    ax2 = ax.twinx()
    
    # Share y-axis with other accuracy axes
    if accuracy_ax_to_share is not None:
        ax2.sharey(accuracy_ax_to_share)
    
    ax2.plot(
        parameter_values,
        accuracy_values,
        label="Accuracy",
        color='#2ca02c',  # Green
        marker='^',
        linewidth=2
    )
    
    # Set labels
    formatted_name = signal_name.replace('_', ' ').title()
    ax.set_ylabel(f"Production: {formatted_name}\n({category})", 
                  fontsize=10, fontweight='bold')
    
    if is_last:
        ax2.set_ylabel("Accuracy", color='#2ca02c', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='#2ca02c')
    else:
        ax2.tick_params(axis='y', labelleft=False, labelright=False)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='both')
    ax.set_axisbelow(True)
    
    return ax2


def plot_signal_correlation_heatmap(
    signals_data: Dict[str, List[float]],
    title: str = "Signal Correlation Matrix"
) -> plt.Figure:
    """
    Create correlation heatmap between all signals.
    
    Useful for understanding which signals provide redundant vs unique information.
    
    Args:
        signals_data: Dictionary of signal name -> values
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    # Convert to DataFrame for easy correlation calculation
    df = pd.DataFrame(signals_data)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(
                j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                ha="center", va="center",
                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                fontsize=8
            )
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def _format_method_name(method_name: str) -> str:
    """Format method name for display."""
    name_map = {
        'gaussian_logits': 'Gaussian Logits',
        'information_theoretic': 'Information Theoretic',
        'it': 'Information Theoretic',
        'dualxda': 'DualXDA'
    }
    return name_map.get(method_name, method_name.replace('_', ' ').title())


def _format_parameter_name(parameter_name: str) -> str:
    """Format parameter name for display."""
    name_map = {
        'noise_rate': 'Label Noise Rate',
        'dataset_size': 'Dataset Size',
        'under_train_per_class': 'Samples per Under-supported Class',
        'regular_train_per_class': 'Samples per Regular Class'
    }
    return name_map.get(parameter_name, parameter_name.replace('_', ' ').title())


# Example usage for testing
if __name__ == "__main__":
    # Simulate some data
    parameter_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # Research methods results
    research_results = {
        'gaussian_logits': {
            'epistemic': [0.05, 0.06, 0.07, 0.09, 0.12],
            'aleatoric': [0.02, 0.05, 0.10, 0.15, 0.20],
            'accuracy': [0.95, 0.92, 0.88, 0.82, 0.75]
        }
    }
    
    # Production signals results (all 7 signals)
    production_signals = {
        'msp_uncertainty': [0.10, 0.12, 0.15, 0.18, 0.22],
        'predictive_entropy': [0.08, 0.10, 0.13, 0.16, 0.20],
        'mutual_info': [0.06, 0.08, 0.11, 0.14, 0.18],
        'inverse_coherence': [0.03, 0.06, 0.11, 0.16, 0.21],  # Aleatoric
        'dominance': [0.04, 0.05, 0.07, 0.09, 0.11],  # Epistemic
        'inverse_mass': [0.04, 0.05, 0.08, 0.10, 0.13],  # Best epistemic
        'inverse_logit_magnitude': [0.07, 0.09, 0.12, 0.15, 0.19],
        'accuracy': [0.94, 0.91, 0.87, 0.81, 0.74]
    }
    
    # Create comparison plot
    fig = plot_benchmark_comparison_grid(
        research_results,
        production_signals,
        parameter_values,
        parameter_name='noise_rate',
        title='Label Noise Sweep: Research Methods vs Production Signals'
    )
    
    plt.show()

# Made with Bob
