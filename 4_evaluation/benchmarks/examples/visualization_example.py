"""
Example: Creating comparison visualizations between research methods and production signals.

This script demonstrates how to:
1. Run benchmarks with both research methods and production system
2. Collect results from both approaches
3. Create side-by-side comparison plots

Usage:
    python examples/visualization_example.py
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_benchmark_comparison_grid, plot_signal_correlation_heatmap


def simulate_label_noise_sweep():
    """
    Simulate a label noise sweep experiment.
    
    In a real scenario, you would:
    1. Run Gaussian Logits method with different noise rates
    2. Run your production system with same noise rates
    3. Collect all 7 signals from production system
    4. Compare results
    """
    # Noise rates to test
    noise_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # Simulate research method results (Gaussian Logits)
    # In reality, these come from running the uq_benchmarks package
    research_results = {
        'gaussian_logits': {
            'epistemic': [0.05, 0.06, 0.07, 0.09, 0.12],
            'aleatoric': [0.02, 0.05, 0.10, 0.15, 0.20],
            'accuracy': [0.95, 0.92, 0.88, 0.82, 0.75]
        }
    }
    
    # Simulate production signals results
    # In reality, these come from your existing production system
    # running the same experiments
    production_signals = {
        # General uncertainty signals
        'msp_uncertainty': [0.10, 0.12, 0.15, 0.18, 0.22],
        'predictive_entropy': [0.08, 0.10, 0.13, 0.16, 0.20],
        'mutual_info': [0.06, 0.08, 0.11, 0.14, 0.18],
        
        # Aleatoric indicator
        'inverse_coherence': [0.03, 0.06, 0.11, 0.16, 0.21],
        
        # Epistemic indicators
        'dominance': [0.04, 0.05, 0.07, 0.09, 0.11],
        'inverse_mass': [0.04, 0.05, 0.08, 0.10, 0.13],  # Best epistemic
        
        # Other
        'inverse_logit_magnitude': [0.07, 0.09, 0.12, 0.15, 0.19],
        
        # Accuracy (same model, so should be similar)
        'accuracy': [0.94, 0.91, 0.87, 0.81, 0.74]
    }
    
    return noise_rates, research_results, production_signals


def simulate_dataset_size_sweep():
    """
    Simulate a dataset size sweep experiment.
    
    Tests how uncertainty changes as training data decreases.
    """
    dataset_sizes = [5000, 4000, 3000, 2000, 1000]
    
    research_results = {
        'gaussian_logits': {
            'epistemic': [0.05, 0.06, 0.08, 0.11, 0.15],
            'aleatoric': [0.03, 0.03, 0.04, 0.04, 0.05],
            'accuracy': [0.95, 0.93, 0.90, 0.85, 0.78]
        }
    }
    
    production_signals = {
        'msp_uncertainty': [0.08, 0.09, 0.11, 0.14, 0.18],
        'predictive_entropy': [0.07, 0.08, 0.10, 0.13, 0.17],
        'mutual_info': [0.05, 0.06, 0.08, 0.11, 0.15],
        'inverse_coherence': [0.03, 0.03, 0.04, 0.05, 0.06],
        'dominance': [0.04, 0.05, 0.07, 0.10, 0.14],
        'inverse_mass': [0.04, 0.05, 0.07, 0.10, 0.14],
        'inverse_logit_magnitude': [0.06, 0.07, 0.09, 0.12, 0.16],
        'accuracy': [0.94, 0.92, 0.89, 0.84, 0.77]
    }
    
    return dataset_sizes, research_results, production_signals


def main():
    """Run visualization examples."""
    
    print("=" * 60)
    print("UQ Benchmarks Visualization Examples")
    print("=" * 60)
    
    # Example 1: Label Noise Sweep
    print("\n1. Creating label noise sweep comparison...")
    noise_rates, research_noise, production_noise = simulate_label_noise_sweep()
    
    fig1 = plot_benchmark_comparison_grid(
        research_results=research_noise,
        production_signals=production_noise,
        parameter_values=noise_rates,
        parameter_name='noise_rate',
        title='Label Noise Sweep: Research Methods vs Production Signals'
    )
    
    # Save figure
    fig1.savefig('label_noise_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: label_noise_comparison.png")
    
    # Example 2: Dataset Size Sweep
    print("\n2. Creating dataset size sweep comparison...")
    dataset_sizes, research_size, production_size = simulate_dataset_size_sweep()
    
    fig2 = plot_benchmark_comparison_grid(
        research_results=research_size,
        production_signals=production_size,
        parameter_values=dataset_sizes,
        parameter_name='dataset_size',
        title='Dataset Size Sweep: Research Methods vs Production Signals'
    )
    
    fig2.savefig('dataset_size_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: dataset_size_comparison.png")
    
    # Example 3: Signal Correlation Analysis
    print("\n3. Creating signal correlation heatmap...")
    
    # Combine all signals for correlation analysis
    all_signals = {**production_noise}
    all_signals.pop('accuracy')  # Remove accuracy for cleaner correlation
    
    fig3 = plot_signal_correlation_heatmap(
        signals_data=all_signals,
        title='Production Signals Correlation Matrix'
    )
    
    fig3.savefig('signal_correlation.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: signal_correlation.png")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nKey Insights to Look For:")
    print("1. Do production signals track research methods?")
    print("2. Which production signal best matches epistemic uncertainty?")
    print("3. Which production signal best matches aleatoric uncertainty?")
    print("4. Are any production signals redundant (high correlation)?")
    print("5. How does accuracy degrade with noise/dataset size?")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()

# Made with Bob
