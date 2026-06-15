"""Plotting utilities for validation notebooks."""

from __future__ import annotations

import pandas as pd

from .constants import ARCHITECTURE_STYLES, SWEEP_TO_X, UNCERTAINTY_SIGNALS


def plot_individual_signals(df, sweep_type="label_noise"):
    """
    Plot individual uncertainty signals for all architectures.
    
    Simplified wrapper that uses sweep_type parameter for backward compatibility.
    
    Args:
        df: DataFrame with metrics
        sweep_type: Either "label_noise" or "dataset_size"
    """
    # Call the detailed implementation with sweep_name parameter
    plot_individual_signals_detailed(df, sweep_name=sweep_type, architectures=None)


def plot_individual_signals_detailed(df: pd.DataFrame, sweep_name: str, architectures: list[str] | None = None) -> None:
    """
    Grid of per-signal AUROC panels (epistemic + aleatoric) plus accuracy.

    Epistemic and aleatoric AUROC are separate detection tasks and must not be averaged.
    """
    import matplotlib.pyplot as plt

    if df.empty:
        print("Warning: Empty dataframe, cannot plot individual signals")
        return

    if sweep_name not in SWEEP_TO_X:
        raise ValueError(f"Unknown sweep_name={sweep_name!r}")

    x_col = SWEEP_TO_X[sweep_name]
    x_label = "Dataset Size (samples per class)" if sweep_name == "dataset_size" else "Label Noise (%)"

    if architectures is None:
        architectures = list(ARCHITECTURE_STYLES.keys())

    available_archs = [arch for arch in architectures if arch in df["architecture"].values]
    if not available_archs:
        print(f"Warning: No data found for architectures: {architectures}")
        return

    available_signals: list[tuple[str, str, str, str]] = []
    signal_columns: dict[str, tuple[str, str]] = {}
    for signal_key, signal_name in UNCERTAINTY_SIGNALS.items():
        aleatoric_col = f"{signal_key}_aleatoric_auroc"
        epistemic_col = f"{signal_key}_epistemic_auroc"
        if aleatoric_col in df.columns and epistemic_col in df.columns:
            available_signals.append(signal_name)
            signal_columns[signal_name] = (aleatoric_col, epistemic_col)

    if not available_signals:
        print("Warning: No individual uncertainty signal columns found in dataframe")
        return

    n_plots = len(available_signals) + 1
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    ax = axes[0]
    for arch in available_archs:
        arch_df = df[df["architecture"] == arch].sort_values(x_col)
        if arch_df.empty:
            continue
        style = ARCHITECTURE_STYLES.get(arch, {"color": "gray", "marker": "o"})
        ax.plot(
            arch_df[x_col],
            arch_df["accuracy"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            label=arch,
        )
    ax.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    for idx, signal_name in enumerate(available_signals, start=1):
        ax = axes[idx]
        aleatoric_col, epistemic_col = signal_columns[signal_name]
        for arch in available_archs:
            arch_df = df[df["architecture"] == arch].sort_values(x_col)
            if arch_df.empty:
                continue
            style = ARCHITECTURE_STYLES.get(arch, {"color": "gray", "marker": "o"})
            ax.plot(
                arch_df[x_col],
                arch_df[epistemic_col],
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                linestyle="-",
                label=f"{arch} - Epistemic",
            )
            ax.plot(
                arch_df[x_col],
                arch_df[aleatoric_col],
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                linestyle=":",
                label=f"{arch} - Aleatoric",
            )
        ax.set_title(signal_name, fontsize=12, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("AUROC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Individual Uncertainty Signals: {sweep_name.replace('_', ' ').title()} Sweep",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()
    plt.show()


def plot_dual_axis(df: pd.DataFrame, sweep_name: str, architecture: str) -> None:
    """
    Plot accuracy and uncertainty on dual y-axes for a single architecture.
    
    Args:
        df: DataFrame with metrics
        sweep_name: Either "dataset_size" or "label_noise"
        architecture: Architecture name to plot
    """
    import matplotlib.pyplot as plt
    
    if df.empty:
        print("Warning: Empty dataframe, cannot plot")
        return
    
    if sweep_name not in SWEEP_TO_X:
        raise ValueError(f"Unknown sweep_name={sweep_name!r}")
    
    x_col = SWEEP_TO_X[sweep_name]
    x_label = "Dataset Size (samples per class)" if sweep_name == "dataset_size" else "Label Noise (%)"
    
    # Filter to the specified architecture
    arch_df = df[df["architecture"] == architecture]
    arch_df = arch_df.loc[arch_df[x_col].argsort()]
    if arch_df.empty:
        print(f"Warning: No data found for architecture: {architecture}")
        return
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(arch_df[x_col], arch_df["accuracy"], color=color, marker='o', linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot uncertainties on right y-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Uncertainty', color=color)
    
    if "mean_total_uncertainty" in arch_df.columns:
        ax2.plot(arch_df[x_col], arch_df["mean_total_uncertainty"], 
                color='tab:red', marker='s', linewidth=2, label='Total', linestyle='-')
    if "mean_epistemic_uncertainty" in arch_df.columns:
        ax2.plot(arch_df[x_col], arch_df["mean_epistemic_uncertainty"], 
                color='tab:orange', marker='^', linewidth=2, label='Epistemic', linestyle='--')
    if "mean_aleatoric_uncertainty" in arch_df.columns:
        ax2.plot(arch_df[x_col], arch_df["mean_aleatoric_uncertainty"], 
                color='tab:green', marker='v', linewidth=2, label='Aleatoric', linestyle=':')
    
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.title(f"{architecture}: {sweep_name.replace('_', ' ').title()} Sweep", 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Made with Bob
