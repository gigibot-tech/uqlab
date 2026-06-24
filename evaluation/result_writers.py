"""
Evaluation result writers: Markdown/CSV/JSON formatting and file output.

Pure metric computation lives in :mod:`uqlab.evaluation.metrics`.
The ``results.pt`` read contract lives in :mod:`uqlab.evaluation.artifacts`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _format_auroc_markdown(value: float | None) -> str:
    """Format AUROC for markdown tables; ``None``/NaN → em dash."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(v):
        return "—"
    return f"{v:.4f}"


def persist_experiment_summaries(
    results_dir: Path,
    *,
    summary: dict,
    args: argparse.Namespace,
    split_spec,
    train_size: int,
    eval_sizes: Dict[str, int],
    auroc_rows: List[Tuple[str, float | None, float | None]],
    clf_rows: List[Tuple[str, float]],
) -> None:
    """Write ``summary.json`` and ``summary.md`` (None-safe AUROC formatting)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    markdown_path = results_dir / "summary.md"
    try:
        markdown = build_results_markdown(
            args=args,
            split_spec=split_spec,
            train_size=train_size,
            eval_sizes=eval_sizes,
            auroc_rows=auroc_rows,
            clf_rows=clf_rows,
        )
    except Exception as exc:
        logger.warning("summary.md build failed (%s); writing minimal fallback", exc)
        markdown = (
            "# Fast Uncertainty Classification Results\n\n"
            "Markdown summary could not be generated; see `summary.json`.\n"
        )
    markdown_path.write_text(markdown, encoding="utf-8")


def build_results_markdown(
    *,
    args: argparse.Namespace,
    split_spec,
    train_size: int,
    eval_sizes: Dict[str, int],
    auroc_rows: List[Tuple[str, float | None, float | None]],
    clf_rows: List[Tuple[str, float]],
) -> str:
    """
    Build a Markdown summary of experiment results.
    
    Args:
        args: Command-line arguments
        split_spec: Data split specification
        train_size: Number of training samples
        eval_sizes: Dictionary of evaluation set sizes
        auroc_rows: List of (signal_name, aleatoric_auroc, epistemic_auroc)
        clf_rows: List of (signal_set_name, macro_f1)
        
    Returns:
        Markdown-formatted results string
    """
    lines = [
        "# Fast Uncertainty Classification Results",
        "",
        "## Setup",
        f"- Noise type: `{args.noise_type}`",
        f"- Under-supported classes: `{split_spec.under_supported_classes}`",
        f"- Train size: `{train_size}`",
        f"- Eval clean: `{eval_sizes['clean']}`",
        f"- Eval aleatoric-like: `{eval_sizes['aleatoric_like']}`",
        f"- Eval epistemic-like: `{eval_sizes['epistemic_like']}`",
        f"- DINOv2 backbone: `{args.dinov2_model}`",
        "",
        "## One-vs-Rest AUROC",
        "",
        "| Signal | Aleatoric-like AUROC | Epistemic-like AUROC |",
        "| --- | ---: | ---: |",
    ]
    for name, alea_auc, epis_auc in auroc_rows:
        lines.append(
            f"| {name} | {_format_auroc_markdown(alea_auc)} | {_format_auroc_markdown(epis_auc)} |"
        )

    lines.extend(
        [
            "",
            "## 3-Way Signal Classifier",
            "",
            "| Signal set | Macro-F1 |",
            "| --- | ---: |",
        ]
    )
    for name, score in clf_rows:
        lines.append(f"| {name} | {score:.4f} |")

    return "\n".join(lines) + "\n"


def print_noisy_eval_samples(
    *,
    eval_group_labels: torch.Tensor,
    eval_dataset_index: torch.Tensor,
    eval_clean_labels: torch.Tensor,
    eval_noisy_labels: torch.Tensor,
    eval_is_noisy: torch.Tensor,
    group_names: Dict[int, str],
    max_rows: int = 40,
) -> None:
    """Print CIFAR-10N index + labels for eval points with ``is_noisy=True``."""
    noisy = eval_is_noisy.bool()
    n_noisy = int(noisy.sum().item())
    n_total = int(eval_group_labels.shape[0])
    print(f"\nNoisy eval samples (is_noisy=True): {n_noisy} / {n_total}")
    if n_noisy == 0:
        return

    print(f"  {'dataset_index':>14}  {'group':<16}  clean  noisy")
    shown = 0
    for i in range(n_total):
        if not bool(noisy[i].item()):
            continue
        grp = group_names[int(eval_group_labels[i].item())]
        idx = int(eval_dataset_index[i].item())
        clean = int(eval_clean_labels[i].item())
        nlabel = int(eval_noisy_labels[i].item())
        print(f"  {idx:>14}  {grp:<16}  {clean:>5}  {nlabel:>5}")
        shown += 1
        if shown >= max_rows:
            remaining = n_noisy - shown
            if remaining > 0:
                print(f"  ... and {remaining} more (see per_sample_signals.csv)")
            break


def save_training_data_csv(
    output_path: Path,
    train_dataset,
    config: Optional[dict] = None,
) -> None:
    """
    Save training data statistics to CSV file.
    
    Extracts and saves:
    - dataset_index: Original CIFAR-10N index
    - clean_label: True label
    - noisy_label: Label used for training (may be flipped)
    - is_noisy: Boolean indicating if label was flipped
    
    If config is provided, also saves it as a separate JSON file alongside the CSV.
    
    Args:
        output_path: Path where to save the CSV file
        train_dataset: Training dataset with attributes:
            - clean_labels: Original true labels
            - targets: Labels used for training (may be noisy)
            - is_noisy: Boolean array indicating flipped labels
            - original_indices: Original dataset indices
        config: Optional experiment configuration dict to save alongside CSV
    """
    import pandas as pd
    import json
    
    print("\n" + "="*80)
    print("Saving training data statistics...")
    print("="*80 + "\n")
    
    try:
        # Extract data from dataset
        clean_labels = train_dataset.clean_labels
        noisy_labels = train_dataset.targets
        is_noisy = train_dataset.is_noisy
        indices = train_dataset.original_indices
        
        # Create DataFrame
        df = pd.DataFrame({
            'dataset_index': indices,
            'clean_label': clean_labels,
            'noisy_label': noisy_labels,
            'is_noisy': is_noisy
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Save config as separate JSON file if provided
        if config:
            config_path = output_path.with_suffix('.config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  Config saved to: {config_path}")
        
        # Calculate and print statistics
        total_samples = len(df)
        noisy_samples = df['is_noisy'].sum()
        clean_samples = total_samples - noisy_samples
        noise_rate = noisy_samples / total_samples if total_samples > 0 else 0
        
        print("📊 Training Data Summary:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Clean samples: {clean_samples:,}")
        print(f"  Noisy samples: {noisy_samples:,}")
        print(f"  Noise rate: {noise_rate:.1%}")
        print(f"  Saved to: {output_path}")
        print()
        
    except AttributeError as e:
        print(f"⚠️  Warning: Could not save training data statistics")
        print(f"   Dataset missing required attributes: {e}")
        print(f"   Skipping training_data.csv generation")
        print()


def save_per_sample_csv(
    output_path: Path,
    eval_group_labels: torch.Tensor,
    eval_clean_labels: torch.Tensor,
    eval_is_noisy: torch.Tensor,
    signal_table: Dict[str, torch.Tensor],
    group_names: Dict[int, str],
    *,
    eval_noisy_labels: torch.Tensor | None = None,
    eval_dataset_index: torch.Tensor | None = None,
    print_noisy_summary: bool = True,
) -> None:
    """
    Save per-sample signals to CSV file.

    When ``eval_noisy_labels`` / ``eval_dataset_index`` are provided, adds
    ``noisy_label`` and ``dataset_index`` columns (CIFAR-10N index).
    """
    n = int(eval_group_labels.shape[0])
    if eval_noisy_labels is None:
        eval_noisy_labels = eval_clean_labels
    if eval_dataset_index is None:
        eval_dataset_index = torch.full((n,), -1, dtype=torch.long)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "group",
            "dataset_index",
            "clean_label",
            "noisy_label",
            "is_noisy",
        ] + list(signal_table.keys())
        writer.writerow(header)

        for i in range(n):
            row = [
                group_names[int(eval_group_labels[i].item())],
                int(eval_dataset_index[i].item()),
                int(eval_clean_labels[i].item()),
                int(eval_noisy_labels[i].item()),
                bool(eval_is_noisy[i].item()),
            ]
            for signal_name in signal_table.keys():
                row.append(float(signal_table[signal_name][i].item()))
            writer.writerow(row)

    if print_noisy_summary:
        print_noisy_eval_samples(
            eval_group_labels=eval_group_labels,
            eval_dataset_index=eval_dataset_index,
            eval_clean_labels=eval_clean_labels,
            eval_noisy_labels=eval_noisy_labels,
            eval_is_noisy=eval_is_noisy,
            group_names=group_names,
        )
