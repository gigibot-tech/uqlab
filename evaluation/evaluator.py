"""
Evaluation metrics and result formatting for uncertainty classification.

Provides:
- Binary AUROC computation
- Confusion matrix and macro-F1 score
- 3-way signal classifier training
- Results formatting (Markdown, CSV)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def binary_auroc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    """
    Calculate binary AUROC (Area Under ROC Curve).
    
    Computes the probability that a randomly chosen positive sample
    has a higher score than a randomly chosen negative sample.
    
    Args:
        scores: Uncertainty scores [N]
        positives: Boolean mask for positive class [N]
        
    Returns:
        AUROC value in [0, 1], or NaN if one class is empty
    """
    scores = scores.flatten().float()
    positives = positives.flatten().bool()
    pos_scores = scores[positives]
    neg_scores = scores[~positives]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return float("nan")
    pairwise = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float() * 0.5
    return float((pairwise + ties).mean().item())


def confusion_matrix(num_classes: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        num_classes: Number of classes
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.long(), y_pred.long()):
        cm[int(t), int(p)] += 1
    return cm


def macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    """
    Calculate macro-averaged F1 score.
    
    Computes F1 for each class independently, then averages.
    Gives equal weight to all classes regardless of support.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        num_classes: Number of classes
        
    Returns:
        Macro-averaged F1 score
    """
    cm = confusion_matrix(num_classes, y_true, y_pred).float()
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn).item()
        if denom == 0:
            f1s.append(0.0)
        else:
            f1s.append(float((2 * tp / (2 * tp + fp + fn)).item()))
    return float(sum(f1s) / len(f1s))


def standardize(train_x: torch.Tensor, test_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standardize features using training set statistics.
    
    Args:
        train_x: Training features [N_train, D]
        test_x: Test features [N_test, D]
        
    Returns:
        Tuple of (standardized_train, standardized_test)
    """
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (test_x - mean) / std


def train_signal_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    *,
    device: torch.device,
    epochs: int = 300,
    learning_rate: float = 0.05,
) -> torch.Tensor:
    """
    Train a simple linear classifier on uncertainty signals.
    
    Used to evaluate how well different signal combinations can
    distinguish between clean, aleatoric, and epistemic samples.
    
    Args:
        x_train: Training features [N_train, D]
        y_train: Training labels [N_train]
        x_test: Test features [N_test, D]
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate
        
    Returns:
        Predicted labels for test set [N_test]
    """
    x_train, x_test = standardize(x_train, x_test)
    model = nn.Linear(int(x_train.shape[1]), int(y_train.max().item()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(x_test).argmax(dim=1).cpu()
    return preds


def split_group_balanced_targets(
    y: torch.Tensor, 
    seed: int, 
    train_fraction: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split data into train/test with balanced class distribution.
    
    Ensures each class has the same train/test split ratio.
    
    Args:
        y: Labels [N]
        seed: Random seed
        train_fraction: Fraction of data for training
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls in sorted(set(int(v) for v in y.tolist())):
        cls_idx = np.where(y.numpy() == cls)[0]
        rng.shuffle(cls_idx)
        cut = max(1, int(len(cls_idx) * train_fraction))
        train_idx.extend(cls_idx[:cut].tolist())
        test_idx.extend(cls_idx[cut:].tolist())
    return torch.as_tensor(train_idx, dtype=torch.long), torch.as_tensor(test_idx, dtype=torch.long)


def predict_eval_groups_single_signal(
    signal_values: torch.Tensor,
    eval_group_labels: torch.Tensor,
    *,
    seed: int,
    train_fraction: float = 0.5,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train a one-feature linear 3-way classifier; return predicted group id per sample.

    Uses the same balanced train/test split as :func:`evaluate_three_way_classification`.
    Predictions are produced for every sample (train + test) after fitting on the train split.

    Returns
    -------
    predicted_labels
        Integer group ids ``0=clean, 1=aleatoric, 2=epistemic`` [N].
    is_test_mask
        True where the sample was in the held-out test split [N].
    """
    if device is None:
        device = torch.device("cpu")

    n = int(signal_values.shape[0])
    x = signal_values.reshape(n, 1).float()
    y = eval_group_labels.long()

    train_idx, test_idx = split_group_balanced_targets(y, seed, train_fraction)
    x_train = x[train_idx]
    y_train = y[train_idx]

    x_train_s, x_all_s = standardize(x_train, x)
    model = nn.Linear(1, int(y.max().item()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    x_train_s = x_train_s.to(device)
    y_train = y_train.to(device)
    x_all_s = x_all_s.to(device)

    for _ in range(300):
        optimizer.zero_grad()
        loss = criterion(model(x_train_s), y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted_labels = model(x_all_s).argmax(dim=1).cpu()

    is_test_mask = torch.zeros(n, dtype=torch.bool)
    is_test_mask[test_idx] = True
    return predicted_labels, is_test_mask


def evaluate_three_way_classification(
    signal_table: Dict[str, torch.Tensor],
    eval_group_labels: torch.Tensor,
    device: torch.device,
    seed: int,
    train_fraction: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Evaluate three-way classification (clean/aleatoric/epistemic) using different signal combinations.
    
    Trains simple linear classifiers on different signal subsets and evaluates
    their ability to distinguish between uncertainty types.
    
    Args:
        signal_table: Dictionary of uncertainty signals
        eval_group_labels: Ground truth group labels [N]
        device: Device to train on
        seed: Random seed for train/test split
        train_fraction: Fraction of data for training
        
    Returns:
        List of (signal_set_name, macro_f1_score) tuples
    """
    # Build signal matrices for different combinations
    signal_matrix_predictive = torch.stack(
        [
            signal_table["msp_uncertainty"],
            signal_table["predictive_entropy"],
            signal_table["mutual_info"],
        ],
        dim=1,
    )
    
    # Build attribution signal matrix from available signals only
    attribution_signals = []
    
    # Core attribution signals (should always be present)
    if "inverse_coherence" in signal_table:
        attribution_signals.append(signal_table["inverse_coherence"])
    if "dominance" in signal_table:
        attribution_signals.append(signal_table["dominance"])
    
    # Logit-based signals (via Representer Theorem)
    if "inverse_mass" in signal_table:
        attribution_signals.append(signal_table["inverse_mass"])
    if "inverse_logit_magnitude" in signal_table:
        attribution_signals.append(signal_table["inverse_logit_magnitude"])
    
    # Stack if we have any attribution signals
    if attribution_signals:
        signal_matrix_attribution = torch.stack(attribution_signals, dim=1)
    else:
        # Fallback: use zeros if no attribution signals
        signal_matrix_attribution = torch.zeros((len(eval_group_labels), 1))
    
    signal_matrix_combined = torch.cat([signal_matrix_predictive, signal_matrix_attribution], dim=1)
    
    # Optional: Add compound signal if available
    if "compound_uncertainty" in signal_table:
        signal_matrix_enhanced = torch.cat(
            [signal_matrix_combined, signal_table["compound_uncertainty"].unsqueeze(1)],
            dim=1,
        )
        signal_sets = [
            ("predictive_only", signal_matrix_predictive),
            ("attribution_only", signal_matrix_attribution),
            ("combined", signal_matrix_combined),
            ("enhanced_with_hybrid", signal_matrix_enhanced),
        ]
    else:
        signal_sets = [
            ("predictive_only", signal_matrix_predictive),
            ("attribution_only", signal_matrix_attribution),
            ("combined", signal_matrix_combined),
        ]
    
    # Split data into train/test with balanced classes
    clf_train_idx, clf_test_idx = split_group_balanced_targets(
        eval_group_labels,
        seed=seed + 1,
        train_fraction=train_fraction
    )
    
    # Train and evaluate classifiers
    results = []
    for name, signal_matrix in signal_sets:
        preds = train_signal_classifier(
            signal_matrix[clf_train_idx],
            eval_group_labels[clf_train_idx],
            signal_matrix[clf_test_idx],
            device=device,
        )
        score = macro_f1(eval_group_labels[clf_test_idx], preds, num_classes=3)
        results.append((name, score))
    
    return results


def build_results_markdown(
    *,
    args: argparse.Namespace,
    split_spec,
    train_size: int,
    eval_sizes: Dict[str, int],
    auroc_rows: List[Tuple[str, float, float]],
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
        lines.append(f"| {name} | {alea_auc:.4f} | {epis_auc:.4f} |")

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
