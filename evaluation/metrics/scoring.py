"""
Evaluation metrics for uncertainty classification (pure computation).

Provides:
- Binary AUROC computation
- Confusion matrix and macro-F1 score
- 3-way signal classifier training

File/format output lives in :mod:`uqlab.evaluation.metrics.artifacts`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def binary_auroc(scores: torch.Tensor, positives: torch.Tensor) -> float:
    """AUROC: does a higher score rank positive samples above negatives? NaN if only one class."""
    scores = scores.flatten().float()
    positives = positives.flatten().bool()
    pos_scores = scores[positives]
    neg_scores = scores[~positives]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return float("nan")
    pairwise = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float() * 0.5
    return float((pairwise + ties).mean().item())


def binary_auroc_or_none(scores: torch.Tensor, positives: torch.Tensor) -> float | None:
    """Same as binary_auroc; returns None when either class is missing (AUROC undefined)."""
    positives = positives.flatten().bool()
    if positives.sum().item() == 0 or (~positives).sum().item() == 0:
        return None
    value = binary_auroc(scores, positives)
    if value != value:
        return None
    return value


def auroc_skip_reason(n_pos: int, n_neg: int, *, axis: str = "aleatoric") -> str | None:
    """Why AUROC was skipped: eval pool had only one class for this axis (e.g. no noisy samples)."""
    if n_pos == 0:
        return f"no {axis} positive samples"
    if n_neg == 0:
        return f"no {axis} negative samples"
    return None


def binary_auroc_vs_group(
    scores: torch.Tensor,
    group_labels: torch.Tensor,
    *,
    positive_group: int,
    negative_group: int,
) -> float | None:
    """AUROC with positives from *positive_group* and negatives from *negative_group* only."""
    group_labels = group_labels.flatten().long()
    pos = group_labels == positive_group
    neg = group_labels == negative_group
    if pos.sum().item() == 0 or neg.sum().item() == 0:
        return None
    mask = pos | neg
    return binary_auroc_or_none(scores.flatten()[mask], pos[mask])


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


_PREDICTIVE_SIGNAL_KEYS = ("msp_uncertainty", "predictive_entropy", "mutual_info")


def _stack_available_signals(
    signal_table: Dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    """Stack signal columns that are present in ``signal_table`` (order preserved)."""
    from uqlab.evaluation.signals.catalog import resolve_signal_table_key

    cols: list[torch.Tensor] = []
    for key in keys:
        resolved = resolve_signal_table_key(signal_table, key)
        if resolved is not None:
            cols.append(signal_table[resolved])
    if not cols:
        return None
    return torch.stack(cols, dim=1)


_ATTRIBUTION_SIGNAL_KEYS = (
    "inverse_coherence",
    "inverse_dominance",
    "inverse_mass",
)


def evaluate_three_way_classification(
    signal_table: Dict[str, torch.Tensor],
    eval_group_labels: torch.Tensor,
    device: torch.device,
    seed: int,
    train_fraction: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Evaluate N-way group classification using different signal combinations.

    ``num_classes`` is inferred from ``eval_group_labels`` (3 for legacy, 4 for four-region).
    
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
    n = len(eval_group_labels)

    # Predictive stack: only columns that were computed (e.g. mutual_info omitted when dropout=0)
    signal_matrix_predictive = _stack_available_signals(signal_table, _PREDICTIVE_SIGNAL_KEYS)
    
    from uqlab.evaluation.signals.catalog import resolve_signal_table_key

    # Build attribution + logit matrices from available columns (legacy or suffixed ids).
    attribution_signals = []
    for key in _ATTRIBUTION_SIGNAL_KEYS:
        resolved = resolve_signal_table_key(signal_table, key)
        if resolved is not None:
            attribution_signals.append(signal_table[resolved])
        elif key == "inverse_dominance" and "dominance" in signal_table:
            attribution_signals.append(1.0 - signal_table["dominance"].clamp(0.0, 1.0))

    logit_key = resolve_signal_table_key(signal_table, "inverse_logit_magnitude")
    if logit_key is not None:
        attribution_signals.append(signal_table[logit_key])
    
    # Stack if we have any attribution signals
    if attribution_signals:
        signal_matrix_attribution = torch.stack(attribution_signals, dim=1)
    else:
        # Fallback: use zeros if no attribution signals
        signal_matrix_attribution = torch.zeros((n, 1))

    combined_parts: list[torch.Tensor] = []
    if signal_matrix_predictive is not None:
        combined_parts.append(signal_matrix_predictive)
    combined_parts.append(signal_matrix_attribution)
    signal_matrix_combined = (
        combined_parts[0]
        if len(combined_parts) == 1
        else torch.cat(combined_parts, dim=1)
    )

    signal_sets: list[tuple[str, torch.Tensor]] = []
    if signal_matrix_predictive is not None:
        signal_sets.append(("predictive_only", signal_matrix_predictive))
    signal_sets.append(("attribution_only", signal_matrix_attribution))
    signal_sets.append(("combined", signal_matrix_combined))

    # Optional: Add compound signal if available
    if "compound_uncertainty" in signal_table:
        signal_matrix_enhanced = torch.cat(
            [signal_matrix_combined, signal_table["compound_uncertainty"].unsqueeze(1)],
            dim=1,
        )
        signal_sets.append(("enhanced_with_hybrid", signal_matrix_enhanced))
    
    # Split data into train/test with balanced classes
    clf_train_idx, clf_test_idx = split_group_balanced_targets(
        eval_group_labels,
        seed=seed + 1,
        train_fraction=train_fraction
    )
    
    num_classes = int(eval_group_labels.max().item()) + 1

    # Train and evaluate classifiers
    results = []
    for name, signal_matrix in signal_sets:
        preds = train_signal_classifier(
            signal_matrix[clf_train_idx],
            eval_group_labels[clf_train_idx],
            signal_matrix[clf_test_idx],
            device=device,
        )
        score = macro_f1(eval_group_labels[clf_test_idx], preds, num_classes=num_classes)
        results.append((name, score))
    
    return results
