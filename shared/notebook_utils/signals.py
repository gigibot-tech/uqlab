"""Uncertainty signal names and Row-3 selection for method-comparison plots."""

from __future__ import annotations

import pandas as pd

from uqlab.evaluation.signals.catalog import (
    aleatoric_signal_ids,
    epistemic_signal_ids,
    signal_labels,
    signal_names,
    step4_signal_groups,
)
from uqlab.shared.config.signals import (
    PLOT_DEFAULT_ALEATORIC_SIGNAL,
    PLOT_DEFAULT_EPISTEMIC_SIGNAL,
    signal_id_column_candidates,
)

SIGNAL_NAMES = signal_names()

# One subplot per type in sweep figures (canonical registry signals only).
SWEEP_SUBPLOT_SIGNALS: tuple[str, ...] = tuple(SIGNAL_NAMES)

# Row 3: rank these candidates by mean AUROC (sweep-appropriate type).
ROW3_CANDIDATE_SIGNALS = list(SIGNAL_NAMES)

# Logit-based epistemic signals (respond to dataset size, stable to noise)
EPISTEMIC_SIGNALS = epistemic_signal_ids()

# Attribution-based aleatoric signal (responds to noise, stable to dataset size)
ALEATORIC_SIGNALS = aleatoric_signal_ids()

# Row 3: fixed trio (DualXDA) + best baseline by sweep AUROC; legacy alias columns still match.
ROW3_FIXED_SIGNALS = [
    PLOT_DEFAULT_ALEATORIC_SIGNAL,
    "inverse_dominance_dualxda",
    PLOT_DEFAULT_EPISTEMIC_SIGNAL,
]
ROW3_FOURTH_CANDIDATES = [
    sid
    for sid in SIGNAL_NAMES
    if sid not in ROW3_FIXED_SIGNALS
]

SIGNAL_LABELS = signal_labels()

# Paper method column order (fuzzy-matched against df["architecture"]).
PAPER_METHOD_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("MC-Dropout", ("mc-dropout", "mc dropout", "mcdropout", "mc_dropout")),
    ("MC-DropConnect", ("mc-dropconnect", "dropconnect", "mcdropconnect", "mc_dropconnect")),
    ("Flipout", ("flipout",)),
    ("Deep Ensemble", ("deep ensemble", "deep-ensemble", "deepensembles", "deep_ensemble")),
]


def sweep_to_auroc_type(sweep_type: str) -> str:
    """Epistemic AUROC for dataset_size sweep; aleatoric for label_noise."""
    return "epistemic" if sweep_type == "dataset_size" else "aleatoric"


def resolve_x_col(df: pd.DataFrame, sweep_type: str) -> str:
    """Pick the x-axis column present in *df* for this sweep."""
    from .constants import SWEEP_TO_X

    preferred = SWEEP_TO_X.get(sweep_type)
    if preferred and preferred in df.columns:
        return preferred
    fallbacks = {
        "dataset_size": ["under_train_per_class", "sweep_value", "dataset_size"],
        "label_noise": ["noise_percent", "noise_rate", "sweep_value"],
    }
    for col in fallbacks.get(sweep_type, []):
        if col in df.columns:
            return col
    raise ValueError(f"No x-axis column for sweep_type={sweep_type!r} in dataframe")


def aggregate_by_sweep(df: pd.DataFrame, x_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Mean-aggregate metrics across architectures at each sweep point."""
    available = [c for c in value_cols if c in df.columns]
    if not available or x_col not in df.columns:
        return pd.DataFrame()
    agg = df.groupby(x_col, as_index=False)[available].mean()
    return agg.sort_values(x_col)


def get_top_n_signals(
    df: pd.DataFrame, n: int = 4, signal_type: str = "epistemic"
) -> list[tuple[str, float]]:
    """
    Top *n* signals by mean AUROC among ROW3_CANDIDATE_SIGNALS.

    Use ``signal_type='epistemic'`` for dataset_size sweeps and ``'aleatoric'``
    for label_noise (see ``sweep_to_auroc_type``).
    """
    auroc_suffix = f"_{signal_type}_auroc"
    signal_aurocs: list[tuple[str, float]] = []

    for signal in ROW3_CANDIDATE_SIGNALS:
        col = f"{signal}{auroc_suffix}"
        if col in df.columns:
            mean_auroc = df[col].mean()
            if pd.notna(mean_auroc):
                signal_aurocs.append((signal, float(mean_auroc)))

    signal_aurocs.sort(key=lambda x: x[1], reverse=True)
    return signal_aurocs[:n]


def get_row3_signals(df: pd.DataFrame, sweep_type: str) -> list[tuple[str, float]]:
    """Row 3 columns: top 4 candidate signals by sweep-appropriate mean AUROC."""
    return get_top_n_signals(df, n=4, signal_type=sweep_to_auroc_type(sweep_type))


def row3_signal_names_for_plots(
    df: pd.DataFrame,
    sweep_type: str,
    *,
    n: int = 4,
) -> list[str]:
    """
    Paper Row 3 layout: fixed attribution trio + best baseline by sweep AUROC.

    Plotting uses ``<signal>_mean`` pooled over **all** eval samples (not per pack).
    """
    ranked = get_row3_signals(df, sweep_type=sweep_type)
    ordered: list[str] = []
    for sig in ROW3_FIXED_SIGNALS:
        for candidate in signal_id_column_candidates(sig):
            col = f"{candidate}_mean"
            if col in df.columns and sig not in ordered:
                ordered.append(sig)
                break
    for sig, _ in ranked:
        if sig not in ordered:
            ordered.append(sig)
        if len(ordered) >= n:
            break
    return ordered[:n]


def _fuzzy_match_architecture(architectures: list[str], patterns: tuple[str, ...]) -> str | None:
    for arch in architectures:
        al = str(arch).lower().replace("_", " ")
        if any(p in al for p in patterns):
            return arch
    return None


def architecture_columns(df: pd.DataFrame, n: int = 4) -> list[str | None]:
    """
    Up to *n* architecture names for rows 1–2.

    Prefer paper methods (MC-Dropout, MC-DropConnect, Flipout, Deep Ensemble)
    via fuzzy match on ``architecture``; otherwise first *n* sorted unique values.
    """
    if "architecture" not in df.columns:
        return [None] * n

    architectures = sorted(a for a in df["architecture"].dropna().unique())
    matched: list[str] = []
    used: set[str] = set()

    for _label, patterns in PAPER_METHOD_PATTERNS:
        if len(matched) >= n:
            break
        arch = _fuzzy_match_architecture([a for a in architectures if a not in used], patterns)
        if arch is not None:
            matched.append(arch)
            used.add(arch)

    for arch in architectures:
        if len(matched) >= n:
            break
        if arch not in used:
            matched.append(arch)
            used.add(arch)

    return (matched + [None] * n)[:n]


def get_method_architecture_mapping(df: pd.DataFrame, n: int = 4) -> dict:
    """
    Return a stable mapping of paper methods → architecture strings present in df.

    Output shape:
      {
        "ordered_methods": [("MC-Dropout", "cnn_mcdropout"), ... up to n],
        "used_fuzzy_matching": bool,
        "unmapped_architectures": [ ... ],
      }
    """
    if "architecture" not in df.columns or df.empty:
        return {
            "ordered_methods": [(label, None) for label, _ in PAPER_METHOD_PATTERNS[:n]],
            "used_fuzzy_matching": False,
            "unmapped_architectures": [],
        }

    architectures = sorted(a for a in df["architecture"].dropna().unique())
    used: set[str] = set()
    ordered: list[tuple[str, str | None]] = []
    used_fuzzy = False

    for label, patterns in PAPER_METHOD_PATTERNS:
        if len(ordered) >= n:
            break
        arch = _fuzzy_match_architecture([a for a in architectures if a not in used], patterns)
        if arch is not None:
            used_fuzzy = True
            ordered.append((label, arch))
            used.add(arch)
        else:
            ordered.append((label, None))

    # Fill remaining columns (if any) with leftover architectures.
    for arch in architectures:
        if len(ordered) >= n:
            break
        if arch not in used:
            ordered.append(("Other", arch))
            used.add(arch)

    unmapped = [a for a in architectures if a not in used]

    return {
        "ordered_methods": ordered[:n],
        "used_fuzzy_matching": used_fuzzy,
        "unmapped_architectures": unmapped,
    }


def format_method_architecture_mapping(mapping: dict) -> str:
    """Human-readable summary of `get_method_architecture_mapping` output."""
    lines = ["Method ↔ architecture mapping (rows 1-2 of paper layout):"]
    for label, arch in mapping["ordered_methods"]:
        if arch is None:
            lines.append(f"  • {label:<15} → (no match in metrics.csv)")
        else:
            lines.append(f"  • {label:<15} → {arch}")
    if mapping["unmapped_architectures"]:
        lines.append("")
        lines.append("Architectures present in data but not shown in the 3×4 plot:")
        for arch in mapping["unmapped_architectures"]:
            lines.append(f"  • {arch}")
    if mapping["used_fuzzy_matching"]:
        lines.append("")
        lines.append("(Mapping resolved by fuzzy matching against paper labels.)")
    return "\n".join(lines)


def print_method_architecture_mapping(df: pd.DataFrame, n: int = 4) -> dict:
    """Print and return the resolved paper-method ↔ architecture mapping."""
    mapping = get_method_architecture_mapping(df, n=n)
    print(format_method_architecture_mapping(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Data-present helpers (used by Streamlit controls and figure builders)
# ---------------------------------------------------------------------------

# Canonical order for disentanglement rows.
DISENTANGLEMENT_ORDER = (
    "gaussian_logits",
    "information_theoretic",
    "pytorch_attribution",
)

DISENTANGLEMENT_LABELS = {
    "gaussian_logits": "Gaussian Logits",
    "information_theoretic": "Information Theoretic",
    "pytorch_attribution": "PyTorch Attribution",
}


def present_architectures(df: pd.DataFrame) -> list[str]:
    """Sorted unique architecture names in *df*."""
    if "architecture" not in df.columns or df.empty:
        return []
    return sorted(a for a in df["architecture"].dropna().unique())


def present_datasets(df: pd.DataFrame) -> list[str]:
    """Sorted unique dataset keys actually present in *df*."""
    if "dataset" not in df.columns or df.empty:
        return []
    return sorted(d for d in df["dataset"].dropna().unique())


def present_sources(df: pd.DataFrame) -> list[str]:
    """Sorted unique source keys (``pytorch_validation`` / ``paper_keras``) in *df*."""
    if "source" not in df.columns or df.empty:
        return []
    return sorted(s for s in df["source"].dropna().unique())


def present_disentanglements(df: pd.DataFrame) -> list[str]:
    """
    Ordered list of disentanglement types actually in *df*, restricted to the
    canonical set ``DISENTANGLEMENT_ORDER``.  Returns the values in the same
    order as ``DISENTANGLEMENT_ORDER`` so figure rows are stable.
    """
    if "disentanglement" not in df.columns or df.empty:
        return []
    present = set(df["disentanglement"].dropna().unique())
    return [d for d in DISENTANGLEMENT_ORDER if d in present]


def disentanglement_label(key: str) -> str:
    """Human-readable label for a disentanglement key, with a safe fallback."""
    return DISENTANGLEMENT_LABELS.get(key, key.replace("_", " ").title())
