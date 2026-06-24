"""Unified results contract for PyTorch validation and Keras paper benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from uqlab.evaluation.signals.catalog import signal_names

SIGNAL_NAMES = signal_names()

SOURCES = ("pytorch_validation", "paper_keras")
SWEEP_TYPES = ("dataset_size", "label_noise")

DISENTANGLEMENT_LABELS = {
    "gaussian_logits": "Gaussian Logits",
    "information_theoretic": "Information Theoretic",
    "pytorch_attribution": "PyTorch Attribution",
}

PAPER_METHODS = ("MC-Dropout", "MC-DropConnect", "Flipout", "Deep Ensemble")
PAPER_DISENTANGLEMENTS = ("gaussian_logits", "information_theoretic")

# Canonical dataset keys. Keep lower_snake_case so they slot directly into
# filenames and the unified schema.  Add new datasets to the ``DATASET_LABELS``
# map as you wire them through the pipelines.
DATASETS = ("cifar10", "fashion_mnist")
DEFAULT_DATASET = "cifar10"
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "fashion_mnist": "Fashion-MNIST",
}

# Map from raw tokens that may appear in legacy paper CSV filenames to the
# canonical dataset key.
_PAPER_DATASET_TOKENS = {
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "fashionmnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
    "fashion_mnist": "fashion_mnist",
    "fashion mnist": "fashion_mnist",
}

UNIFIED_KEY_COLUMNS = (
    "sweep_type",
    "sweep_value",
    "dataset",
    "source",
    "method",
    "disentanglement",
    "architecture",
)

UNIFIED_UNCERTAINTY_COLUMNS = (
    "mean_aleatoric_uncertainty",
    "mean_epistemic_uncertainty",
    "mean_total_uncertainty",
    "accuracy",
)

UNIFIED_COLUMNS = (
    *UNIFIED_KEY_COLUMNS,
    *UNIFIED_UNCERTAINTY_COLUMNS,
    *(
        f"{signal}_{kind}_auroc"
        for signal in SIGNAL_NAMES
        for kind in ("aleatoric", "epistemic")
    ),
)


@dataclass(frozen=True)
class UnifiedRow:
    sweep_type: str
    sweep_value: float
    source: str
    method: str
    disentanglement: str
    architecture: str
    dataset: str = DEFAULT_DATASET
    accuracy: float | None = None
    mean_aleatoric_uncertainty: float | None = None
    mean_epistemic_uncertainty: float | None = None
    mean_total_uncertainty: float | None = None

    def to_dict(self) -> dict:
        base = {f.name: getattr(self, f.name) for f in fields(self)}
        return base


def dataset_label(key: str) -> str:
    """Human-readable label for a dataset key, with a safe title-case fallback."""
    return DATASET_LABELS.get(key, key.replace("_", " ").title())


def _normalize_dataset_token(token: str) -> str:
    """Lookup a raw filename/string token into a canonical dataset key."""
    key = str(token).strip().lower()
    return _PAPER_DATASET_TOKENS.get(key, key)


def _project_root(results_root: Path | None) -> Path:
    if results_root is not None:
        return results_root.parent if results_root.name == "results" else results_root
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "results").is_dir() and (parent / "streamlit_app.py").exists():
            return parent
        if (parent / "results").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


def _validation_metrics_path(results_root: Path, sweep_type: str) -> Path:
    return results_root / "validation" / f"{sweep_type}_sweep" / "metrics.csv"


def _paper_metrics_path(results_root: Path, sweep_type: str) -> Path:
    return results_root / "paper_benchmarks" / sweep_type / "metrics.csv"


def _infer_sweep_type(df: pd.DataFrame, sweep_type: str) -> pd.Series:
    if "sweep_type" in df.columns:
        return df["sweep_type"].fillna(sweep_type)
    return pd.Series(sweep_type, index=df.index)


def _infer_sweep_value(df: pd.DataFrame, sweep_type: str) -> pd.Series:
    if "sweep_value" in df.columns:
        return pd.to_numeric(df["sweep_value"], errors="coerce")
    if sweep_type == "dataset_size":
        for col in ("under_train_per_class", "dataset_size"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
    if sweep_type == "label_noise":
        if "noise_percent" in df.columns:
            return pd.to_numeric(df["noise_percent"], errors="coerce")
        if "noise_rate" in df.columns:
            return pd.to_numeric(df["noise_rate"], errors="coerce") * 100.0
    return pd.Series(pd.NA, index=df.index)


def adapt_pytorch_metrics_csv(df: pd.DataFrame, sweep_type: str) -> pd.DataFrame:
    """Map legacy validation metrics.csv rows to the unified schema."""
    if df.empty:
        return df

    out = df.copy()
    out["source"] = "pytorch_validation"
    out["method"] = out["architecture"] if "architecture" in out.columns else "unknown"
    out["disentanglement"] = "pytorch_attribution"
    out["sweep_type"] = _infer_sweep_type(out, sweep_type)
    out["sweep_value"] = _infer_sweep_value(out, sweep_type)
    if "architecture" not in out.columns:
        out["architecture"] = out["method"]
    # PyTorch pipeline currently only runs on CIFAR-10/CIFAR-10N.  When/if a
    # Fashion-MNIST adapter lands, write the actual dataset into metrics.csv
    # and that value will flow through unchanged here.
    if "dataset" not in out.columns:
        out["dataset"] = DEFAULT_DATASET
    else:
        out["dataset"] = out["dataset"].fillna(DEFAULT_DATASET).map(_normalize_dataset_token)
    return out


def adapt_paper_metrics_csv(df: pd.DataFrame, sweep_type: str) -> pd.DataFrame:
    """Ensure paper benchmark rows use unified column names."""
    if df.empty:
        return df

    out = df.copy()
    out["source"] = out.get("source", pd.Series("paper_keras", index=out.index))
    out["sweep_type"] = _infer_sweep_type(out, sweep_type)
    out["sweep_value"] = _infer_sweep_value(out, sweep_type)
    if "method" not in out.columns and "architecture" in out.columns:
        out["method"] = out["architecture"]
    if "architecture" not in out.columns:
        out["architecture"] = out.get("method", "paper")
    if "dataset" not in out.columns:
        out["dataset"] = DEFAULT_DATASET
    else:
        out["dataset"] = out["dataset"].fillna(DEFAULT_DATASET).map(_normalize_dataset_token)
    return out


def _read_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _align_columns(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(UNIFIED_COLUMNS))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in UNIFIED_COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    return combined


def load_unified_metrics(
    sweep_type: str,
    sources: Sequence[str] = SOURCES,
    results_root: Path | None = None,
    dataset: str | None = None,
) -> pd.DataFrame:
    """
    Load and merge pytorch_validation + optional paper_keras metrics.

    Parameters
    ----------
    sweep_type : str
        ``"dataset_size"`` or ``"label_noise"``.
    sources : tuple/list
        Subset of ``SOURCES`` (default: both).
    results_root : Path | None
        Optional override for the results directory.
    dataset : str | None
        Optional filter on the ``dataset`` key column.  When provided, only
        rows whose dataset matches (after canonicalisation) are returned.
    """
    if sweep_type not in SWEEP_TYPES:
        raise ValueError(f"sweep_type must be one of {SWEEP_TYPES}, got {sweep_type!r}")

    root = _project_root(results_root)
    results = results_root if results_root is not None else root / "results"
    frames: list[pd.DataFrame] = []

    if "pytorch_validation" in sources:
        pytorch_path = _validation_metrics_path(results, sweep_type)
        pytorch_df = _read_metrics_csv(pytorch_path)
        if not pytorch_df.empty:
            frames.append(adapt_pytorch_metrics_csv(pytorch_df, sweep_type))

    if "paper_keras" in sources:
        paper_path = _paper_metrics_path(results, sweep_type)
        paper_df = _read_metrics_csv(paper_path)
        if not paper_df.empty:
            frames.append(adapt_paper_metrics_csv(paper_df, sweep_type))

    # Optional dataset filter applied after schema-alignment.
    if dataset is not None:
        dataset_key = _normalize_dataset_token(dataset)
        frames = [
            f[f["dataset"] == dataset_key].copy() if "dataset" in f.columns else f
            for f in frames
        ]

    combined = _align_columns(frames)
    if combined.empty:
        return combined

    combined = combined.sort_values(
        ["source", "method", "disentanglement", "sweep_value"],
        na_position="last",
    ).reset_index(drop=True)
    return combined


def append_pytorch_row(
    row: UnifiedRow | dict,
    sweep_type: str,
    results_root: Path | None = None,
) -> Path:
    """Append one pytorch_validation row to the sweep metrics.csv."""
    root = _project_root(results_root)
    results = results_root if results_root is not None else root / "results"
    path = _validation_metrics_path(results, sweep_type)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = row if isinstance(row, dict) else row.to_dict()
    record.setdefault("source", "pytorch_validation")
    record.setdefault("disentanglement", "pytorch_attribution")
    record.setdefault("sweep_type", sweep_type)
    record.setdefault("dataset", DEFAULT_DATASET)

    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    else:
        updated = pd.DataFrame([record])
    updated.to_csv(path, index=False)
    return path


def append_paper_row(
    row: UnifiedRow | dict,
    sweep_type: str,
    results_root: Path | None = None,
) -> Path:
    """Append one paper_keras row to results/paper_benchmarks/{sweep}/metrics.csv."""
    root = _project_root(results_root)
    results = results_root if results_root is not None else root / "results"
    path = _paper_metrics_path(results, sweep_type)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = row if isinstance(row, dict) else row.to_dict()
    record.setdefault("source", "paper_keras")
    record.setdefault("dataset", DEFAULT_DATASET)
    record.setdefault("sweep_type", sweep_type)

    if path.exists():
        existing = pd.read_csv(path)
        updated = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    else:
        updated = pd.DataFrame([record])
    updated.to_csv(path, index=False)
    return path


def flatten_paper_result_csvs(
    paper_data_folder: Path,
    sweep_type: str,
    meta_experiment_name: str,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Flatten per-(method × disentanglement) paper CSVs into unified metrics.csv.

    Expects files like:
    {meta}_{dataset}_{uq_name}_{disentanglement_name}_results.csv
    """
    rows: list[dict] = []
    if not paper_data_folder.exists():
        return pd.DataFrame()

    method_map = {
        "mc_dropout": "MC-Dropout",
        "mc_dropconnect": "MC-DropConnect",
        "flipout": "Flipout",
        "deep_ensemble": "Deep Ensemble",
    }

    for csv_path in sorted(paper_data_folder.rglob("*.csv")):
        name = csv_path.stem.lower()
        if "_results" not in name and "results" not in name:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        disentanglement = None
        for key in ("gaussian_logits", "information_theoretic"):
            if key in name:
                disentanglement = key
                break
        if disentanglement is None:
            continue

        method = None
        for token, label in method_map.items():
            if token in name:
                method = label
                break
        if method is None:
            continue

        # Parse dataset token from filename: e.g.
        # "decreasing_dataset_CIFAR10_mc_dropout_gaussian_logits_results.csv".
        dataset = DEFAULT_DATASET
        for token in _PAPER_DATASET_TOKENS:
            if token in name:
                dataset = _PAPER_DATASET_TOKENS[token]
                break

        if df.empty:
            continue

        for idx in range(len(df)):
            row = {col: df[col].iloc[idx] for col in df.columns}
            row.update(
                {
                    "source": "paper_keras",
                    "sweep_type": sweep_type,
                    "method": method,
                    "disentanglement": disentanglement,
                    "architecture": method,
                    "dataset": dataset,
                }
            )
            if "sweep_value" not in row:
                if sweep_type == "dataset_size" and "dataset_size" in row:
                    row["sweep_value"] = row["dataset_size"]
                elif sweep_type == "label_noise":
                    if "noise_percent" in row:
                        row["sweep_value"] = row["noise_percent"]
                    elif "noise_rate" in row:
                        row["sweep_value"] = float(row["noise_rate"]) * 100.0
            rows.append(row)

    unified = adapt_paper_metrics_csv(pd.DataFrame(rows), sweep_type)
    if output_path is not None and not unified.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(output_path, index=False)
    return unified
