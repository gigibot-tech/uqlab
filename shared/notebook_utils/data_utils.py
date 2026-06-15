"""Data loading and processing utilities for validation notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import SWEEP_TO_X


@dataclass(frozen=True)
class SweepLoadResult:
    """Loaded sweep data plus human-readable warnings."""

    df: pd.DataFrame
    warnings: list[str]


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from an arbitrary working directory."""
    search_start = (start or Path.cwd()).resolve()
    candidates = [search_start, *search_start.parents, Path(__file__).resolve().parents[3]]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "scripts" / "run_validation_experiments.py").exists():
            return candidate

    raise FileNotFoundError("Could not locate project root from notebook context.")


def validation_dir(project_root: Path | None = None) -> Path:
    """Return the validation results directory."""
    return find_project_root(project_root) / "results" / "validation"


def run_validation_experiments(
    *,
    project_root: Path | None = None,
    sweep: str = "both",
    mode: str = "full",
    output_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the central validation pipeline from notebooks."""
    root = find_project_root(project_root)
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_validation_experiments.py"),
        "--mode",
        mode,
        "--sweep",
        sweep,
    ]
    if output_dir is not None:
        cmd.extend(["--output_dir", str(output_dir)])

    return subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _rank_correlation(x: pd.Series, y: pd.Series) -> float:
    """Compute a Spearman-like rank correlation without SciPy."""
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(joined) < 2:
        return float("nan")
    return float(joined["x"].rank(method="average").corr(joined["y"].rank(method="average")))


def load_sweep_metrics(
    sweep_name: str,
    *,
    project_root: Path | None = None,
) -> SweepLoadResult:
    """Load one sweep's metrics.csv and normalize its columns."""
    if sweep_name not in SWEEP_TO_X:
        raise ValueError(f"Unknown sweep_name={sweep_name!r}")

    root = find_project_root(project_root)
    csv_path = validation_dir(root) / f"{sweep_name}_sweep" / "metrics.csv"
    warnings: list[str] = []

    if not csv_path.exists():
        warnings.append(f"Missing results file: {csv_path}")
        return SweepLoadResult(pd.DataFrame(), warnings)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        warnings.append(f"Results file is empty: {csv_path}")
        return SweepLoadResult(pd.DataFrame(), warnings)

    if df.empty:
        warnings.append(f"No rows found in {csv_path}")
        return SweepLoadResult(df, warnings)

    df = df.copy()
    df["sweep_type"] = sweep_name

    if sweep_name == "label_noise":
        if "noise_percent" not in df.columns and "noise_rate" in df.columns:
            if float(df["noise_rate"].max()) <= 1.0:
                df["noise_percent"] = df["noise_rate"] * 100.0
                warnings.append(
                    "Legacy label-noise results detected: only `noise_rate` in [0, 1] was found. "
                    "If these rows came from the old runner, the injected noise may have been off by 100x."
                )
            else:
                df["noise_percent"] = df["noise_rate"]
        if "noise_rate" not in df.columns and "noise_percent" in df.columns:
            df["noise_rate"] = df["noise_percent"] / 100.0

    return SweepLoadResult(df, warnings)


def summarize_trends(df: pd.DataFrame, sweep_name: str) -> pd.DataFrame:
    """Summarize monotonic trends expected from the paper."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "architecture",
                "metric",
                "expected_direction",
                "rank_correlation",
                "passed",
            ]
        )

    x_col = SWEEP_TO_X[sweep_name]
    expectations: list[tuple[str, str]] = []
    if sweep_name == "dataset_size":
        expectations = [
            ("accuracy", "increase"),
            ("mean_epistemic_uncertainty", "decrease"),
            ("mean_aleatoric_uncertainty", "stable_or_small"),
            ("mean_total_uncertainty", "decrease"),
        ]
    elif sweep_name == "label_noise":
        expectations = [
            ("accuracy", "decrease"),
            ("mean_aleatoric_uncertainty", "increase"),
            ("mean_epistemic_uncertainty", "mixed"),
            ("mean_total_uncertainty", "increase"),
        ]

    rows: list[dict[str, object]] = []
    for architecture in sorted(df["architecture"].dropna().unique()):
        arch_df = df[df["architecture"] == architecture].sort_values(x_col)
        for metric, expected in expectations:
            if metric not in arch_df.columns:
                continue
            corr = _rank_correlation(arch_df[x_col], arch_df[metric])
            passed = None
            if expected == "increase":
                passed = bool(corr > 0)
            elif expected == "decrease":
                passed = bool(corr < 0)
            elif expected == "stable_or_small":
                passed = bool(abs(corr) < 0.5)

            rows.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "expected_direction": expected,
                    "rank_correlation": corr,
                    "passed": passed,
                }
            )

    return pd.DataFrame(rows)


def validate_decomposition(df: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """Check that total uncertainty equals epistemic plus aleatoric."""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    required = [
        "architecture",
        "mean_total_uncertainty",
        "mean_epistemic_uncertainty",
        "mean_aleatoric_uncertainty",
    ]
    available = [col for col in required if col in df.columns]
    if len(available) != len(required):
        return pd.DataFrame()

    for _, row in df.iterrows():
        total = float(row["mean_total_uncertainty"])
        epistemic = float(row["mean_epistemic_uncertainty"])
        aleatoric = float(row["mean_aleatoric_uncertainty"])
        expected_total = epistemic + aleatoric
        error = abs(total - expected_total)
        rows.append(
            {
                "architecture": row["architecture"],
                "sweep_type": row.get("sweep_type", "unknown"),
                "total_measured": total,
                "total_expected": expected_total,
                "absolute_error": error,
                "passed": bool(error <= tolerance),
            }
        )

    return pd.DataFrame(rows)


def validate_basic_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Validate non-negativity and finite bounds."""
    if df.empty:
        return pd.DataFrame(columns=["metric", "check", "violations", "passed"])

    checks: list[dict[str, object]] = []
    for metric in [
        "accuracy",
        "mean_epistemic_uncertainty",
        "mean_aleatoric_uncertainty",
        "mean_total_uncertainty",
    ]:
        if metric not in df.columns:
            continue

        series = df[metric]
        if metric == "accuracy":
            violations = int(((series < 0) | (series > 1) | series.isna() | np.isinf(series)).sum())
            checks.append(
                {
                    "metric": metric,
                    "check": "0 <= accuracy <= 1 and finite",
                    "violations": violations,
                    "passed": violations == 0,
                }
            )
        else:
            violations = int(((series < 0) | series.isna() | np.isinf(series)).sum())
            checks.append(
                {
                    "metric": metric,
                    "check": "uncertainty >= 0 and finite",
                    "violations": violations,
                    "passed": violations == 0,
                }
            )

    return pd.DataFrame(checks)


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return only the requested columns that exist."""
    return df[[column for column in columns if column in df.columns]].copy()

# Made with Bob
