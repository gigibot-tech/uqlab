"""
Aggregate uqlab campaign runs into paper ``ExperimentResults`` semantics.

Paper metric (``disentanglement_error``): per sweep point store ``scores`` (accuracy),
``aleatorics`` / ``epistemics`` as **global means** over all eval samples from
``predict_disentangling``, with X = ``Percentage`` (0–1 fraction).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from uqlab.evaluation.reporting.sweep_line_plot import (
    SWEEP_KIND_DATASET_SIZE,
    SWEEP_KIND_LABEL_NOISE,
    build_sweep_metrics_frame,
)
from uqlab.evaluation.benchmarks.disentangling.experiment import ExperimentDisentanglingModel
from uqlab.run_artifacts import metrics_row_from_run
from uqlab.vendor.disentanglement_error.util import ExperimentResults

PAPER_EXPERIMENT_LABELS = {
    SWEEP_KIND_LABEL_NOISE: "Label Noise",
    SWEEP_KIND_DATASET_SIZE: "Decreasing Dataset",
}


@dataclass(frozen=True)
class PaperSweepSeries:
    """One experiment arm in paper plot form."""

    experiment: str
    sweep_kind: str
    percentages: list[float]
    results: ExperimentResults
    run_ids: list[str]

    def to_dataframe(self) -> pd.DataFrame:
        """Long format — one row per (sweep point, metric); use ``wide_dataframe`` for plotting."""
        rows: list[dict[str, Any]] = []
        for idx, pct in enumerate(self.percentages):
            for metric, values in (
                ("scores", self.results.scores),
                ("aleatorics", self.results.aleatorics),
                ("epistemics", self.results.epistemics),
            ):
                if idx >= len(values):
                    continue
                rows.append(
                    {
                        "Experiment": self.experiment,
                        "Percentage": pct,
                        "metric": metric,
                        "value": float(values[idx]),
                        "Run_Index": 0,
                        "run_id": self.run_ids[idx] if idx < len(self.run_ids) else None,
                    }
                )
        return pd.DataFrame(rows)

    def wide_dataframe(self) -> pd.DataFrame:
        """One row per sweep point — convenient for Plotly."""
        return pd.DataFrame(
            {
                "Experiment": self.experiment,
                "Percentage": self.percentages,
                "scores": self.results.scores,
                "aleatorics": self.results.aleatorics,
                "epistemics": self.results.epistemics,
                "run_id": self.run_ids,
            }
        )


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _regular_train_per_class(cfg: dict[str, Any], *, default: int = 300) -> int:
    data = cfg.get("data") or {}
    unc = cfg.get("uncertainty") or cfg.get("uncertainty_config") or {}
    for source in (data, unc, cfg):
        val = source.get("regular_train_per_class")
        if val is not None:
            return max(1, int(val))
    return default


def percentage_from_run_row(
    row: dict[str, Any],
    sweep_kind: str,
    *,
    regular_train_per_class: int = 300,
) -> float | None:
    """Map uqlab sweep keys to paper ``Percentage`` (0–1)."""
    if sweep_kind == SWEEP_KIND_LABEL_NOISE:
        noise = row.get("noise_percent")
        if noise is None:
            return None
        return float(noise) / 100.0

    under = row.get("under_train_per_class")
    if under is None:
        return None
    return float(under) / float(regular_train_per_class)


def discover_campaign_run_dirs(campaign_dir: Path) -> list[tuple[str, Path]]:
    """
    List ``(run_id, results_dir)`` under a flat campaign folder.

    Supports validation layout (``<campaign>/<exp>/results.pt``) and
    Streamlit layout (``<campaign>/<run_id>/results/results.pt``).
    """
    campaign_dir = Path(campaign_dir)
    if not campaign_dir.is_dir():
        return []

    found: list[tuple[str, Path]] = []
    for sub in sorted(campaign_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        if (sub / "results.pt").is_file() or (sub / "summary.json").is_file():
            found.append((sub.name, sub))
            continue
        nested = sub / "results"
        if (nested / "results.pt").is_file() or (nested / "summary.json").is_file():
            found.append((sub.name, nested))
    return found


def _sweep_row_from_results_dir(results_dir: Path, sweep_kind: str) -> dict[str, Any]:
    """Metrics row plus sweep-axis columns from ``summary.json`` when needed."""
    import json

    row = dict(metrics_row_from_run(results_dir))
    summary_path = results_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        data = (summary.get("config") or {}).get("data") or {}
        if sweep_kind == SWEEP_KIND_LABEL_NOISE:
            noise = data.get("aleatoric_noise_percentage")
            if noise is not None and "noise_percent" not in row:
                val = float(noise)
                row["noise_percent"] = val * 100.0 if val <= 1.0 else val
        else:
            under = data.get("under_train_per_class")
            if under is not None and "under_train_per_class" not in row:
                row["under_train_per_class"] = int(under)

    folder = results_dir.name if results_dir.name != "results" else results_dir.parent.name
    if sweep_kind == SWEEP_KIND_LABEL_NOISE and "noise_percent" not in row and "_noise" in folder:
        try:
            row["noise_percent"] = float(folder.rpartition("_noise")[2])
        except ValueError:
            pass
    if sweep_kind == SWEEP_KIND_DATASET_SIZE and "under_train_per_class" not in row:
        for sep in ("_under", "_size"):
            if sep in folder:
                try:
                    row["under_train_per_class"] = int(folder.rpartition(sep)[2])
                except ValueError:
                    pass
                break
    return row


def build_paper_sweep_series_from_campaign_dir(
    campaign_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
) -> PaperSweepSeries:
    """
    Collate sweep points from a flat campaign directory (validation or CLI sweeps).

    Same semantics as ``json_results_to_df`` after aggregation.
    """
    if sweep_kind not in (SWEEP_KIND_LABEL_NOISE, SWEEP_KIND_DATASET_SIZE):
        raise ValueError(f"Unsupported sweep_kind {sweep_kind!r}")

    points: list[tuple[float, str, dict[str, float]]] = []
    for run_id, results_dir in discover_campaign_run_dirs(campaign_dir):
        paper_vals = paper_point_from_results_dir(
            results_dir,
            aleatoric_signal=aleatoric_signal,
            epistemic_signal=epistemic_signal,
        )
        if paper_vals is None:
            continue

        row = _sweep_row_from_results_dir(results_dir, sweep_kind)
        cfg = _load_run_config(results_dir.parent if results_dir.name == "results" else results_dir)
        regular = _regular_train_per_class(cfg)
        pct = percentage_from_run_row(row, sweep_kind, regular_train_per_class=regular)
        if pct is None:
            continue
        points.append((pct, run_id, paper_vals))

    if len(points) < 2:
        raise ValueError(
            f"Need at least 2 completed runs under {campaign_dir} for a paper plot "
            f"(found {len(points)} with accuracy + global signal means)."
        )

    points.sort(key=lambda item: item[0])
    return PaperSweepSeries(
        experiment=PAPER_EXPERIMENT_LABELS[sweep_kind],
        sweep_kind=sweep_kind,
        percentages=[p[0] for p in points],
        results=ExperimentResults(
            scores=[p[2]["score"] for p in points],
            aleatorics=[p[2]["aleatorics"] for p in points],
            epistemics=[p[2]["epistemics"] for p in points],
        ),
        run_ids=[p[1] for p in points],
    )


def paper_point_from_results_dir(
    results_dir: Path,
    *,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
) -> dict[str, float] | None:
    """
    One sweep point: global score / aleatoric / epistemic means (all eval samples).

    Uses the same signal pairing as ``ExperimentDisentanglingModel`` by default.
    """
    defaults = ExperimentDisentanglingModel()
    alea_sig = aleatoric_signal or defaults.aleatoric_signal
    epi_sig = epistemic_signal or defaults.epistemic_signal

    metrics = metrics_row_from_run(results_dir)
    score = metrics.get("accuracy")
    alea = metrics.get(f"{alea_sig}_mean")
    epi = metrics.get(f"{epi_sig}_mean")

    if score is None or alea is None or epi is None:
        return None

    return {
        "score": float(score),
        "aleatorics": float(alea),
        "epistemics": float(epi),
    }


def build_paper_sweep_series(
    run_ids: list[str],
    experiments_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
) -> PaperSweepSeries:
    """
    Collate completed campaign runs into one ``ExperimentResults`` curve.

    Raises ``ValueError`` when no run yields a complete paper point.
    """
    if sweep_kind not in (SWEEP_KIND_LABEL_NOISE, SWEEP_KIND_DATASET_SIZE):
        raise ValueError(f"Unsupported sweep_kind {sweep_kind!r}")

    metrics_df = build_sweep_metrics_frame(run_ids, experiments_dir)
    if metrics_df.empty:
        raise ValueError("No completed runs with metrics in this sweep group")

    points: list[tuple[float, str, dict[str, float]]] = []
    for run_id in run_ids:
        results_dir = experiments_dir / run_id / "results"
        paper_vals = paper_point_from_results_dir(
            results_dir,
            aleatoric_signal=aleatoric_signal,
            epistemic_signal=epistemic_signal,
        )
        if paper_vals is None:
            continue

        row = metrics_df[metrics_df["run_id"] == run_id]
        cfg = _load_run_config(experiments_dir / run_id)
        regular = _regular_train_per_class(cfg)
        row_dict = row.iloc[0].to_dict() if not row.empty else {}
        pct = percentage_from_run_row(row_dict, sweep_kind, regular_train_per_class=regular)
        if pct is None:
            continue
        points.append((pct, run_id, paper_vals))

    if len(points) < 2:
        raise ValueError(
            "Need at least 2 runs with accuracy and global signal means in results.pt "
            "for a paper-style plot."
        )

    points.sort(key=lambda item: item[0])
    percentages = [p[0] for p in points]
    run_ids_ordered = [p[1] for p in points]
    results = ExperimentResults(
        scores=[p[2]["score"] for p in points],
        aleatorics=[p[2]["aleatorics"] for p in points],
        epistemics=[p[2]["epistemics"] for p in points],
    )

    return PaperSweepSeries(
        experiment=PAPER_EXPERIMENT_LABELS[sweep_kind],
        sweep_kind=sweep_kind,
        percentages=percentages,
        results=results,
        run_ids=run_ids_ordered,
    )


def paper_correlations(
    series: PaperSweepSeries,
    *,
    rank: bool = False,
) -> dict[str, float | None]:
    """
    Paper-relevant correlations per arm.

    Label noise → aleatorics vs scores; decreasing dataset → epistemics vs scores.
    Also returns the off-diagonal pair for context.
    """
    corr_fn = spearmanr if rank else pearsonr
    scores = series.results.scores
    alea_corr: float | None = None
    epi_corr: float | None = None

    if len(scores) >= 2 and len(series.results.aleatorics) == len(scores):
        alea_corr = float(corr_fn(series.results.aleatorics, scores).statistic)
    if len(scores) >= 2 and len(series.results.epistemics) == len(scores):
        epi_corr = float(corr_fn(series.results.epistemics, scores).statistic)

    if series.sweep_kind == SWEEP_KIND_LABEL_NOISE:
        primary = {"metric": "aleatorics", "correlation": alea_corr}
        secondary = {"metric": "epistemics", "correlation": epi_corr}
    else:
        primary = {"metric": "epistemics", "correlation": epi_corr}
        secondary = {"metric": "aleatorics", "correlation": alea_corr}

    return {
        "primary_metric": primary["metric"],
        "primary_correlation": primary["correlation"],
        "secondary_metric": secondary["metric"],
        "secondary_correlation": secondary["correlation"],
        "aleatoric_correlation": alea_corr,
        "epistemic_correlation": epi_corr,
        "rank": rank,
    }
