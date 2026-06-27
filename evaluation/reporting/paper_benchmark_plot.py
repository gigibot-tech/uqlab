"""
Paper-style benchmark plot payload: three curves vs Percentage (0–1).

Scores, aleatorics, epistemics on a single Y axis — matches ``json_results_to_df`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from uqlab.evaluation.reporting.campaign_score import (
    PaperSweepSeries,
    build_paper_sweep_series,
    build_paper_sweep_series_from_campaign_dir,
    paper_correlations,
)

PAPER_TRACE_COLORS = {
    "scores": "#2ca02c",
    "aleatorics": "#1f77b4",
    "epistemics": "#ff7f0e",
}

PAPER_TRACE_LABELS = {
    "scores": "Accuracy (score)",
    "aleatorics": "Aleatoric (global mean)",
    "epistemics": "Epistemic (global mean)",
}


@dataclass(frozen=True)
class PaperBenchmarkPlotPayload:
    """JSON-serializable paper plot description."""

    experiment: str
    sweep_kind: str
    x_label: str
    y_label: str
    traces: list[dict[str, Any]]
    points: int
    correlations: dict[str, Any]
    run_ids: list[str]
    aleatoric_signal: str
    epistemic_signal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "sweep_kind": self.sweep_kind,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "traces": self.traces,
            "points": self.points,
            "correlations": self.correlations,
            "run_ids": self.run_ids,
            "aleatoric_signal": self.aleatoric_signal,
            "epistemic_signal": self.epistemic_signal,
        }


def payload_from_series(
    series: PaperSweepSeries,
    *,
    aleatoric_signal: str,
    epistemic_signal: str,
    rank_correlation: bool = False,
) -> PaperBenchmarkPlotPayload:
    """Build plot payload from an already-collated ``PaperSweepSeries``."""
    wide = series.wide_dataframe()
    x_vals = [float(v) for v in wide["Percentage"].tolist()]

    traces: list[dict[str, Any]] = []
    for metric in ("scores", "aleatorics", "epistemics"):
        y_vals = [None if pd.isna(v) else float(v) for v in wide[metric].tolist()]
        traces.append(
            {
                "name": PAPER_TRACE_LABELS[metric],
                "metric": metric,
                "x": x_vals,
                "y": y_vals,
                "color": PAPER_TRACE_COLORS[metric],
                "dash": "solid",
            }
        )

    return PaperBenchmarkPlotPayload(
        experiment=series.experiment,
        sweep_kind=series.sweep_kind,
        x_label="Percentage (0–1)",
        y_label="Score / global uncertainty mean",
        traces=traces,
        points=len(wide),
        correlations=paper_correlations(series, rank=rank_correlation),
        run_ids=series.run_ids,
        aleatoric_signal=aleatoric_signal,
        epistemic_signal=epistemic_signal,
    )


def build_paper_benchmark_plot(
    run_ids: list[str],
    experiments_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
    rank_correlation: bool = False,
) -> PaperBenchmarkPlotPayload:
    """Build three-line paper plot from Streamlit ``experiments_dir/<run_id>/results/``."""
    from uqlab.evaluation.benchmarks.disentangling.experiment import ExperimentDisentanglingModel

    defaults = ExperimentDisentanglingModel()
    alea_sig = aleatoric_signal or defaults.aleatoric_signal
    epi_sig = epistemic_signal or defaults.epistemic_signal

    series = build_paper_sweep_series(
        run_ids,
        experiments_dir,
        sweep_kind=sweep_kind,
        aleatoric_signal=alea_sig,
        epistemic_signal=epi_sig,
    )
    return payload_from_series(
        series,
        aleatoric_signal=alea_sig,
        epistemic_signal=epi_sig,
        rank_correlation=rank_correlation,
    )


def build_paper_benchmark_plot_from_campaign_dir(
    campaign_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
    rank_correlation: bool = False,
) -> PaperBenchmarkPlotPayload:
    """Build plot from a flat campaign folder (validation sweeps)."""
    from uqlab.evaluation.benchmarks.disentangling.experiment import ExperimentDisentanglingModel

    defaults = ExperimentDisentanglingModel()
    alea_sig = aleatoric_signal or defaults.aleatoric_signal
    epi_sig = epistemic_signal or defaults.epistemic_signal

    series = build_paper_sweep_series_from_campaign_dir(
        campaign_dir,
        sweep_kind=sweep_kind,
        aleatoric_signal=alea_sig,
        epistemic_signal=epi_sig,
    )
    return payload_from_series(
        series,
        aleatoric_signal=alea_sig,
        epistemic_signal=epi_sig,
        rank_correlation=rank_correlation,
    )


def save_paper_benchmark_png(payload: PaperBenchmarkPlotPayload, path: Path, *, dpi: int = 150) -> Path:
    """Save three-line paper plot PNG (notebook ``df.plot()`` equivalent). Disk only."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for save_paper_benchmark_png") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x_vals = payload.traces[0]["x"] if payload.traces else []

    fig, ax = plt.subplots(figsize=(8, 5))
    for trace in payload.traces:
        ax.plot(
            x_vals,
            trace["y"],
            marker="o",
            label=trace["name"],
            color=trace.get("color"),
        )
    ax.set_xlabel(payload.x_label)
    ax.set_ylabel(payload.y_label)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.set_title(payload.experiment)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def persist_campaign_paper_plot(
    campaign_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
    rank_correlation: bool = False,
) -> dict[str, Path]:
    """
    Campaign end: PNG + CSV (``json_results_to_df`` wide form).

    Writes ``{sweep_kind}_three_line.png`` and ``{sweep_kind}_curves.csv``.
    """
    campaign_dir = Path(campaign_dir)
    payload = build_paper_benchmark_plot_from_campaign_dir(
        campaign_dir,
        sweep_kind=sweep_kind,
        aleatoric_signal=aleatoric_signal,
        epistemic_signal=epistemic_signal,
        rank_correlation=rank_correlation,
    )

    png_path = campaign_dir / f"{sweep_kind}_three_line.png"
    save_paper_benchmark_png(payload, png_path)

    wide_rows: list[dict[str, Any]] = []
    x_vals = payload.traces[0]["x"] if payload.traces else []
    metric_by_name = {t["metric"]: t["y"] for t in payload.traces}
    for idx, pct in enumerate(x_vals):
        wide_rows.append(
            {
                "Experiment": payload.experiment,
                "Percentage": pct,
                "scores": metric_by_name.get("scores", [None] * len(x_vals))[idx],
                "aleatorics": metric_by_name.get("aleatorics", [None] * len(x_vals))[idx],
                "epistemics": metric_by_name.get("epistemics", [None] * len(x_vals))[idx],
                "run_id": payload.run_ids[idx] if idx < len(payload.run_ids) else None,
            }
        )
    csv_path = campaign_dir / f"{sweep_kind}_curves.csv"
    pd.DataFrame(wide_rows).to_csv(csv_path, index=False)

    return {"png": png_path, "csv": csv_path}


def paper_dataframe_from_campaign_dir(
    campaign_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
) -> pd.DataFrame:
    """
    Wide table matching vendor ``json_results_to_df`` (scores / aleatorics / epistemics columns).

    Use with the paper one-liner::

        df.drop("Run_Index", axis=1).groupby(["Experiment", "Percentage"]).mean().groupby("Experiment").plot()
    """
    series = build_paper_sweep_series_from_campaign_dir(
        campaign_dir,
        sweep_kind=sweep_kind,
        aleatoric_signal=aleatoric_signal,
        epistemic_signal=epistemic_signal,
    )
    wide = series.wide_dataframe()
    return wide.assign(Run_Index=0)[
        ["Experiment", "Percentage", "scores", "aleatorics", "epistemics", "Run_Index"]
    ]
