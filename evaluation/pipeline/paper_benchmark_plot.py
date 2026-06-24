"""
Paper-style benchmark plot payload: three curves vs Percentage (0–1).

Scores, aleatorics, epistemics on a single Y axis — matches ``json_results_to_df`` semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from uqlab.evaluation.pipeline.campaign_score import (
    build_paper_sweep_series,
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


def build_paper_benchmark_plot(
    run_ids: list[str],
    experiments_dir: Path,
    *,
    sweep_kind: str,
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
    rank_correlation: bool = False,
) -> PaperBenchmarkPlotPayload:
    """Build three-line paper plot from on-disk campaign runs."""
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
        aleatoric_signal=alea_sig,
        epistemic_signal=epi_sig,
    )
