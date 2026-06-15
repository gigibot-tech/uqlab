"""
Shared utilities for Risk–Coverage (AURC) experiments.

Goal: keep AURC computation + artifacts consistent across scripts:
  - experiments/risk_coverage/run_aurc.py (MC Dropout + Surgical)
  - experiments/risk_coverage/run_aurc_msp_baselines.py (classic MSP baselines)

This module centralizes:
  - AURC (+ optional E-AURC) computation
  - risk–coverage CSV writing
  - plotting to PDF/PNG
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from uqlab.legacy_metrics.standard_uq_metrics import StandardUQMetrics


@dataclass(frozen=True)
class RiskCoverageArtifacts:
    curve_csv: str
    pdf: str
    png: str


def _write_curve_csv(path: Path, coverage: np.ndarray, risks: Dict[str, np.ndarray]) -> None:
    methods = list(risks.keys())
    with open(path, "w") as f:
        f.write("coverage," + ",".join([f"risk_{m}" for m in methods]) + "\n")
        for i in range(len(coverage)):
            f.write(str(float(coverage[i])))
            for m in methods:
                f.write("," + str(float(risks[m][i])))
            f.write("\n")


def _default_style(name: str) -> Dict:
    if name in ("surgical",):
        return {"color": "tab:blue", "linestyle": "-", "linewidth": 2}
    if name in ("mcvar",):
        return {"color": "gray", "linestyle": ":", "linewidth": 2}
    if name in ("entropy",):
        return {"color": "tab:red", "linestyle": "--", "linewidth": 2}
    if name in ("msp",):
        return {"color": "tab:green", "linestyle": "-.", "linewidth": 2}
    if name in ("aleatoric_var",):
        return {"color": "tab:orange", "linestyle": "-.", "linewidth": 2}
    return {"linewidth": 2}


def build_risk_coverage_report(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: Dict[str, torch.Tensor],
    dataset_name: str,
    save_dir: Path,
    plot_title: Optional[str] = None,
    plot_order: Optional[list[str]] = None,
    compute_eaurc: bool = False,
    n_bins: int = 15,
) -> Dict:
    """
    Compute AURC (and optional E-AURC) for multiple uncertainty scores, and write artifacts.

    Args:
      predictions: [N,C] class probabilities
      labels: [N]
      scores: dict name -> [N] uncertainty (lower means more certain / kept first)
    """
    metrics = StandardUQMetrics(n_bins=n_bins)

    aurc: Dict[str, float] = {}
    eaurc: Dict[str, float] = {}
    aurc_opt: Optional[float] = None
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for name, s in scores.items():
        a, cov, risk = metrics.calculate_aurc(predictions, labels, s)
        aurc[name] = float(a)
        curves[name] = {"coverage": cov, "risk": risk}

    if compute_eaurc:
        aurc_opt = float(metrics.calculate_aurc_opt(predictions, labels))
        for name, s in scores.items():
            e, _a, _opt = metrics.calculate_eaurc(predictions, labels, s)
            eaurc[name] = float(e)

    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"risk_coverage_{dataset_name.replace(' ', '_').replace('/', '_')}"
    curve_csv = save_dir / f"{stem}.csv"

    # Use any curve's coverage as the shared x-axis (they are all N steps).
    any_name = next(iter(curves.keys()))
    coverage = curves[any_name]["coverage"]
    _write_curve_csv(curve_csv, coverage, {k: v["risk"] for k, v in curves.items()})

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = plot_order or list(scores.keys())
    for name in order:
        if name not in curves:
            continue
        style = _default_style(name)
        label = f"{name} (AURC={aurc[name]:.4f})"
        if compute_eaurc and name in eaurc:
            label += f", E-AURC={eaurc[name]:.4f}"
        ax.plot(curves[name]["coverage"], curves[name]["risk"], label=label, **style)

    ax.set_xlabel("Coverage (fraction kept)")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title(plot_title or f"Risk–Coverage: {dataset_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    pdf = save_dir / f"{stem}.pdf"
    png = save_dir / f"{stem}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "aurc": aurc,
        "eaurc": (None if not compute_eaurc else eaurc),
        "aurc_opt": (None if not compute_eaurc else aurc_opt),
        "curve_csv": str(curve_csv),
        "plots": {"pdf": str(pdf), "png": str(png)},
        "curves": curves,
    }

