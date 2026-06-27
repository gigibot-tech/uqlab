"""Score multiple disentanglement bridge pairs from one ``results.pt``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uqlab.evaluation.metrics.artifacts import EvalRunArtifacts
from uqlab.shared.config.signals import (
    bridge_job_requirements,
    iter_disentangling_bridge_pairs,
    resolve_disentangling_signal_pair,
)
from uqlab.vendor.disentanglement_error.error_metric import calculate_disentanglement_error


def score_bridge_pairs_from_results(
    results_path: Path | str,
    *,
    modes: Iterable[str] | None = None,
    pairs: Iterable[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Loop bridge presets on a single completed run (no re-training).

    Each entry: ``preset``, ``aleatoric_signal``, ``epistemic_signal``,
    ``requirements``, ``pred`` / ``aleatorics`` / ``epistemics`` shapes, and
    optional ``disentanglement_score`` when the vendor metric can run.
    """
    artifacts = EvalRunArtifacts.from_results_pt(results_path)
    rows: list[dict[str, Any]] = []

    preset_pairs = list(pairs) if pairs is not None else iter_disentangling_bridge_pairs(modes)
    for preset_name, alea, epi in preset_pairs:
        row: dict[str, Any] = {
            "preset": preset_name,
            "aleatoric_signal": alea,
            "epistemic_signal": epi,
            "requirements": bridge_job_requirements(alea, epi),
            "results_path": str(artifacts.results_path),
        }
        try:
            pred, aleatorics, epistemics = artifacts.disentangling_vectors(
                aleatoric_signal=alea,
                epistemic_signal=epi,
            )
        except KeyError as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue

        row["n_samples"] = int(len(pred))
        row["aleatorics_mean"] = float(np.mean(aleatorics))
        row["epistemics_mean"] = float(np.mean(epistemics))
        rows.append(row)

    return rows


def score_bridge_pair_with_vendor_metric(
    results_path: Path | str,
    *,
    predict_mode: str = "paper",
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
) -> float:
    """
    Run ``calculate_disentanglement_error`` for one bridge pair using vectors from ``results.pt``.

    Uses dummy ``x_test`` / labels — the vendor metric only needs the uncertainty vectors
    and accuracy correlation structure from the loaded run.
    """
    alea, epi = resolve_disentangling_signal_pair(
        predict_mode=predict_mode,
        aleatoric_signal=aleatoric_signal,
        epistemic_signal=epistemic_signal,
    )
    artifacts = EvalRunArtifacts.from_results_pt(results_path)
    pred, aleatorics, epistemics = artifacts.disentangling_vectors(
        aleatoric_signal=alea,
        epistemic_signal=epi,
    )
    n = len(pred)
    x_dummy = np.zeros((n, 1), dtype=np.float32)
    y_dummy = pred.astype(np.int64)

    class _ResultsReader:
        def predict_disentangling(self, x):
            return pred, aleatorics, epistemics

    return float(
        calculate_disentanglement_error(
            x_dummy,
            y_dummy,
            _ResultsReader(),
            return_json=False,
        )
    )
