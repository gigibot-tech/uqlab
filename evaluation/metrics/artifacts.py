"""
Evaluation artifacts — contract between runner output and UQ consumers.

``signal_table`` is owned by evaluation (built in ``evaluation/signals/``,
written to ``results.pt`` by ``run_experiment_core``). Vendor code must not
define or parse ``signal_table``; use :class:`EvalRunArtifacts` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

_MC_DROPOUT_SIGNALS = frozenset({"expected_entropy", "mutual_info", "predictive_entropy", "msp_uncertainty"})
_ATTRIBUTION_SIGNALS = frozenset({
    "inverse_coherence",
    "inverse_dominance",
    "inverse_mass",
    "inverse_coherence_dualxda",
    "inverse_dominance_dualxda",
    "inverse_mass_dualxda",
    "inverse_coherence_ek_fak",
    "inverse_dominance_ek_fak",
    "inverse_mass_ek_fak",
})


def _as_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _lookup_signal_vector(
    signal_table: dict[str, np.ndarray],
    signal_id: str,
) -> np.ndarray | None:
    from uqlab.evaluation.signals.registry import resolve_signal_table_key

    key = resolve_signal_table_key(signal_table, signal_id)
    if key is None:
        return None
    return signal_table[key]


def _numpy_signal_table(raw: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in raw.items():
        out[str(key)] = _as_numpy(value).reshape(-1)
    return out


def _missing_signal_message(path: Path, signal: str, role: str, paired: str) -> str:
    msg = f"Missing {role} signal {signal!r} in {path}"
    if signal in _MC_DROPOUT_SIGNALS or paired in _MC_DROPOUT_SIGNALS:
        msg += (
            ". MC-dropout metrics are omitted when dropout=0 or mc_passes is too low "
            f"(enable dropout and include {signal!r} in evaluation.signals)."
        )
    elif signal in _ATTRIBUTION_SIGNALS or paired in _ATTRIBUTION_SIGNALS:
        msg += (
            ". Attribution metrics require the matching DA backend during the job "
            f"(evaluation.attribution_backends: dualxda and/or ek_fak, include {signal!r} "
            "in evaluation.signals)."
        )
    return msg


@dataclass
class EvalRunArtifacts:
    """Runner output consumed by plots, API, and the disentanglement bridge."""

    run_dir: Path | None
    results_path: Path
    predictions: np.ndarray
    signal_table: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_results_pt(cls, path: Path | str) -> EvalRunArtifacts:
        results_path = Path(path)
        data = torch.load(results_path, map_location="cpu", weights_only=False)
        signal_table = _numpy_signal_table(data.get("signal_table") or {})

        if "predictions" in data:
            predictions = _as_numpy(data["predictions"]).reshape(-1)
        elif "eval_clean_labels" in data:
            predictions = _as_numpy(data["eval_clean_labels"]).reshape(-1)
        else:
            raise KeyError(f"No predictions or eval_clean_labels in {results_path}")

        run_dir = results_path.parent if results_path.name == "results.pt" else None
        return cls(
            run_dir=run_dir,
            results_path=results_path,
            predictions=predictions.astype(np.int64),
            signal_table=signal_table,
        )

    def disentangling_vectors(
        self,
        *,
        aleatoric_signal: str,
        epistemic_signal: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vendored ``predict_disentangling`` shape: ``(pred, aleatoric, epistemic)``.

        Reads precomputed ``signal_table`` columns. MC dropout and DualXDA run during
        ``run_experiment_core``, not here — see ``uqlab.shared.config.signals``.
        """
        path = self.results_path
        aleatoric = _lookup_signal_vector(self.signal_table, aleatoric_signal)
        if aleatoric is None:
            raise KeyError(
                _missing_signal_message(path, aleatoric_signal, "aleatoric", epistemic_signal)
            )
        epistemic = _lookup_signal_vector(self.signal_table, epistemic_signal)
        if epistemic is None:
            raise KeyError(
                _missing_signal_message(path, epistemic_signal, "epistemic", aleatoric_signal)
            )
        predictions = self.predictions

        n = len(predictions)
        if len(aleatoric) != n or len(epistemic) != n:
            raise ValueError(
                f"Signal length mismatch in {path}: pred={n}, "
                f"alea={len(aleatoric)}, epi={len(epistemic)}"
            )

        return (
            predictions,
            aleatoric.astype(np.float64),
            epistemic.astype(np.float64),
        )


def uncertainty_vectors_from_results_pt(
    results_path: Path | str,
    *,
    aleatoric_signal: str,
    epistemic_signal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper around :class:`EvalRunArtifacts`."""
    return EvalRunArtifacts.from_results_pt(results_path).disentangling_vectors(
        aleatoric_signal=aleatoric_signal,
        epistemic_signal=epistemic_signal,
    )
