"""Uncertainty signal families and evaluation validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from uqlab.evaluation.signals.catalog import (
    METRIC_META,
    SIGNAL_ID_ALIASES,
    flatten_enabled_signals,
    metric_runtime_ok,
    normalize_signal_id,
    signals_by_family,
    signals_from_flat_list,
)

_REGISTRY_BY_FAMILY = signals_by_family()

# Default job signals: DualXDA attribution only (EK-FAC metrics are opt-in).
DEFAULT_SIGNALS: Dict[str, List[str]] = {
    "predictive": list(_REGISTRY_BY_FAMILY["predictive"]),
    "logit": ["inverse_logit_magnitude"],
    "attribution": [
        "inverse_coherence_dualxda",
        "inverse_dominance_dualxda",
        "inverse_mass_dualxda",
    ],
}

# Single flat list for Step 4 UI defaults, workflow_defaults, and headless YAML fallback.
DEFAULT_SELECTED_SIGNALS: List[str] = [
    "msp_uncertainty",
    "predictive_entropy",
    "expected_entropy",
    "mutual_info",
    "inverse_coherence_dualxda",
    "inverse_dominance_dualxda",
    "inverse_mass_dualxda",
    "inverse_logit_magnitude",
]

# Opt-in EK-FAC metrics (require ``kronfluence`` when any is enabled).
EK_FAK_OPTIONAL_SIGNALS: List[str] = [
    mid for mid in _REGISTRY_BY_FAMILY["attribution"] + _REGISTRY_BY_FAMILY["logit"]
    if mid.endswith("_ek_fak")
]

# Sweep line-plot picker defaults (DualXDA bridge pair; legacy alias columns still resolve).
PLOT_DEFAULT_ALEATORIC_SIGNAL = "inverse_coherence_dualxda"
PLOT_DEFAULT_EPISTEMIC_SIGNAL = "inverse_mass_dualxda"

# Full registry grouping (includes opt-in EK-FAC metrics).
SIGNAL_FAMILIES = _REGISTRY_BY_FAMILY

# Backward-compatible alias.
SIGNAL_REGISTRY = METRIC_META

_ALL_SIGNALS = set(METRIC_META.keys()) | set(SIGNAL_ID_ALIASES.keys())

DisentanglingBridgeMode = Literal["paper", "signal", "signal_dualxda", "signal_ek_fak"]

BRIDGE_PRESET_ALIASES: Dict[str, DisentanglingBridgeMode] = {
    "signal": "signal_dualxda",
}


@dataclass(frozen=True)
class SignalCatalogEntry:
    """Shared metadata for docs, UI labels, and bridge validation."""

    id: str
    framework: str
    method: str
    display_name: str


SIGNAL_CATALOG: Dict[str, SignalCatalogEntry] = {
    "msp_uncertainty": SignalCatalogEntry(
        "msp_uncertainty",
        "Information-theoretic",
        "MC dropout",
        "MSP Uncertainty (Information-theoretic · MC dropout)",
    ),
    "predictive_entropy": SignalCatalogEntry(
        "predictive_entropy",
        "Information-theoretic",
        "MC dropout",
        "Predictive Entropy (Information-theoretic · MC dropout)",
    ),
    "expected_entropy": SignalCatalogEntry(
        "expected_entropy",
        "Information-theoretic",
        "MC dropout",
        "Expected Entropy (Information-theoretic · MC dropout)",
    ),
    "mutual_info": SignalCatalogEntry(
        "mutual_info",
        "Information-theoretic",
        "MC dropout",
        "Mutual Information (Information-theoretic · MC dropout)",
    ),
    "inverse_coherence_dualxda": SignalCatalogEntry(
        "inverse_coherence_dualxda",
        "DualXDA",
        "DualXDA",
        "Inverse Coherence (DualXDA)",
    ),
    "inverse_dominance_dualxda": SignalCatalogEntry(
        "inverse_dominance_dualxda",
        "DualXDA",
        "DualXDA",
        "Inverse Dominance (DualXDA)",
    ),
    "inverse_mass_dualxda": SignalCatalogEntry(
        "inverse_mass_dualxda",
        "DualXDA",
        "DualXDA",
        "Inverse Mass (DualXDA)",
    ),
    "inverse_coherence_ek_fak": SignalCatalogEntry(
        "inverse_coherence_ek_fak",
        "EK-FAC",
        "Kronfluence",
        "Inverse Coherence (EK-FAC)",
    ),
    "inverse_dominance_ek_fak": SignalCatalogEntry(
        "inverse_dominance_ek_fak",
        "EK-FAC",
        "Kronfluence",
        "Inverse Dominance (EK-FAC)",
    ),
    "inverse_mass_ek_fak": SignalCatalogEntry(
        "inverse_mass_ek_fak",
        "EK-FAC",
        "Kronfluence",
        "Inverse Mass (EK-FAC)",
    ),
    "inverse_logit_magnitude": SignalCatalogEntry(
        "inverse_logit_magnitude",
        "Representer logit",
        "Deterministic forward",
        "Inverse Logit Magnitude (Representer logit)",
    ),
}

DISENTANGLING_BRIDGE_PRESETS: Dict[DisentanglingBridgeMode, Tuple[str, str]] = {
    "paper": ("expected_entropy", "mutual_info"),
    "signal_dualxda": ("inverse_coherence_dualxda", "inverse_mass_dualxda"),
    "signal_ek_fak": ("inverse_coherence_ek_fak", "inverse_mass_ek_fak"),
}

PREDICT_DISENTANGLING_NOTE = (
    "``predict_disentangling`` does not run MC dropout or attribution backends. It reads "
    "precomputed columns from ``results.pt`` / ``signal_table``. Paper bridge "
    "(``expected_entropy`` + ``mutual_info``) requires the job to have computed the "
    "``mc_dropout`` source (``model.dropout > 0``, ``evaluation.mc_passes > 0``). "
    "DualXDA bridge (``inverse_coherence_dualxda`` + ``inverse_mass_dualxda``) requires "
    "``attribution_dualxda`` during the job. EK-FAC bridge "
    "(``inverse_coherence_ek_fak`` + ``inverse_mass_ek_fak``) requires "
    "``attribution_ek_fak`` (optional ``kronfluence`` package)."
)


def _normalize_bridge_mode(predict_mode: str) -> DisentanglingBridgeMode:
    resolved = BRIDGE_PRESET_ALIASES.get(predict_mode, predict_mode)
    if resolved not in DISENTANGLING_BRIDGE_PRESETS:
        raise ValueError(
            f"Unknown predict_mode: {predict_mode!r}. "
            f"Choose from {sorted(DISENTANGLING_BRIDGE_PRESETS)} "
            f"(alias: signal → signal_dualxda)."
        )
    return resolved  # type: ignore[return-value]


def signal_display_name(signal_id: str) -> str:
    sid = normalize_signal_id(signal_id)
    entry = SIGNAL_CATALOG.get(sid)
    if entry is not None:
        return entry.display_name
    metric = METRIC_META.get(sid)
    return metric.label if metric is not None else sid


def _metric_sources(signal_id: str) -> tuple[str, ...]:
    sid = normalize_signal_id(signal_id)
    return METRIC_META[sid].sources


def signals_require_mc_dropout(signal_ids: Iterable[str]) -> bool:
    return any("mc_dropout" in _metric_sources(sid) for sid in signal_ids)


def signals_require_attribution(signal_ids: Iterable[str]) -> bool:
    sources = set()
    for sid in signal_ids:
        sources.update(_metric_sources(sid))
    return bool(sources & {"attribution", "attribution_dualxda", "attribution_ek_fak", "attribution_graddot"})


def _attribution_backend_note(signal_id: str) -> str | None:
    sid = normalize_signal_id(signal_id)
    if sid.endswith("_ek_fak"):
        return "evaluation.attribution_backends includes ek_fak (Kronfluence during eval)"
    if sid.endswith("_graddot"):
        return "evaluation.attribution_backends includes graddot (gradient dot product during eval)"
    if sid.endswith("_dualxda") or sid in SIGNAL_ID_ALIASES:
        return "evaluation.attribution_backends includes dualxda (DualXDA tracer during eval)"
    for src in _metric_sources(sid):
        if src == "attribution_ek_fak":
            return "evaluation.attribution_backends includes ek_fak (Kronfluence during eval)"
        if src == "attribution_graddot":
            return "evaluation.attribution_backends includes graddot (gradient dot product during eval)"
        if src in {"attribution_dualxda", "attribution"}:
            return "evaluation.attribution_backends includes dualxda (DualXDA tracer during eval)"
    return None


def bridge_job_requirements(aleatoric_signal: str, epistemic_signal: str) -> Dict[str, str]:
    """What ``run_experiment_core`` must compute before ``predict_disentangling`` succeeds."""
    pair = (normalize_signal_id(aleatoric_signal), normalize_signal_id(epistemic_signal))
    notes: Dict[str, str] = {
        "predict_disentangling": "read-only: loads ``signal_table`` from ``results.pt``",
    }
    if signals_require_mc_dropout(pair):
        notes["mc_dropout"] = "model.dropout > 0 and evaluation.mc_passes > 0 during the job"
    if signals_require_attribution(pair):
        for sid in pair:
            note = _attribution_backend_note(sid)
            if note:
                key = "ek_fak" if "ek_fak" in note else "dualxda"
                notes[key] = note
    return notes


def resolve_disentangling_signal_pair(
    *,
    predict_mode: DisentanglingBridgeMode | str = "paper",
    aleatoric_signal: str | None = None,
    epistemic_signal: str | None = None,
    workflow: Mapping[str, Any] | None = None,
) -> Tuple[str, str]:
    """
    Resolve vendor bridge columns for ``predict_disentangling``.

    Priority: explicit kwargs → ``workflow['uncertainty_config']`` → ``predict_mode`` preset.
    """
    mode = _normalize_bridge_mode(str(predict_mode))
    preset_alea, preset_epi = DISENTANGLING_BRIDGE_PRESETS[mode]
    wf_alea: str | None = None
    wf_epi: str | None = None
    if workflow:
        uc = workflow.get("uncertainty_config") or {}
        wf_alea = uc.get("aleatoric_signal")
        wf_epi = uc.get("epistemic_signal")

    alea = aleatoric_signal if aleatoric_signal is not None else wf_alea
    epi = epistemic_signal if epistemic_signal is not None else wf_epi
    if alea is None:
        alea = preset_alea
    if epi is None:
        epi = preset_epi
    return normalize_signal_id(str(alea)), normalize_signal_id(str(epi))


def derive_attribution_backends_from_signals(signal_ids: Iterable[str]) -> tuple[str, ...]:
    """Infer which DA backends to run from enabled metric ids (no separate config knob)."""
    backends: list[str] = []
    for raw in signal_ids:
        sid = normalize_signal_id(str(raw))
        metric = METRIC_META.get(sid)
        if metric is None:
            continue
        for src in metric.sources:
            if src in {"attribution_dualxda", "attribution"} and "dualxda" not in backends:
                backends.append("dualxda")
            elif src == "attribution_ek_fak" and "ek_fak" not in backends:
                backends.append("ek_fak")
            elif src == "attribution_graddot" and "graddot" not in backends:
                backends.append("graddot")
    return tuple(backends)


def plot_default_signal_for_sweep(sweep_kind: str) -> str:
    """Default signal in sweep line plots (label-noise → aleatoric, dataset-size → epistemic)."""
    if sweep_kind in ("label_noise", "aleatoric"):
        return PLOT_DEFAULT_ALEATORIC_SIGNAL
    return PLOT_DEFAULT_EPISTEMIC_SIGNAL


def signal_id_column_candidates(signal_id: str) -> List[str]:
    """Registry id plus legacy alias keys that may appear in older ``metrics.csv`` / plots."""
    sid = normalize_signal_id(signal_id)
    out = [sid]
    for alias, target in SIGNAL_ID_ALIASES.items():
        if target == sid and alias not in out:
            out.append(alias)
    return out


def iter_disentangling_bridge_pairs(
    modes: Iterable[str] | None = None,
) -> List[Tuple[str, str, str]]:
    """
    Named bridge presets for campaign scoring on one ``results.pt``.

    Returns ``(preset_name, aleatoric_signal, epistemic_signal)`` tuples.
    """
    if modes is None:
        modes = list(DISENTANGLING_BRIDGE_PRESETS.keys())
    out: List[Tuple[str, str, str]] = []
    for mode in modes:
        resolved = BRIDGE_PRESET_ALIASES.get(mode, mode)
        if resolved not in DISENTANGLING_BRIDGE_PRESETS:
            continue
        alea, epi = DISENTANGLING_BRIDGE_PRESETS[resolved]  # type: ignore[index]
        out.append((str(resolved), alea, epi))
    return out


def flatten_signals(signals: Mapping[str, List[str]]) -> List[str]:
    return flatten_enabled_signals(signals)


def normalize_evaluation_signals(raw: Optional[Mapping[str, Any] | Iterable[str]]) -> Dict[str, List[str]]:
    if not raw:
        return dict(DEFAULT_SIGNALS)
    if isinstance(raw, (list, tuple)):
        return signals_from_flat_list(raw)
    if isinstance(raw, dict) and any(k in raw for k in DEFAULT_SIGNALS):
        return {
            family: [normalize_signal_id(s) for s in raw.get(family, DEFAULT_SIGNALS[family])]
            for family in DEFAULT_SIGNALS
        }
    return dict(DEFAULT_SIGNALS)


def prune_signals_for_runtime(
    signals: Mapping[str, List[str]],
    *,
    mc_passes: int,
    dropout: float,
) -> Dict[str, List[str]]:
    """Drop metrics whose source requirements fail at runtime."""
    flat = [normalize_signal_id(s) for s in flatten_signals(signals)]
    kept = [s for s in flat if s in METRIC_META and metric_runtime_ok(s, mc_passes=mc_passes, dropout=dropout)]
    if not kept:
        return {family: [] for family in DEFAULT_SIGNALS}
    return signals_from_flat_list(kept)


def validate_evaluation_signals(
    *,
    signals: Mapping[str, List[str]],
    mc_passes: int,
    dropout: float,
) -> None:
    """Raise ValueError when predictive MC signals are requested without MC/dropout."""
    flat = {normalize_signal_id(s) for s in flatten_signals(signals)}
    unknown = flat - set(METRIC_META.keys())
    if unknown:
        raise ValueError(f"Unknown evaluation signals: {sorted(unknown)}")

    for mid in flat:
        if mid not in METRIC_META:
            continue
        if not metric_runtime_ok(mid, mc_passes=mc_passes, dropout=dropout):
            if mid == "mutual_info":
                if mc_passes <= 0:
                    raise ValueError("mutual_info requires evaluation.mc_passes > 0")
                if dropout <= 0:
                    raise ValueError("mutual_info requires model.dropout > 0")
            elif mid == "predictive_entropy" and mc_passes <= 0:
                raise ValueError(
                    "Predictive entropy requires evaluation.mc_passes > 0 "
                    f"(got mc_passes={mc_passes})"
                )
