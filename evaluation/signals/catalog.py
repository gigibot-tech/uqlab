"""Torch-free signal metadata catalog (labels, families, source requirements)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Mapping, Set

SignalFamily = Literal["predictive", "logit", "attribution"]

SIGNAL_ID_ALIASES: Dict[str, str] = {
    "dominance": "inverse_dominance_dualxda",
    "inverse_coherence": "inverse_coherence_dualxda",
    "inverse_mass": "inverse_mass_dualxda",
    "inverse_dominance": "inverse_dominance_dualxda",
}


@dataclass(frozen=True)
class MetricMeta:
    id: str
    family: SignalFamily
    label: str
    sources: tuple[str, ...]
    min_dropout: float = 0.0
    epistemic: bool = False
    aleatoric: bool = False


METRIC_META: Dict[str, MetricMeta] = {
    "msp_uncertainty": MetricMeta(
        id="msp_uncertainty",
        family="predictive",
        label="MSP",
        sources=("mc_dropout",),
    ),
    "predictive_entropy": MetricMeta(
        id="predictive_entropy",
        family="predictive",
        label="Predictive Entropy",
        sources=("mc_dropout",),
        min_dropout=1e-12,
    ),
    "expected_entropy": MetricMeta(
        id="expected_entropy",
        family="predictive",
        label="Expected Entropy",
        sources=("mc_dropout",),
        min_dropout=1e-12,
        aleatoric=True,
    ),
    "mutual_info": MetricMeta(
        id="mutual_info",
        family="predictive",
        label="Mutual Information",
        sources=("mc_dropout",),
        min_dropout=1e-12,
        epistemic=True,
    ),
    "inverse_coherence_dualxda": MetricMeta(
        id="inverse_coherence_dualxda",
        family="attribution",
        label="Inverse Coherence (DualXDA)",
        sources=("attribution_dualxda",),
        aleatoric=True,
    ),
    "inverse_dominance_dualxda": MetricMeta(
        id="inverse_dominance_dualxda",
        family="attribution",
        label="Inverse Dominance (DualXDA)",
        sources=("attribution_dualxda",),
        epistemic=True,
    ),
    "inverse_mass_dualxda": MetricMeta(
        id="inverse_mass_dualxda",
        family="logit",
        label="Inverse Mass (DualXDA)",
        sources=("attribution_dualxda",),
        epistemic=True,
    ),
    "inverse_coherence_ek_fak": MetricMeta(
        id="inverse_coherence_ek_fak",
        family="attribution",
        label="Inverse Coherence (EK-FAC)",
        sources=("attribution_ek_fak",),
        aleatoric=True,
    ),
    "inverse_dominance_ek_fak": MetricMeta(
        id="inverse_dominance_ek_fak",
        family="attribution",
        label="Inverse Dominance (EK-FAC)",
        sources=("attribution_ek_fak",),
        epistemic=True,
    ),
    "inverse_mass_ek_fak": MetricMeta(
        id="inverse_mass_ek_fak",
        family="logit",
        label="Inverse Mass (EK-FAC)",
        sources=("attribution_ek_fak",),
        epistemic=True,
    ),
    "inverse_coherence_graddot": MetricMeta(
        id="inverse_coherence_graddot",
        family="attribution",
        label="Inverse Coherence (GradDot)",
        sources=("attribution_graddot",),
        aleatoric=True,
    ),
    "inverse_dominance_graddot": MetricMeta(
        id="inverse_dominance_graddot",
        family="attribution",
        label="Inverse Dominance (GradDot)",
        sources=("attribution_graddot",),
        epistemic=True,
    ),
    "inverse_mass_graddot": MetricMeta(
        id="inverse_mass_graddot",
        family="logit",
        label="Inverse Mass (GradDot)",
        sources=("attribution_graddot",),
        epistemic=True,
    ),
    "inverse_logit_magnitude": MetricMeta(
        id="inverse_logit_magnitude",
        family="logit",
        label="Inverse Logit Magnitude",
        sources=("deterministic_forward",),
        epistemic=True,
    ),
    "attribution_entropy_dualxda": MetricMeta(
        id="attribution_entropy_dualxda",
        family="attribution",
        label="Attribution Entropy (DualXDA)",
        sources=("attribution_dualxda",),
        epistemic=True,
    ),
    "attribution_participation_dualxda": MetricMeta(
        id="attribution_participation_dualxda",
        family="attribution",
        label="Attribution Participation (DualXDA)",
        sources=("attribution_dualxda",),
        epistemic=True,
    ),
    "attribution_signed_split_dualxda": MetricMeta(
        id="attribution_signed_split_dualxda",
        family="attribution",
        label="Attribution Signed Split (DualXDA)",
        sources=("attribution_dualxda",),
        aleatoric=True,
    ),
    "attribution_variance_dualxda": MetricMeta(
        id="attribution_variance_dualxda",
        family="attribution",
        label="Attribution Variance (DualXDA)",
        sources=("attribution_dualxda",),
        aleatoric=True,
    ),
}


def normalize_signal_id(signal_id: str) -> str:
    return SIGNAL_ID_ALIASES.get(signal_id, signal_id)


def normalize_enabled_metrics(enabled: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for mid in enabled:
        resolved = normalize_signal_id(str(mid))
        if resolved in METRIC_META:
            out.add(resolved)
    return out


def metric_runtime_ok(metric_id: str, *, mc_passes: int, dropout: float) -> bool:
    metric = METRIC_META[normalize_signal_id(metric_id)]
    for src_id in metric.sources:
        from uqlab.evaluation.signals.sources import SOURCE_REGISTRY

        entry = SOURCE_REGISTRY[src_id]
        if mc_passes < entry.min_mc_passes:
            return False
    if metric.min_dropout > 0 and dropout <= 0:
        return False
    return True


def signal_names() -> List[str]:
    return list(METRIC_META.keys())


def signal_labels() -> Dict[str, str]:
    return {mid: m.label for mid, m in METRIC_META.items()}


def epistemic_signal_ids() -> List[str]:
    return [mid for mid, m in METRIC_META.items() if m.epistemic]


def aleatoric_signal_ids() -> List[str]:
    return [mid for mid, m in METRIC_META.items() if m.aleatoric]


def signals_by_family() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"predictive": [], "logit": [], "attribution": []}
    for mid, m in METRIC_META.items():
        out[m.family].append(mid)
    return out


def signals_from_flat_list(names: Iterable[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {family: [] for family in ("predictive", "logit", "attribution")}
    for raw in names:
        sid = normalize_signal_id(str(raw))
        metric = METRIC_META.get(sid)
        if metric is None:
            raise ValueError(f"Unknown signal id: {raw!r}")
        if sid not in out[metric.family]:
            out[metric.family].append(sid)
    return out


def flatten_enabled_signals(signals: Mapping[str, List[str]]) -> List[str]:
    enabled = {normalize_signal_id(s) for family in signals.values() for s in family}
    return [mid for mid in METRIC_META if mid in enabled]


def resolve_signal_table_key(
    signal_table: Mapping[str, object],
    signal_id: str,
) -> str | None:
    """Resolve suffixed metric ids and legacy alias keys in ``signal_table`` dicts."""
    sid = normalize_signal_id(signal_id)
    for candidate in (sid, signal_id):
        if candidate in signal_table:
            return candidate
    for alias, target in SIGNAL_ID_ALIASES.items():
        if target == sid and alias in signal_table:
            return alias
    return None


def step4_signal_groups() -> Dict[str, List[str]]:
    return {
        "Epistemic Signals": epistemic_signal_ids(),
        "Aleatoric Signals": aleatoric_signal_ids(),
        "Baseline Signals (MC dropout)": [
            mid for mid, m in METRIC_META.items() if m.family == "predictive"
        ],
    }
