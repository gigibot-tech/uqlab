"""
EK-FAK metric registry: metrics read primitives produced by sources.

Adding a metric: one ``MetricMeta`` in ``catalog.METRIC_META`` plus a compute
callable here; UI and YAML derive metadata from the catalog.

Paper vs Signal (MC dropout decomposition, Keras ``disentanglement_error`` mapping)::

    predictive_entropy  = H[ mean(p(y|x,θ)) ]           # total uncertainty
    expected_entropy    = E[ H(p(y|x,θ)) ]              # aleatoric
    mutual_info         = predictive_entropy - expected_entropy  # epistemic

``predictive_entropy`` and ``mutual_info`` read MC primitives directly;
``expected_entropy`` is derived as ``MC_ENTROPY - MC_MUTUAL_INFO``.

DA backends (DualXDA vs EK-FAC) share structure primitives but use different
source keys and suffixed metric ids (``inverse_*_dualxda``, ``inverse_*_ek_fak``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Set

import torch

from uqlab.evaluation.signals.catalog import (
    METRIC_META,
    MetricMeta,
    SIGNAL_ID_ALIASES,
    aleatoric_signal_ids,
    flatten_enabled_signals,
    metric_runtime_ok,
    normalize_enabled_metrics,
    normalize_signal_id,
    resolve_signal_table_key,
    signal_labels,
    signal_names,
    signals_by_family,
    signals_from_flat_list,
    step4_signal_groups,
)
from uqlab.evaluation.signals.primitives import (
    DUALXDA_COHERENCE,
    DUALXDA_DOMINANCE,
    DUALXDA_MASS,
    EK_FAK_COHERENCE,
    EK_FAK_DOMINANCE,
    EK_FAK_MASS,
    GRADDOT_COHERENCE,
    GRADDOT_DOMINANCE,
    GRADDOT_MASS,
    FWD_DET_LOGITS,
    FWD_MEAN_PRED,
    MC_ENTROPY,
    MC_MEAN_PRED,
    MC_MUTUAL_INFO,
    PrimitiveStore,
    inverse_coherence_from_coherence,
    predicted_class_logit_magnitude,
    reciprocal_uncertainty,
)
from uqlab.evaluation.signals.sources import EvalContext, run_sources, sources_for_metrics


@dataclass(frozen=True, kw_only=True)
class MetricEntry(MetricMeta):
    compute: Callable[[PrimitiveStore], torch.Tensor]


def _msp(store: PrimitiveStore) -> torch.Tensor:
    return 1.0 - store[MC_MEAN_PRED].max(dim=1).values


def _predictive_entropy(store: PrimitiveStore) -> torch.Tensor:
    return store[MC_ENTROPY]


def _mutual_info(store: PrimitiveStore) -> torch.Tensor:
    return store[MC_MUTUAL_INFO]


def _expected_entropy(store: PrimitiveStore) -> torch.Tensor:
    return store[MC_ENTROPY] - store[MC_MUTUAL_INFO]


def _inverse_coherence_dualxda(store: PrimitiveStore) -> torch.Tensor:
    return inverse_coherence_from_coherence(store[DUALXDA_COHERENCE])


def _inverse_dominance_dualxda(store: PrimitiveStore) -> torch.Tensor:
    return (1.0 - store[DUALXDA_DOMINANCE].clamp(0.0, 1.0)).float()


def _inverse_mass_dualxda(store: PrimitiveStore) -> torch.Tensor:
    return reciprocal_uncertainty(store[DUALXDA_MASS])


def _inverse_coherence_ek_fak(store: PrimitiveStore) -> torch.Tensor:
    return inverse_coherence_from_coherence(store[EK_FAK_COHERENCE])


def _inverse_dominance_ek_fak(store: PrimitiveStore) -> torch.Tensor:
    return (1.0 - store[EK_FAK_DOMINANCE].clamp(0.0, 1.0)).float()


def _inverse_mass_ek_fak(store: PrimitiveStore) -> torch.Tensor:
    return reciprocal_uncertainty(store[EK_FAK_MASS])


def _inverse_coherence_graddot(store: PrimitiveStore) -> torch.Tensor:
    return inverse_coherence_from_coherence(store[GRADDOT_COHERENCE])


def _inverse_dominance_graddot(store: PrimitiveStore) -> torch.Tensor:
    return (1.0 - store[GRADDOT_DOMINANCE].clamp(0.0, 1.0)).float()


def _inverse_mass_graddot(store: PrimitiveStore) -> torch.Tensor:
    return reciprocal_uncertainty(store[GRADDOT_MASS])


def _inverse_logit_magnitude(store: PrimitiveStore) -> torch.Tensor:
    mag = predicted_class_logit_magnitude(store[FWD_DET_LOGITS], store[FWD_MEAN_PRED])
    return reciprocal_uncertainty(mag)


_COMPUTE: Dict[str, Callable[[PrimitiveStore], torch.Tensor]] = {
    "msp_uncertainty": _msp,
    "predictive_entropy": _predictive_entropy,
    "expected_entropy": _expected_entropy,
    "mutual_info": _mutual_info,
    "inverse_coherence_dualxda": _inverse_coherence_dualxda,
    "inverse_dominance_dualxda": _inverse_dominance_dualxda,
    "inverse_mass_dualxda": _inverse_mass_dualxda,
    "inverse_coherence_ek_fak": _inverse_coherence_ek_fak,
    "inverse_dominance_ek_fak": _inverse_dominance_ek_fak,
    "inverse_mass_ek_fak": _inverse_mass_ek_fak,
    "inverse_coherence_graddot": _inverse_coherence_graddot,
    "inverse_dominance_graddot": _inverse_dominance_graddot,
    "inverse_mass_graddot": _inverse_mass_graddot,
    "inverse_logit_magnitude": _inverse_logit_magnitude,
}


def _metric_entry(meta: MetricMeta) -> MetricEntry:
    return MetricEntry(
        id=meta.id,
        family=meta.family,
        label=meta.label,
        sources=meta.sources,
        min_dropout=meta.min_dropout,
        epistemic=meta.epistemic,
        aleatoric=meta.aleatoric,
        compute=_COMPUTE[meta.id],
    )


METRICS: Dict[str, MetricEntry] = {mid: _metric_entry(meta) for mid, meta in METRIC_META.items()}

# Backward-compatible alias used across config and tests.
SIGNAL_REGISTRY = METRICS


def prune_enabled_metrics(
    enabled: Iterable[str],
    *,
    mc_passes: int,
    dropout: float,
) -> Set[str]:
    normalized = normalize_enabled_metrics(enabled)
    return {
        mid
        for mid in normalized
        if metric_runtime_ok(mid, mc_passes=mc_passes, dropout=dropout)
    }


def build_signal_table(
    ctx: EvalContext,
    enabled: Set[str] | None = None,
) -> Dict[str, torch.Tensor]:
    """Run required sources, then evaluate enabled metrics."""
    if enabled is None:
        enabled = set(METRICS.keys())
    enabled = prune_enabled_metrics(enabled, mc_passes=ctx.mc_passes, dropout=ctx.dropout)
    if not enabled:
        return {}
    needed = sources_for_metrics(enabled)
    store = run_sources(needed, ctx)
    return {mid: METRICS[mid].compute(store).float() for mid in enabled if mid in METRICS}


def legacy_store_from_kwargs(
    *,
    attribution_signals: Dict[str, torch.Tensor],
    det_logits: torch.Tensor,
    mean_pred_det: torch.Tensor,
    mc_uq: Dict[str, torch.Tensor],
    backend: str = "dualxda",
) -> PrimitiveStore:
    """Build a primitive store without running sources (unit tests / legacy callers)."""
    prefix = f"{backend}."
    store: PrimitiveStore = {
        FWD_DET_LOGITS: det_logits,
        FWD_MEAN_PRED: mean_pred_det,
        MC_MEAN_PRED: mc_uq["mean_prediction"],
        MC_ENTROPY: mc_uq["entropy"],
        MC_MUTUAL_INFO: mc_uq["mutual_info"],
        f"{prefix}coherence": attribution_signals["coherence"],
        f"{prefix}mass": attribution_signals["mass"],
        f"{prefix}dominance": attribution_signals["dominance"],
    }
    if backend == "dualxda":
        from uqlab.evaluation.signals.primitives import (
            ATTR_COHERENCE,
            ATTR_DOMINANCE,
            ATTR_MASS,
        )

        store[ATTR_COHERENCE] = attribution_signals["coherence"]
        store[ATTR_MASS] = attribution_signals["mass"]
        store[ATTR_DOMINANCE] = attribution_signals["dominance"]
    return store


def build_signal_table_from_store(
    store: PrimitiveStore,
    enabled: Set[str] | None = None,
    *,
    mc_passes: int = 10,
    dropout: float = 0.0,
) -> Dict[str, torch.Tensor]:
    if enabled is None:
        enabled = set(METRICS.keys())
    enabled = prune_enabled_metrics(enabled, mc_passes=mc_passes, dropout=dropout)
    return {mid: METRICS[mid].compute(store).float() for mid in enabled if mid in METRICS}


__all__ = [
    "METRICS",
    "METRIC_META",
    "MetricEntry",
    "MetricMeta",
    "SIGNAL_ID_ALIASES",
    "SIGNAL_REGISTRY",
    "aleatoric_signal_ids",
    "build_signal_table",
    "build_signal_table_from_store",
    "epistemic_signal_ids",
    "flatten_enabled_signals",
    "legacy_store_from_kwargs",
    "metric_runtime_ok",
    "normalize_enabled_metrics",
    "normalize_signal_id",
    "prune_enabled_metrics",
    "resolve_signal_table_key",
    "signal_labels",
    "signal_names",
    "signals_by_family",
    "signals_from_flat_list",
    "step4_signal_groups",
]
