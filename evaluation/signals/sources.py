"""Source functions that populate :class:`~uqlab.evaluation.signals.primitives.PrimitiveStore`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Set

import torch
import torch.nn as nn

from uqlab.evaluation.signals.attribution import compute_attribution_structure_signals
from uqlab.evaluation.signals.ek_fak import compute_ek_fak_structure_signals
from uqlab.evaluation.signals.graddot import compute_graddot_structure_signals
from uqlab.evaluation.signals.primitives import (
    FWD_DET_LOGITS,
    FWD_MEAN_PRED,
    MC_ENTROPY,
    MC_MEAN_PRED,
    MC_MUTUAL_INFO,
    PrimitiveStore,
    attribution_distribution_store,
    namespaced_attribution_store,
    predicted_class_logit_magnitude,
)
from uqlab.evaluation.signals.dualxda_tracer import DualXDATracer, infer_classifier_layer_name
from uqlab.evaluation.signals.mc_dropout import calculate_mc_dropout_uncertainty
from uqlab.models.mc_dropout import mc_forward_efficient

AttributionFn = Callable[["EvalContext"], PrimitiveStore]

VALID_ATTRIBUTION_BACKENDS = frozenset({"dualxda", "ek_fak", "graddot"})


def resolve_attribution_backends(
    backends: Iterable[str] | None,
    attribution_method: str | None,
) -> tuple[str, ...]:
    if backends:
        ordered: list[str] = []
        for backend in backends:
            if backend not in VALID_ATTRIBUTION_BACKENDS:
                raise ValueError(
                    f"Unknown attribution backend: {backend!r}. "
                    f"Valid: {sorted(VALID_ATTRIBUTION_BACKENDS)}"
                )
            if backend not in ordered:
                ordered.append(backend)
        return tuple(ordered)
    return (attribution_method or "dualxda",)


@dataclass
class EvalContext:
    """Inputs for running primitive sources during fast-pilot eval."""

    model: nn.Module
    train_dataset: object
    eval_inputs: torch.Tensor
    eval_x: torch.Tensor
    device: torch.device
    train_batch_size: int
    top_k: int
    mc_passes: int
    dropout: float
    attribution_method: str
    run_cache_dir: Path
    num_classes: int = 10
    primitive_cache: PrimitiveStore | None = None
    attribution_backends: tuple[str, ...] = ("dualxda",)


@dataclass(frozen=True)
class SourceEntry:
    id: str
    run: Callable[[EvalContext], PrimitiveStore]
    min_mc_passes: int = 0


def forward_primitives(ctx: EvalContext) -> PrimitiveStore:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        ctx.model.eval()
        batch_size = ctx.train_batch_size
        for start in range(0, int(ctx.eval_x.shape[0]), batch_size):
            end = min(start + batch_size, int(ctx.eval_x.shape[0]))
            chunks.append(ctx.model(ctx.eval_x[start:end]).cpu())
    det_logits = torch.cat(chunks, dim=0)
    mean_pred = torch.softmax(det_logits, dim=1)
    return {
        FWD_DET_LOGITS: det_logits,
        FWD_MEAN_PRED: mean_pred,
    }


def mc_dropout_primitives(ctx: EvalContext) -> PrimitiveStore:
    cache = ctx.primitive_cache or {}
    if FWD_MEAN_PRED in cache:
        mean_pred_det = cache[FWD_MEAN_PRED]
    else:
        mean_pred_det = forward_primitives(ctx)[FWD_MEAN_PRED]
    if ctx.mc_passes > 0:
        mc_predictions = mc_forward_efficient(
            ctx.model,
            ctx.eval_x,
            ctx.mc_passes,
            sample_batch_size=ctx.train_batch_size,
        ).cpu()
        uq = calculate_mc_dropout_uncertainty(mc_predictions)
        return {
            MC_MEAN_PRED: uq["mean_prediction"],
            MC_ENTROPY: uq["entropy"],
            MC_MUTUAL_INFO: uq["mutual_info"],
        }

    n_samples = int(mean_pred_det.shape[0])
    return {
        MC_MEAN_PRED: mean_pred_det,
        MC_ENTROPY: torch.zeros(n_samples),
        MC_MUTUAL_INFO: torch.zeros(n_samples),
    }


def dualxda_primitives(ctx: EvalContext) -> PrimitiveStore:
    cache = ctx.primitive_cache or {}
    if FWD_MEAN_PRED in cache:
        mean_pred_det = cache[FWD_MEAN_PRED]
    else:
        mean_pred_det = forward_primitives(ctx)[FWD_MEAN_PRED]
    tracer = DualXDATracer(
        model=ctx.model,
        train_dataset=ctx.train_dataset,
        layer_name=infer_classifier_layer_name(ctx.model),
        device=str(ctx.device),
        cache_dir=str(ctx.run_cache_dir / "dualxda"),
    )
    raw = compute_attribution_structure_signals(
        tracer,
        ctx.model,
        ctx.eval_inputs,
        mean_pred_det,
        ctx.train_dataset,
        device=ctx.device,
        batch_size=ctx.train_batch_size,
        top_k=ctx.top_k,
        num_classes=ctx.num_classes,
    )
    store = namespaced_attribution_store(
        "dualxda",
        raw["coherence"],
        raw["mass"],
        raw["dominance"],
        write_legacy_alias=True,
    )
    if "influence_matrix" in raw:
        from uqlab.evaluation.signals.primitives import INFLUENCE_DUALXDA

        store[INFLUENCE_DUALXDA] = raw["influence_matrix"]
    dist_store = attribution_distribution_store(
        "dualxda",
        entropy=raw["entropy"],
        participation=raw["participation"],
        signed_split=raw["signed_split"],
        variance=raw["variance"],
    )
    return {**store, **dist_store}


def graddot_primitives(ctx: EvalContext) -> PrimitiveStore:
    cache = ctx.primitive_cache or {}
    if FWD_MEAN_PRED in cache:
        mean_pred_det = cache[FWD_MEAN_PRED]
    else:
        mean_pred_det = forward_primitives(ctx)[FWD_MEAN_PRED]
    raw = compute_graddot_structure_signals(
        ctx.model,
        ctx.train_dataset,
        ctx.eval_inputs,
        mean_pred_det,
        device=ctx.device,
        top_k=ctx.top_k,
        run_cache_dir=ctx.run_cache_dir,
    )
    influence = raw.pop("influence_matrix", None)
    store = namespaced_attribution_store(
        "graddot",
        raw["coherence"],
        raw["mass"],
        raw["dominance"],
    )
    if influence is not None:
        from uqlab.evaluation.signals.primitives import INFLUENCE_GRADDOT

        store[INFLUENCE_GRADDOT] = influence
    return store


def ek_fak_primitives(ctx: EvalContext) -> PrimitiveStore:
    cache = ctx.primitive_cache or {}
    if FWD_MEAN_PRED in cache:
        mean_pred_det = cache[FWD_MEAN_PRED]
    else:
        mean_pred_det = forward_primitives(ctx)[FWD_MEAN_PRED]
    raw = compute_ek_fak_structure_signals(
        ctx.model,
        ctx.train_dataset,
        ctx.eval_inputs,
        mean_pred_det,
        device=ctx.device,
        batch_size=ctx.train_batch_size,
        top_k=ctx.top_k,
        run_cache_dir=ctx.run_cache_dir,
        train_batch_size=ctx.train_batch_size,
    )
    influence = raw.pop("influence_matrix", None)
    store = namespaced_attribution_store(
        "ek_fak",
        raw["coherence"],
        raw["mass"],
        raw["dominance"],
    )
    if influence is not None:
        from uqlab.evaluation.signals.primitives import INFLUENCE_EK_FAK

        store[INFLUENCE_EK_FAK] = influence
    return store


ATTRIBUTION_METHODS: Dict[str, AttributionFn] = {
    "dualxda": dualxda_primitives,
    "ek_fak": ek_fak_primitives,
    "graddot": graddot_primitives,
}


def attribution_dualxda_primitives(ctx: EvalContext) -> PrimitiveStore:
    return dualxda_primitives(ctx)


def attribution_ek_fak_primitives(ctx: EvalContext) -> PrimitiveStore:
    return ek_fak_primitives(ctx)


def attribution_graddot_primitives(ctx: EvalContext) -> PrimitiveStore:
    return graddot_primitives(ctx)


def attribution_primitives(ctx: EvalContext) -> PrimitiveStore:
    """Legacy single-backend source (DualXDA only)."""
    return dualxda_primitives(ctx)


SOURCE_REGISTRY: Dict[str, SourceEntry] = {
    "deterministic_forward": SourceEntry("deterministic_forward", forward_primitives),
    "mc_dropout": SourceEntry("mc_dropout", mc_dropout_primitives, min_mc_passes=1),
    "attribution": SourceEntry("attribution", attribution_primitives),
    "attribution_dualxda": SourceEntry("attribution_dualxda", attribution_dualxda_primitives),
    "attribution_ek_fak": SourceEntry("attribution_ek_fak", attribution_ek_fak_primitives),
    "attribution_graddot": SourceEntry("attribution_graddot", attribution_graddot_primitives),
}


def run_sources(source_ids: Iterable[str], ctx: EvalContext) -> PrimitiveStore:
    """Run only the requested sources and merge into one primitive store."""
    store: PrimitiveStore = {}
    order = (
        "deterministic_forward",
        "mc_dropout",
        "attribution",
        "attribution_dualxda",
        "attribution_ek_fak",
        "attribution_graddot",
    )
    requested = set(source_ids)
    for sid in order:
        if sid not in requested:
            continue
        entry = SOURCE_REGISTRY[sid]
        if entry.min_mc_passes > 0 and ctx.mc_passes < entry.min_mc_passes:
            raise ValueError(
                f"Source {sid!r} requires mc_passes >= {entry.min_mc_passes} "
                f"(got {ctx.mc_passes})"
            )
        ctx.primitive_cache = store
        store.update(entry.run(ctx))
    return store


def sources_for_metrics(metric_ids: Iterable[str]) -> Set[str]:
    from uqlab.evaluation.signals.registry import METRICS, normalize_signal_id

    needed: Set[str] = set()
    for mid in metric_ids:
        resolved = normalize_signal_id(str(mid))
        needed.update(METRICS[resolved].sources)
    return needed


def logit_magnitude_from_store(store: PrimitiveStore) -> torch.Tensor:
    return predicted_class_logit_magnitude(store[FWD_DET_LOGITS], store[FWD_MEAN_PRED])
