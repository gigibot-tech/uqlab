"""
Declarative provenance for fast-pilot uncertainty signals.

Each exported column in ``build_fast_pilot_signal_table`` has a matching
:class:`SignalFormulaSpec` describing operands (parts) and operators so runs
can be audited without reading implementation code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FormulaPart:
    """Named intermediate quantity fed into a signal."""

    name: str
    description: str
    source: str  # e.g. "dualxda_trace", "mc_dropout", "deterministic_forward"


@dataclass(frozen=True)
class FormulaOperator:
    """One step in the computation DAG (in order)."""

    op: str  # e.g. "topk", "sum_abs", "reciprocal", "one_minus"
    inputs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalFormulaSpec:
    """Full recipe for one scalar signal per eval sample."""

    signal: str
    label: str
    category: str  # predictive | attribution_structure | derived
    parts: tuple[FormulaPart, ...]
    operators: tuple[FormulaOperator, ...]
    formula: str  # human-readable
    implementation: str  # module.function
    exported_in_fast_pilot: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _topk_attribution_parts(top_k: int) -> tuple[FormulaPart, ...]:
    return (
        FormulaPart(
            "T_i",
            f"DualXDA attribution to training sample i (full trace); top-{top_k} by |T_i|",
            "dualxda_trace",
        ),
        FormulaPart(
            "topk_idx",
            f"Indices of top-{top_k} influences by |T_i| per eval sample",
            "dualxda_trace",
        ),
    )


def fast_pilot_signal_formula_specs(
    *,
    top_k: int = 10,
    mc_passes: int = 30,
    mass_eps: float = 1e-8,
) -> dict[str, SignalFormulaSpec]:
    """Canonical formulas for columns written by ``build_fast_pilot_signal_table``."""
    attr_parts = _topk_attribution_parts(top_k)
    common_topk_ops = (
        FormulaOperator("topk", ("T_i",), {"k": top_k, "key": "abs"}),
        FormulaOperator("select", ("T_i", "topk_idx"), {}),
    )

    specs: list[SignalFormulaSpec] = [
        SignalFormulaSpec(
            signal="msp_uncertainty",
            label="MSP uncertainty",
            category="predictive",
            parts=(
                FormulaPart("p_bar", "Mean softmax over MC passes", "mc_dropout"),
            ),
            operators=(
                FormulaOperator("mc_mean", ("p(y|x,θ_t)",), {"n_passes": mc_passes}),
                FormulaOperator("max", ("p_bar",), {"dim": "class"}),
                FormulaOperator("one_minus", ("max_p",), {}),
            ),
            formula="1 - max_c E_t[p(y=c|x,θ_t)]",
            implementation="attribution_signals.map_mc_dropout_to_predictive_signals",
        ),
        SignalFormulaSpec(
            signal="predictive_entropy",
            label="Predictive entropy",
            category="predictive",
            parts=(FormulaPart("p_bar", "Mean softmax over MC passes", "mc_dropout"),),
            operators=(
                FormulaOperator("mc_mean", ("p(y|x,θ_t)",), {"n_passes": mc_passes}),
                FormulaOperator("entropy", ("p_bar",), {"eps": 1e-10}),
            ),
            formula="H(E_t[p(y|x,θ_t)]) = -Σ_c p̄_c log p̄_c",
            implementation="mc_dropout_uq.calculate_mc_dropout_uncertainty",
        ),
        SignalFormulaSpec(
            signal="mutual_info",
            label="Mutual information (epistemic proxy)",
            category="predictive",
            parts=(
                FormulaPart("p_t", "Softmax per MC pass", "mc_dropout"),
                FormulaPart("p_bar", "Mean softmax over passes", "mc_dropout"),
            ),
            operators=(
                FormulaOperator("mc_stack", ("p(y|x,θ_t)",), {"n_passes": mc_passes}),
                FormulaOperator("entropy", ("p_bar",), {"eps": 1e-10}),
                FormulaOperator("mc_mean_entropy", ("p_t",), {"eps": 1e-10}),
                FormulaOperator("subtract", ("H(p_bar)", "E_t[H(p_t)]"), {}),
            ),
            formula="H(E_t[p]) - E_t[H(p_t)]",
            implementation="mc_dropout_uq.calculate_mc_dropout_uncertainty",
        ),
        SignalFormulaSpec(
            signal="coherence",
            label="Attribution coherence (top-k)",
            category="attribution_structure",
            parts=attr_parts,
            operators=(
                *common_topk_ops,
                FormulaOperator("sum_signed", ("T_topk",), {}),
                FormulaOperator("sum_abs", ("T_topk",), {}),
                FormulaOperator("ratio_abs", ("sum_signed", "sum_abs"), {"eps": mass_eps}),
            ),
            formula="|Σ_{i∈topk} T_i| / (Σ_{i∈topk} |T_i| + ε)",
            implementation="attribution_signals.topk_influence_metrics",
        ),
        SignalFormulaSpec(
            signal="inverse_coherence",
            label="Inverse coherence",
            category="derived",
            parts=(FormulaPart("coherence", "Top-k coherence per sample", "attribution_structure"),),
            operators=(FormulaOperator("one_minus", ("coherence",), {"clamp": "[0,1]"}),),
            formula="1 - coherence",
            implementation="attribution_signals.inverse_coherence_from_coherence",
        ),
        SignalFormulaSpec(
            signal="dominance",
            label="Attribution dominance (top-k)",
            category="attribution_structure",
            parts=attr_parts,
            operators=(
                *common_topk_ops,
                FormulaOperator("max_abs", ("T_topk",), {}),
                FormulaOperator("sum_abs", ("T_topk",), {}),
                FormulaOperator("ratio", ("max_abs", "sum_abs"), {"eps": mass_eps}),
            ),
            formula="max_{i∈topk}|T_i| / (Σ_{i∈topk}|T_i| + ε)",
            implementation="attribution_signals.topk_influence_metrics",
        ),
        SignalFormulaSpec(
            signal="inverse_mass",
            label="Inverse attribution mass",
            category="derived",
            parts=(
                FormulaPart("mass_k", "sum of |T_i| over top-k supporters", "attribution_structure"),
            ),
            operators=(
                FormulaOperator("reciprocal", ("mass_k",), {"eps": mass_eps}),
            ),
            formula=f"1 / (mass_k + {mass_eps})",
            implementation="attribution_signals.reciprocal_uncertainty",
        ),
        SignalFormulaSpec(
            signal="inverse_logit_magnitude",
            label="Inverse logit magnitude",
            category="derived",
            parts=(
                FormulaPart(
                    "logit_mag",
                    "|logit| for predicted class (deterministic eval forward)",
                    "deterministic_forward",
                ),
            ),
            operators=(FormulaOperator("reciprocal", ("logit_mag",), {"eps": mass_eps}),),
            formula=f"1 / (|logit_pred| + {mass_eps})",
            implementation="attribution_signals.reciprocal_uncertainty",
        ),
    ]

    # Computed in DualXDA pass but not exported in fast-pilot table (document for audit).
    specs.extend(
        [
            SignalFormulaSpec(
                signal="label_disagreement",
                label="Supporter label entropy",
                category="attribution_semantic",
                parts=(
                    *_topk_attribution_parts(top_k),
                    FormulaPart("supporter_labels", "Train labels at top-k positive supporters", "train_metadata"),
                ),
                operators=(
                    FormulaOperator("topk_positive", ("T_i",), {"k": top_k}),
                    FormulaOperator("normalized_entropy", ("supporter_labels",), {"normalize": "log(C)"}),
                ),
                formula="H_norm(labels of top-k positive supporters)",
                implementation="attribution_signals.normalized_entropy_from_labels",
                exported_in_fast_pilot=False,
            ),
            SignalFormulaSpec(
                signal="mass",
                label="Top-k attribution mass (raw)",
                category="attribution_structure",
                parts=attr_parts,
                operators=(*common_topk_ops, FormulaOperator("sum_abs", ("T_topk",), {})),
                formula="Σ_{i∈topk}|T_i|",
                implementation="attribution_signals.topk_influence_metrics",
                exported_in_fast_pilot=False,
            ),
        ]
    )

    return {s.signal: s for s in specs}


def build_signal_formula_manifest(
    *,
    top_k: int = 10,
    mc_passes: int = 30,
    mass_eps: float = 1e-8,
    eval_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    JSON-serializable manifest: formulas + eval protocol notes.

    ``eval_protocol`` should describe that eval pools are fixed for a sweep
    point (architecture-invariant), matching the disentanglement paper bench.
    """
    specs = fast_pilot_signal_formula_specs(
        top_k=top_k, mc_passes=mc_passes, mass_eps=mass_eps
    )
    exported = [k for k, s in specs.items() if s.exported_in_fast_pilot]
    return {
        "schema_version": 1,
        "eval_protocol": eval_protocol or {},
        "parameters": {"top_k": top_k, "mc_passes": mc_passes, "mass_eps": mass_eps},
        "exported_signals": exported,
        "signals": {name: spec.to_dict() for name, spec in specs.items()},
    }
