"""Per-run vendor disentanglement score from ``results.pt`` (post-hoc, no re-training)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from uqlab.evaluation.benchmarks.disentangling.bridge_sweep import (
    score_bridge_pair_with_vendor_metric,
    score_bridge_pairs_from_results,
)
from uqlab.runtime_paths import resolve_experiment_results_dir
from uqlab.shared.config.signals import PREDICT_DISENTANGLING_NOTE, bridge_job_requirements
from uqlab.ui_components.ui_debug import ui_on

_BRIDGE_PRESETS: list[tuple[str, str]] = [
    ("paper", "Paper (expected_entropy + mutual_info)"),
    ("signal", "DualXDA (inverse_coherence + inverse_mass)"),
    ("signal_ek_fak", "EK-FAC"),
]


def _results_pt_path(experiment: dict[str, Any]) -> Path | None:
    eid = experiment.get("id")
    if not eid:
        return None
    results_dir = resolve_experiment_results_dir(
        str(eid),
        results_path=experiment.get("results_path"),
    )
    direct = results_dir / "results.pt"
    if direct.is_file():
        return direct
    return None


def render_disentanglement_score_panel(
    experiment: dict[str, Any],
    *,
    key_prefix: str,
) -> None:
    """Score one completed run using vendored ``calculate_disentanglement_error`` vectors."""
    if not ui_on("results_disentanglement_score"):
        return

    results_pt = _results_pt_path(experiment)
    if results_pt is None:
        st.caption("Disentanglement score unavailable — no `results.pt` on disk for this run.")
        return

    safe = f"{key_prefix}_{experiment.get('id', 'run')}"
    st.markdown("#### Disentanglement score (vendor metric)")
    st.caption(
        "Post-hoc score from uncertainty vectors in `results.pt`. "
        "Does not re-train. See signal requirements below."
    )

    preset_ids = [p[0] for p in _BRIDGE_PRESETS]
    preset_labels = {p[0]: p[1] for p in _BRIDGE_PRESETS}
    mode = st.selectbox(
        "Bridge preset",
        preset_ids,
        format_func=lambda m: preset_labels.get(m, m),
        key=f"{safe}_disent_preset",
    )

    if st.button("Compute disentanglement means", key=f"{safe}_disent_score_btn"):
        st.session_state.pop(f"{safe}_disent_score", None)
        st.session_state.pop(f"{safe}_disent_score_note", None)
        # Per-preset aleatoric/epistemic means read straight from results.pt (always works).
        try:
            rows = score_bridge_pairs_from_results(results_pt)
            st.session_state[f"{safe}_disent_rows"] = rows
        except (KeyError, FileNotFoundError, ValueError) as exc:
            st.session_state.pop(f"{safe}_disent_rows", None)
            st.error(f"Could not read uncertainty vectors: {exc}")
            reqs = bridge_job_requirements(*(_preset_signals(mode)))
            st.caption(f"Job requirements for this preset: {reqs}")

        # Vendor scalar DE is a sweep-correlation metric (refits at each noise /
        # dataset-size level) — it cannot be derived from a single run's vectors.
        try:
            score = score_bridge_pair_with_vendor_metric(results_pt, predict_mode=mode)
            st.session_state[f"{safe}_disent_score"] = float(score)
            st.session_state[f"{safe}_disent_mode"] = mode
        except Exception:  # noqa: BLE001 - vendor metric needs a refit sweep, not one run
            st.session_state[f"{safe}_disent_score_note"] = (
                "Scalar disentanglement **error** is a sweep-correlation metric "
                "(it refits the model across label-noise and dataset-size levels, like "
                "`disentanglement_error/examples/CIFAR10_it_demo.ipynb`). It can't be computed "
                "from one finished run — use a sweep, or read the per-region AUROC / means below."
            )

    score = st.session_state.get(f"{safe}_disent_score")
    if score is not None:
        used_mode = st.session_state.get(f"{safe}_disent_mode", mode)
        st.metric(
            "Disentanglement error",
            f"{score:.6f}",
            help=f"Preset: {preset_labels.get(used_mode, used_mode)}",
        )
    note = st.session_state.get(f"{safe}_disent_score_note")
    if note:
        st.info(note)

    rows = st.session_state.get(f"{safe}_disent_rows")
    if rows:
        df = pd.DataFrame(rows)
        display_cols = [
            c
            for c in (
                "preset",
                "aleatoric_signal",
                "epistemic_signal",
                "n_samples",
                "aleatorics_mean",
                "epistemics_mean",
                "error",
            )
            if c in df.columns
        ]
        if display_cols:
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    with st.expander("Signal requirements (predict_disentangling)", expanded=False):
        st.caption(PREDICT_DISENTANGLING_NOTE)


def _preset_signals(mode: str) -> tuple[str, str]:
    from uqlab.shared.config.signals import resolve_disentangling_signal_pair

    return resolve_disentangling_signal_pair(predict_mode=mode)
