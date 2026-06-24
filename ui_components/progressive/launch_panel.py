"""Shared launch panel — preflight summary + launch controls for Step 5 and sidebar."""

from __future__ import annotations

from typing import Any, Callable, Dict, Literal, Optional

import pandas as pd
import streamlit as st

from uqlab_orchestrator.launch_types import LaunchReadiness
from uqlab.ui_components.progressive.plot_probe_panel import render_compact_probe_cta
from uqlab.ui_components.progressive.sweep_launch_cards import (
    SweepLaunchCallbacks,
    render_sweep_launch_cards,
)
from uqlab.ui_components.results.checkpoint_resume import (
    apply_resume_to_workflow,
    commit_step2_checkpoint_selection,
)
from uqlab.ui_components.ui_debug import ui_on

Layout = Literal["compact", "main"]


def _is_blocking(readiness: LaunchReadiness) -> bool:
    return bool(readiness.config_error or readiness.duplicate_ok)


def render_launch_summary(readiness: LaunchReadiness, *, compact: bool) -> None:
    """One-line aggregate status for launch preflight."""
    if readiness.config_error:
        st.error(readiness.summary)
        return
    if readiness.duplicate_ok:
        st.success(readiness.summary)
        return
    if readiness.resume or readiness.plot_probe:
        st.warning(readiness.summary)
        return
    if compact:
        st.caption(readiness.summary)
    else:
        st.info(readiness.summary)


def render_blocking_preflight(
    readiness: LaunchReadiness,
    workflow: Dict[str, Any],
    *,
    key_prefix: str,
) -> None:
    """Block launch — config error or duplicate OK only."""
    if readiness.config_error:
        st.markdown(readiness.config_error)
        if st.button(
            "Edit Step 3 configuration",
            key=f"{key_prefix}_goto_step3",
            use_container_width=True,
        ):
            workflow["step3_complete"] = False
            st.rerun()
        return

    if readiness.duplicate_ok:
        if st.button(
            "Open Results",
            key=f"{key_prefix}_open_results",
            use_container_width=True,
        ):
            st.session_state["scroll_to_results"] = True
            st.rerun()


def render_secondary_preflight(
    readiness: LaunchReadiness,
    workflow: Dict[str, Any],
    *,
    key_prefix: str,
    on_apply_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Optional recovery below launch row — resume or plot-probe suggestion."""
    if readiness.resume:
        offer = readiness.resume
        label = (
            f"Resume whole sweep +{offer.delta_epochs} epochs "
            f"({offer.n_runs} checkpoints)"
            if offer.n_runs >= 2
            else f"Resume +{offer.delta_epochs} epochs"
        )
        if st.button(label, key=f"{key_prefix}_resume_sweep", use_container_width=True):
            if offer.n_runs >= 2:
                state = apply_resume_to_workflow(
                    offer.campaign_experiments,
                    extra_epochs=int(offer.delta_epochs),
                    mode="sweep",
                )
                if state:
                    wf = state["workflow"]
                    wf["step2_complete"] = True
                    wf["step4_complete"] = True
                    st.session_state.workflow = wf
                    st.session_state.pending_checkpoint_id = state["pending_checkpoint_id"]
                    st.session_state.resume_source_label = state.get("resume_source_label")
                    st.rerun()
            else:
                exp = offer.campaign_experiments[0]
                commit_step2_checkpoint_selection(
                    workflow,
                    {
                        "mode": "single",
                        "pick": exp,
                        "campaign_experiments": [exp],
                        "resume_map": None,
                    },
                    extra_epochs=int(offer.delta_epochs),
                )
                st.session_state.workflow = workflow
                st.session_state.resume_source_label = exp.get("name", str(exp["id"])[:8])
                st.rerun()

    if readiness.plot_probe and on_apply_launch:
        render_compact_probe_cta(
            readiness.plot_probe,
            key_prefix=f"{key_prefix}_probe",
            on_launch=on_apply_launch,
        )
        if readiness.plot_probe_sample_size:
            render_compact_probe_cta(
                readiness.plot_probe_sample_size,
                key_prefix=f"{key_prefix}_probe_sample",
                on_launch=on_apply_launch,
                title="Alternative — same epochs, larger data budget (test)",
            )

    if ui_on("plot_probe_suggestions") and readiness.candidate_cfgs:
        with st.expander("Preflight detail (debug)", expanded=False):
            st.caption(f"{readiness.n_runs} launch candidate(s)")
            if readiness.plot_probe and readiness.plot_probe.diffs:
                rows = [
                    {
                        "Field": d.field,
                        "Current": d.before,
                        "Suggested": d.after,
                    }
                    for d in readiness.plot_probe.diffs
                ]
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )


def render_launch_controls(
    workflow: Dict[str, Any],
    callbacks: SweepLaunchCallbacks,
    readiness: LaunchReadiness,
    *,
    layout: Layout,
    key_prefix: str,
) -> None:
    """Autostart checkbox + primary / Run both launch row."""
    blocked = _is_blocking(readiness)
    render_sweep_launch_cards(
        workflow,
        callbacks,
        compact=(layout == "compact"),
        key_prefix=key_prefix,
        blocked=blocked,
    )


def render_launch_panel(
    workflow: Dict[str, Any],
    callbacks: SweepLaunchCallbacks,
    readiness: LaunchReadiness,
    *,
    layout: Layout,
    key_prefix: str,
    on_apply_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Shared entry for Step 5 (main) and sidebar (compact)."""
    if not ui_on("step5_launch_cards") and layout == "compact":
        return

    render_launch_summary(readiness, compact=(layout == "compact"))

    resume_map = workflow.get("resume_checkpoints") or {}
    if resume_map and layout == "main":
        tc = workflow.get("training_config") or {}
        st.caption(
            f"Sweep checkpoint resume — {len(resume_map)} point(s), "
            f"+{tc.get('additional_epochs', '?')} epochs each."
        )

    if _is_blocking(readiness):
        render_blocking_preflight(readiness, workflow, key_prefix=key_prefix)
        return

    if layout == "main":
        st.markdown("---")
    render_launch_controls(
        workflow,
        callbacks,
        readiness,
        layout=layout,
        key_prefix=key_prefix,
    )

    if readiness.resume or readiness.plot_probe:
        with st.expander("Suggested fix (optional)", expanded=False):
            render_secondary_preflight(
                readiness,
                workflow,
                key_prefix=key_prefix,
                on_apply_launch=on_apply_launch,
            )


__all__ = ["render_launch_panel"]
