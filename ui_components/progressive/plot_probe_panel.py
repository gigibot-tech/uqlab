"""UI for duplicate-gated plot probe outcomes and redo suggestions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import pandas as pd
import streamlit as st

from uqlab_orchestrator.plot_probe.outcome import PlotProbeResult
from uqlab_orchestrator.plot_probe.suggest import WorkflowSuggestion
from uqlab.ui_components.progressive.launch_session import get_launch_autostart
from uqlab.ui_components.ui_debug import ui_on

if TYPE_CHECKING:
    from uqlab_orchestrator.plot_probe.duplicate_gate import DuplicateOutcome


def render_probe_ok_banner(
    outcome: DuplicateOutcome,
    *,
    key_prefix: str,
) -> None:
    """Duplicate config with successful artifacts + completion + viz."""
    rep = outcome.group.representative_name
    pts = outcome.probe.points
    st.success(
        f"**Duplicate config** already has plottable results (`{rep}`, "
        f"{pts} plot point(s)). Skip relaunch and open **Results**."
    )
    if st.button(
        "Highlight matching run",
        key=f"{key_prefix}_highlight_dup",
        use_container_width=True,
    ):
        st.session_state.highlight_experiment_id = outcome.group.representative_id
        st.rerun()


def render_compact_probe_cta(
    suggestion: WorkflowSuggestion,
    *,
    key_prefix: str,
    on_launch: Callable[[Dict[str, Any], bool], None],
    title: str | None = None,
) -> None:
    """Preflight CTA — warning + apply-only or apply with launch (respects autostart)."""
    if title:
        st.markdown(f"**{title}**")
    st.warning(suggestion.reason)
    if suggestion.pool_note:
        st.caption(f"CIFAR-10 pool check: {suggestion.pool_note}")
    col_apply, col_launch = st.columns(2)
    with col_apply:
        if st.button(
            "Apply to workflow only",
            key=f"{key_prefix}_apply_only",
            use_container_width=True,
        ):
            st.session_state.workflow = suggestion.patched_workflow
            st.session_state["scroll_to_step5"] = True
            st.rerun()
    with col_launch:
        if st.button(
            "Apply suggested fix & launch",
            type="primary",
            key=f"{key_prefix}_apply_launch",
            use_container_width=True,
        ):
            st.session_state.workflow = suggestion.patched_workflow
            st.session_state["scroll_to_step5"] = True
            on_launch(suggestion.patched_workflow, get_launch_autostart())
            st.rerun()


def render_results_plot_status(
    probe: PlotProbeResult,
    *,
    source_label: str,
    key_prefix: str,
    insufficient_runs: bool = False,
) -> None:
    """Results §2 — read-only plot status; launch decisions live in Step 5."""
    if insufficient_runs:
        st.info(
            "Need at least **2 completed runs** with `results.pt` on disk to draw sweep plots. "
            "Refresh or wait for the sweep to finish."
        )
    elif not probe.ok:
        message = (probe.message or "Plot check failed.").strip()
        if len(message) > 160:
            message = message[:157] + "…"
        st.warning(
            f"Plot not available for `{source_label}`"
            + (f": {message}" if message else ".")
        )
    else:
        return

    if st.button(
        "Review launch in Step 5",
        key=f"{key_prefix}_goto_step5",
        use_container_width=True,
    ):
        st.session_state["scroll_to_step5"] = True
        st.rerun()


def render_redo_suggestion(
    suggestion: WorkflowSuggestion,
    *,
    key_prefix: str,
    on_apply: Callable[[Dict[str, Any]], None],
    on_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
    alt_suggestion: WorkflowSuggestion | None = None,
) -> None:
    """Failed probe — diff table plus Review / Apply / Apply & launch (no auto-launch)."""
    st.warning(suggestion.reason)

    if suggestion.diffs:
        rows = [
            {"Field": d.field, "Current": d.before, "Suggested": d.after}
            for d in suggestion.diffs
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No automatic field changes — review training/eval settings manually.")

    review_state_key = f"{key_prefix}_review_open"
    col_review, col_apply, col_launch = st.columns(3)
    with col_review:
        if st.button(
            "Review config changes",
            key=f"{key_prefix}_review",
            use_container_width=True,
        ):
            st.session_state[review_state_key] = not st.session_state.get(
                review_state_key, False
            )
            st.rerun()
    with col_apply:
        if st.button(
            "Apply",
            key=f"{key_prefix}_apply",
            use_container_width=True,
        ):
            on_apply(suggestion.patched_workflow)
            st.rerun()
    with col_launch:
        if on_launch and st.button(
            "Apply & launch",
            type="primary",
            key=f"{key_prefix}_apply_launch",
            use_container_width=True,
        ):
            on_apply(suggestion.patched_workflow)
            on_launch(suggestion.patched_workflow, True)
            st.rerun()

    if st.session_state.get(review_state_key):
        with st.expander("Patched workflow", expanded=True):
            st.json(suggestion.patched_workflow)

    if alt_suggestion:
        st.markdown("---")
        st.markdown("**Alternative — same epochs, larger data budget (test)**")
        st.info(alt_suggestion.reason)
        if alt_suggestion.pool_note:
            st.caption(f"CIFAR-10 pool check: {alt_suggestion.pool_note}")
        if alt_suggestion.diffs:
            alt_rows = [
                {"Field": d.field, "Current": d.before, "Suggested": d.after}
                for d in alt_suggestion.diffs
            ]
            st.dataframe(
                pd.DataFrame(alt_rows),
                use_container_width=True,
                hide_index=True,
            )
        col_alt_apply, col_alt_launch = st.columns(2)
        with col_alt_apply:
            if st.button(
                "Apply sample-size fix",
                key=f"{key_prefix}_alt_apply",
                use_container_width=True,
            ):
                on_apply(alt_suggestion.patched_workflow)
                st.rerun()
        with col_alt_launch:
            if on_launch and st.button(
                "Apply sample-size fix & launch",
                type="secondary",
                key=f"{key_prefix}_alt_apply_launch",
                use_container_width=True,
            ):
                on_apply(alt_suggestion.patched_workflow)
                on_launch(alt_suggestion.patched_workflow, True)
                st.rerun()


def render_duplicate_outcome_panel(
    outcome: DuplicateOutcome,
    workflow: Dict[str, Any],
    *,
    key_prefix: str,
    on_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Render ok banner or redo suggestion for one duplicate group."""
    if not ui_on("plot_probe_suggestions"):
        return

    def _apply(patched: Dict[str, Any]) -> None:
        st.session_state.workflow = patched

    if outcome.probe.ok:
        render_probe_ok_banner(outcome, key_prefix=key_prefix)
        return

    if outcome.suggestion:
        render_redo_suggestion(
            outcome.suggestion,
            key_prefix=key_prefix,
            on_apply=_apply,
            on_launch=on_launch,
            alt_suggestion=outcome.sample_size_suggestion,
        )
    else:
        st.warning(outcome.probe.message or "Plot check failed — no automatic suggestion.")


def render_probe_failure_panel(
    probe: PlotProbeResult,
    workflow: Dict[str, Any],
    *,
    source_label: str,
    key_prefix: str,
    on_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Results §2 — plot failed; suggest patched workflow from probe."""
    if not ui_on("plot_probe_suggestions"):
        return
    if probe.ok:
        return

    from uqlab_orchestrator.plot_probe.suggest import suggest_workflow_patch

    suggestion = suggest_workflow_patch(
        workflow,
        probe=probe,
        source_label=source_label,
    )

    def _apply(patched: Dict[str, Any]) -> None:
        st.session_state.workflow = patched
        st.session_state["scroll_to_step5"] = True

    render_redo_suggestion(
        suggestion,
        key_prefix=key_prefix,
        on_apply=_apply,
        on_launch=on_launch,
    )


def render_step5_duplicate_probes(
    workflow: Dict[str, Any],
    candidate_cfgs: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    *,
    project_root,
    key_prefix: str = "step5_probe",
    on_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Step 5 gate: duplicate match → outcome probe → panel."""
    if not ui_on("plot_probe_suggestions"):
        return
    if not candidate_cfgs or not experiments:
        return

    from uqlab_orchestrator.plot_probe import (
        assess_duplicate_outcome,
        find_duplicate_groups,
    )

    groups = find_duplicate_groups(
        candidate_cfgs,
        experiments,
        project_root=project_root,
    )
    if not groups:
        return

    st.markdown("#### Plot check (duplicate config)")
    for i, group in enumerate(groups[:3]):
        outcome = assess_duplicate_outcome(
            group,
            workflow,
            project_root=project_root,
        )
        with st.container():
            render_duplicate_outcome_panel(
                outcome,
                workflow,
                key_prefix=f"{key_prefix}_{i}",
                on_launch=on_launch,
            )
