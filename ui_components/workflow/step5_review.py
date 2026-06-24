"""Step 5 — Review and launch (progressive workflow UI)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from uqlab_orchestrator.launch_types import LaunchReadiness
from uqlab.runtime_paths import repository_root
from uqlab.ui_components.progressive.sweep_launch_cards import SweepLaunchCallbacks


def render_step5_review(
    workflow: Dict[str, Any],
    *,
    readiness: LaunchReadiness,
    launch_callbacks: SweepLaunchCallbacks,
    on_apply_launch: Optional[Callable[[Dict[str, Any], bool], None]] = None,
) -> None:
    """Render Step 5 review panel — thin wrapper around shared launch panel."""
    st.markdown("### Step 5: Review & launch")
    st.caption(
        "For sweeps and long runs, start the backend with "
        "`cd backend && ./start_backend_prod.sh` (no auto-reload) so training "
        "finishes and writes `checkpoint.pt`."
    )

    tc = workflow.get("training_config") or {}
    if tc.get("use_checkpoint"):
        src = st.session_state.get("resume_source_label") or tc.get("checkpoint_id", "?")
        extra = tc.get("additional_epochs", "?")
        prior = tc.get("prior_epochs", "?")
        st.success(
            f"**Checkpoint resume** from `{src}` — train **{extra}** more epochs "
            f"({prior} → {tc.get('epochs')} total)."
        )
        eval_cfg = workflow.get("evaluation_config") or {}
        st.caption(
            f"Eval: {eval_cfg.get('eval_per_group', '?')} samples/group, "
            f"{eval_cfg.get('mc_passes', '?')} MC passes."
        )

    with st.expander("View full workflow state", expanded=False):
        st.json(workflow)

    with st.expander("Methods schematic", expanded=False):
        from uqlab.ui_components.visualization.thesis.thesis_diagram_viz import (
            render_thesis_diagram_panel,
        )

        render_thesis_diagram_panel(
            workflow=workflow,
            project_root=repository_root(),
            key_prefix="step5_thesis",
            default_symbolic=True,
        )

    dropout = float(tc.get("dropout") or 0.0)
    mc = int((workflow.get("evaluation_config") or {}).get("mc_passes") or 0)
    if mc > 1 and dropout <= 0.0:
        st.warning(
            "**MC dropout note:** dropout is 0 — epistemic MC signals (`mutual_info`) "
            "will not vary between passes."
        )

    from uqlab.ui_components.progressive.launch_panel import render_launch_panel

    render_launch_panel(
        workflow,
        launch_callbacks,
        readiness,
        layout="main",
        key_prefix="step5",
        on_apply_launch=on_apply_launch,
    )
