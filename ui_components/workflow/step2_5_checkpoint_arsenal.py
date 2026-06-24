"""Step 2.5 — optional checkpoint arsenal review before uncertainty setup."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from uqlab.ui_components.results.checkpoint_arsenal_viz import (
    open_checkpoint_arsenal_dialog,
    render_checkpoint_arsenal_inline,
)


def render_step2_5_checkpoint_arsenal(
    experiments: List[Dict[str, Any]],
    workflow: Dict[str, Any],
    *,
    key_prefix: str = "step25",
) -> None:
    """
    Optional Step 2.5 — never blocks workflow progression.

    User can skip, expand inline, or open full dialog overlay.
    """
    skipped = st.session_state.get(f"{key_prefix}_skipped", False)
    expanded = st.session_state.get(f"{key_prefix}_expanded", False)

    st.markdown("### 🔖 Step 2.5: Checkpoint Arsenal (optional)")
    st.caption(
        "Resumable checkpoints grouped by model and sweep family. "
        "Click a chip to load into Step 2. Use `./start_backend_prod.sh` for stable training."
    )

    col_skip, col_review, col_dialog = st.columns(3)
    with col_skip:
        if st.button("Skip", key=f"{key_prefix}_skip", use_container_width=True):
            st.session_state[f"{key_prefix}_skipped"] = True
            st.session_state[f"{key_prefix}_expanded"] = False
            st.rerun()
    with col_review:
        if st.button("Review inline", key=f"{key_prefix}_expand", use_container_width=True):
            st.session_state[f"{key_prefix}_skipped"] = False
            st.session_state[f"{key_prefix}_expanded"] = True
            st.rerun()
    with col_dialog:
        if st.button("Open arsenal", key=f"{key_prefix}_dialog_btn", use_container_width=True):
            st.session_state[f"{key_prefix}_skipped"] = False
            open_checkpoint_arsenal_dialog(experiments, workflow, key_prefix=key_prefix)

    if skipped and not expanded:
        st.caption("Checkpoint arsenal skipped — continue to Step 3.")
        return

    if expanded:
        render_checkpoint_arsenal_inline(experiments, workflow, key_prefix=key_prefix)
