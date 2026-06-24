"""Streamlit session initialization for the progressive workflow UI."""

from __future__ import annotations

import streamlit as st

from uqlab_orchestrator.config import default_workflow, merge_workflow_defaults
from uqlab.ui_components.ui_debug import init_ui_debug, sync_results_auto_refresh
from uqlab.ui_components.progressive.launch_session import ensure_launch_session


def ensure_workflow_initialized() -> None:
    if "workflow" not in st.session_state:
        st.session_state.workflow = default_workflow()
    else:
        st.session_state.workflow = merge_workflow_defaults(st.session_state.workflow)
    if "launch_result" not in st.session_state:
        st.session_state.launch_result = None
    if "results_auto_refresh" not in st.session_state:
        st.session_state.results_auto_refresh = False
    if "highlight_experiment_id" not in st.session_state:
        st.session_state.highlight_experiment_id = None
    if "pending_checkpoint_id" not in st.session_state:
        st.session_state.pending_checkpoint_id = None
    if "pending_checkpoint_label" not in st.session_state:
        st.session_state.pending_checkpoint_label = None
    if "resume_source_label" not in st.session_state:
        st.session_state.resume_source_label = None
    if "step2_force_checkpoint_mode" not in st.session_state:
        st.session_state.step2_force_checkpoint_mode = False
    if "experiment_selection_in_sidebar" not in st.session_state:
        st.session_state.experiment_selection_in_sidebar = True
    if "sidebar_campaign_experiments" not in st.session_state:
        st.session_state.sidebar_campaign_experiments = None
    init_ui_debug()
    sync_results_auto_refresh()
    ensure_launch_session()
