"""Checkpoint arsenal dialog and inline renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from uqlab.evaluation.pipeline.checkpoint_arsenal import (
    CheckpointArsenal,
    arsenal_cache_key,
    build_checkpoint_arsenal,
    collect_sweep_axis_options,
    filter_arsenal_sections,
)
from uqlab_orchestrator.experiment_registry import (
    experiments_root,
    load_experiment_config,
    workflow_from_run_yaml,
)
from uqlab.ui_components.results.model_group_row import render_model_section


def _get_experiments_dir(project_root: Path | None = None) -> Path:
    return experiments_root(project_root)


def _load_arsenal(
    experiments: List[Dict[str, Any]],
    *,
    project_root: Path | None = None,
    force_refresh: bool = False,
) -> CheckpointArsenal:
    experiments_dir = _get_experiments_dir(project_root)
    cache_key = arsenal_cache_key(experiments)
    cached = st.session_state.get("checkpoint_arsenal_cache")
    if (
        not force_refresh
        and isinstance(cached, dict)
        and cached.get("key") == cache_key
        and cached.get("arsenal") is not None
    ):
        return cached["arsenal"]
    arsenal = build_checkpoint_arsenal(experiments, experiments_dir)
    st.session_state["checkpoint_arsenal_cache"] = {"key": cache_key, "arsenal": arsenal}
    return arsenal


def apply_checkpoint_to_workflow(workflow: Dict[str, Any], experiment_id: str) -> None:
    """Pre-fill workflow from saved run config and enable resume mode."""
    cfg = load_experiment_config(experiment_id)
    if not cfg:
        st.warning(f"No config.yaml for {experiment_id[:8]}…")
        return
    restored = workflow_from_run_yaml(cfg)
    for key in ("dataset_config", "training_config", "uncertainty_config", "evaluation_config"):
        if key in restored:
            workflow[key] = restored[key]
    workflow["training_config"]["use_checkpoint"] = True
    workflow["training_config"]["checkpoint_id"] = experiment_id
    prior_epochs = int((cfg.get("training") or {}).get("epochs") or 12)
    workflow["training_config"]["prior_epochs"] = prior_epochs
    workflow["training_config"]["additional_epochs"] = workflow["training_config"].get(
        "additional_epochs", 10
    )
    workflow["training_config"]["epochs"] = prior_epochs + int(
        workflow["training_config"]["additional_epochs"]
    )
    st.session_state.pending_checkpoint_id = experiment_id
    st.session_state.step2_force_checkpoint_mode = True
    st.session_state.resume_source_label = experiment_id[:8]


def _render_filter_bar(
    arsenal: CheckpointArsenal,
    *,
    key_prefix: str,
) -> tuple[str, str, str, bool]:
    model_options = ["All"] + [s.model_label for s in arsenal.sections]
    axis_options = ["Any"] + list(collect_sweep_axis_options(arsenal))

    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    with c1:
        model_label = st.selectbox(
            "Model",
            model_options,
            key=f"{key_prefix}_filter_model",
        )
    with c2:
        sweep_axis = st.selectbox(
            "Sweep axis",
            axis_options,
            key=f"{key_prefix}_filter_axis",
        )
    with c3:
        search_text = st.text_input(
            "Search",
            placeholder="id, name, sweep value…",
            key=f"{key_prefix}_filter_search",
        )
    with c4:
        hide_singletons = st.checkbox(
            "Hide singletons",
            key=f"{key_prefix}_filter_hide_single",
        )
    return model_label, search_text, sweep_axis, hide_singletons


def render_arsenal_content(
    experiments: List[Dict[str, Any]],
    workflow: Dict[str, Any],
    *,
    key_prefix: str = "arsenal",
) -> Optional[str]:
    """Interactive arsenal body; returns experiment id if user picked a checkpoint."""
    arsenal = _load_arsenal(
        experiments,
        force_refresh=st.session_state.get(f"{key_prefix}_refresh", False),
    )
    st.session_state.pop(f"{key_prefix}_refresh", None)

    if arsenal.is_empty:
        st.info(
            "No resumable checkpoints yet (`results/checkpoint.pt`). "
            "Finish a training run with the backend in **production mode** "
            "(`./start_backend_prod.sh`)."
        )
        return None

    model_label, search_text, sweep_axis, hide_singletons = _render_filter_bar(
        arsenal, key_prefix=key_prefix
    )

    view = filter_arsenal_sections(
        arsenal,
        model_label=model_label,
        search_text=search_text,
        sweep_axis=sweep_axis,
        hide_singletons=hide_singletons,
    )

    st.caption(
        f"{view.n_checkpoints} checkpoint(s) shown · "
        f"{len(view.sections)} model type(s) · "
        "click a chip to load into Step 2 resume mode"
    )

    if view.is_empty:
        st.warning("No checkpoints match the current filters.")
        return None

    selected: Optional[str] = None
    for section in view.sections:
        picked = render_model_section(section, key_prefix=key_prefix)
        if picked:
            selected = picked

    if selected:
        apply_checkpoint_to_workflow(workflow, selected)
        st.success(f"Loaded checkpoint `{selected[:8]}…` into Step 2.")
    return selected


@st.dialog("Checkpoint arsenal", width="large")
def _arsenal_dialog(
    experiments: List[Dict[str, Any]],
    workflow: Dict[str, Any],
    key_prefix: str,
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", key=f"{key_prefix}_dialog_refresh"):
            st.session_state[f"{key_prefix}_refresh"] = True
            st.rerun()
    with col2:
        if st.button("Close", key=f"{key_prefix}_dialog_close"):
            st.session_state.pop(f"{key_prefix}_dialog_open", None)
            st.rerun()
    render_arsenal_content(
        experiments,
        workflow,
        key_prefix=f"{key_prefix}_dialog",
    )


def open_checkpoint_arsenal_dialog(
    experiments: List[Dict[str, Any]],
    workflow: Dict[str, Any],
    *,
    key_prefix: str = "arsenal",
) -> None:
    _arsenal_dialog(experiments, workflow, key_prefix)


def render_checkpoint_arsenal_inline(
    experiments: List[Dict[str, Any]],
    workflow: Dict[str, Any],
    *,
    key_prefix: str = "arsenal",
) -> Optional[str]:
    """Inline arsenal panel (Step 2.5 expanded view)."""
    if st.button("Refresh arsenal", key=f"{key_prefix}_inline_refresh"):
        st.session_state[f"{key_prefix}_refresh"] = True
        st.rerun()
    return render_arsenal_content(
        experiments,
        workflow,
        key_prefix=f"{key_prefix}_inline",
    )
