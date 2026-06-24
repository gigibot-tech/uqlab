"""Step 2 — Training setup (progressive workflow UI)."""

from __future__ import annotations

from typing import Any, Callable, Dict

import requests
import streamlit as st

from uqlab.ui_components.progressive.api_client import fetch_experiments
from uqlab.ui_components.results.checkpoint_arsenal_viz import open_checkpoint_arsenal_dialog
from uqlab.ui_components.results.checkpoint_resume import (
    commit_step2_checkpoint_selection,
    render_campaign_checkpoint_picker,
)
from uqlab_orchestrator.experiment_registry import load_experiment_config


def render_step2_training(
    workflow: Dict[str, Any],
    *,
    api_base_url: str,
    get_headers: Callable[[], dict],
) -> bool:
    """
    Render Step 2. Returns True if the progressive flow should stop here.
    """
    pending_id = st.session_state.get("pending_checkpoint_id")
    resume_label = st.session_state.get("resume_source_label")
    if pending_id or st.session_state.get("step2_force_checkpoint_mode"):
        cfg = load_experiment_config(str(pending_id)) if pending_id else None
        prior_epochs = int((cfg or {}).get("training", {}).get("epochs") or 2)
        extra = int(workflow.get("training_config", {}).get("additional_epochs") or 10)
        src = resume_label or str(pending_id)[:8]
        st.info(
            f"**Resume mode** — continuing from **{src}** "
            f"(trained **{prior_epochs}** epochs). "
            f"Set additional epochs below, then continue."
        )

    default_mode_idx = 1 if (pending_id or st.session_state.get("step2_force_checkpoint_mode")) else 0
    training_mode = st.radio(
        "Training mode",
        ["Train new model", "Use existing checkpoint"],
        index=default_mode_idx,
        help="Train a new model or load a pre-trained checkpoint",
    )

    if training_mode == "Train new model":
        st.session_state.step2_force_checkpoint_mode = False
        st.markdown("#### Model Configuration")

        col1, col2 = st.columns(2)
        with col1:
            arch_options = ["dinov2-small", "dinov2-base", "resnet18", "resnet50", "pixel-mlp"]
            saved_arch = workflow["training_config"].get("model_architecture", "resnet18")
            arch_index = arch_options.index(saved_arch) if saved_arch in arch_options else 2
            model_arch = st.selectbox(
                "Model architecture",
                arch_options,
                index=arch_index,
                help=(
                    "DINOv2 models are pre-trained vision transformers; "
                    "pixel-mlp is a small flatten→hidden→classes MLP trained "
                    "end-to-end on raw pixels (with MC dropout)."
                ),
            )
            hidden_dim = st.number_input(
                "Hidden dimension",
                min_value=64,
                max_value=1024,
                value=workflow["training_config"].get("hidden_dim", 256),
                step=64,
            )
            saved_dropout = float(workflow["training_config"].get("dropout", 0.0))
            dropout = st.slider("Dropout rate", 0.0, 0.5, saved_dropout, 0.05)

        with col2:
            epochs = st.number_input(
                "Training epochs",
                min_value=1,
                max_value=100,
                value=int(workflow["training_config"].get("epochs", 12)),
            )
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=0.1,
                value=float(workflow["training_config"].get("learning_rate", 0.001)),
                format="%.4f",
            )
            batch_size = st.selectbox("Batch size", [64, 128, 256, 512], index=2)

        if st.button("✓ Continue to Uncertainty Configuration", type="primary", use_container_width=True):
            workflow["step2_complete"] = True
            workflow["training_config"] = {
                "use_checkpoint": False,
                "model_architecture": model_arch,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
            workflow.pop("resume_checkpoints", None)
            st.session_state.pending_checkpoint_id = None
            st.session_state.resume_source_label = None
            st.session_state.step2_force_checkpoint_mode = False
            st.rerun()
        return True

    st.markdown("#### Select Checkpoint")
    if st.button("Browse all checkpoints…", key="step2_browse_arsenal"):
        try:
            experiments_for_arsenal = fetch_experiments(api_base_url, get_headers)
            open_checkpoint_arsenal_dialog(experiments_for_arsenal, workflow, key_prefix="step2_arsenal")
        except requests.exceptions.RequestException as exc:
            st.error(f"Failed to load experiments: {exc}")
    extra_epochs = st.number_input(
        "Additional epochs to train on top",
        min_value=1,
        max_value=100,
        value=int(workflow.get("training_config", {}).get("additional_epochs") or 10),
        help="Loads weights from the checkpoint, then trains this many more epochs.",
    )

    try:
        experiments = fetch_experiments(api_base_url, get_headers)
        selection = render_campaign_checkpoint_picker(
            experiments,
            workflow,
            key_prefix="step2_ckpt",
            pending_checkpoint_id=str(pending_id) if pending_id else None,
        )

        if selection:
            pick = selection["pick"]
            picked_id = str(pick["id"])
            picked_cfg = load_experiment_config(picked_id) or {}
            prior_epochs = int((picked_cfg.get("training") or {}).get("epochs") or 2)
            total_epochs = prior_epochs + int(extra_epochs)

            if selection["mode"] == "sweep":
                resume_map = selection["resume_map"] or {}
                st.caption(
                    f"Will train **{extra_epochs}** more epochs per sweep point "
                    f"({prior_epochs} → **{total_epochs}** total) using **{len(resume_map)}** checkpoints."
                )
                st.success(
                    f"Sweep resume: **{len(resume_map)}** checkpoint(s) mapped "
                    f"to sweep points — each run continues from its own checkpoint."
                )
            else:
                st.caption(
                    f"Will train **{extra_epochs}** more epochs "
                    f"({prior_epochs} → **{total_epochs}** total) using weights from this run."
                )

            if st.button("✓ Continue to Uncertainty Configuration", type="primary", use_container_width=True):
                commit_step2_checkpoint_selection(
                    workflow,
                    selection,
                    extra_epochs=int(extra_epochs),
                )
                st.rerun()
        elif not [e for e in experiments if e.get("status") == "completed"]:
            st.warning("No completed experiments found. Please train a new model.")

    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch experiments: {exc}")
        st.info("Falling back to training new model...")

    return True
