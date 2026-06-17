"""Sidebar paper sweep launch (Fig. 3 + Fig. 4 paired campaigns)."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict

import pandas as pd
import streamlit as st

from uqlab.ui_components.ui_debug import ui_on
from uqlab_orchestrator.config.validation_config import (
    FIXED_REGULAR_TRAIN_PER_CLASS,
    FIXED_UNDER_TRAIN_ALEATORIC_ARM,
)

LaunchFn = Callable[[bool], Dict[str, Any]]
SummaryFn = Callable[[str], Dict[str, Any]]


def new_campaign_timestamp() -> str:
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def build_paper_profile_workflow(
    workflow: Dict[str, Any],
    profile: str,
    mode: str,
    *,
    under_train_sweep: Callable[[str], list],
    label_noise_sweep: Dict[str, list],
) -> Dict[str, Any]:
    """Build workflow for one paper arm: ``under_train`` (Fig. 3) or ``noise`` (Fig. 4)."""
    paper = copy.deepcopy(workflow)
    uc = paper.setdefault("uncertainty_config", {})
    uc["sweep_enabled"] = True
    uc["sweep_mode"] = mode

    # "Mirroring" rule:
    # - Fig. 3 sweeps the epistemic axis (under-training) while keeping the aleatoric axis
    #   fixed to whatever Step 3 configured (on/off + custom noise if present).
    # - Fig. 4 sweeps the aleatoric axis (label noise) while keeping the epistemic axis fixed
    #   to whatever Step 3 configured (on/off + under-training if present).
    uc_src = workflow.get("uncertainty_config", {}) or {}
    ep_on = bool(uc_src.get("epistemic_enabled", False))
    alea_on = bool(uc_src.get("aleatoric_enabled", False))

    under_vals = under_train_sweep(mode)
    noise_vals = label_noise_sweep.get(mode, label_noise_sweep["quick"])
    paper.setdefault("dataset_config", {})["noise_type"] = "clean_label"

    if profile == "under_train":
        uc["sweep_kind"] = "dataset_size"
        # Sweep epistemic axis (Fig. 3), keep aleatoric fixed.
        uc["epistemic_enabled"] = True
        uc["aleatoric_enabled"] = alea_on
        uc["under_supported"] = uc_src.get("under_supported") or uc.get("under_supported") or "random:2"
        uc["under_train_per_class"] = uc_src.get("under_train_per_class") or 50
        uc["regular_train_per_class"] = uc_src.get("regular_train_per_class") or 300
        uc["epistemic_sweep_enabled"] = True
        uc["epistemic_sweep_values"] = under_vals
        uc["aleatoric_sweep_enabled"] = False
        uc["custom_noise_rate"] = (
            uc_src.get("custom_noise_rate") if alea_on else None
        )
    elif profile == "noise":
        uc["sweep_kind"] = "label_noise"
        uc["epistemic_enabled"] = ep_on
        uc["aleatoric_enabled"] = True
        if ep_on:
            uc["under_supported"] = uc_src.get("under_supported") or "random:2"
            uc["under_train_per_class"] = uc_src.get("under_train_per_class") or 50
            uc["regular_train_per_class"] = (
                uc_src.get("regular_train_per_class") or FIXED_REGULAR_TRAIN_PER_CLASS
            )
        else:
            # Paper Fig. 4 arm: fixed training budget, sweep only label noise.
            uc["under_supported"] = "random:2"
            uc["under_train_per_class"] = FIXED_UNDER_TRAIN_ALEATORIC_ARM
            uc["regular_train_per_class"] = FIXED_REGULAR_TRAIN_PER_CLASS
        uc["custom_noise_rate"] = None
        uc["aleatoric_sweep_enabled"] = True
        uc["aleatoric_sweep_values"] = noise_vals
        uc["epistemic_sweep_enabled"] = False
    else:
        raise ValueError(f"Unknown paper profile: {profile!r}")

    return paper


def render_sidebar_paper_launch(
    workflow: Dict[str, Any],
    *,
    on_launch_both: LaunchFn,
    on_launch_epis: LaunchFn,
    on_launch_alea: LaunchFn,
    aligned_sweep_summary: SummaryFn,
    key_prefix: str = "sb_paper",
) -> None:
    """Paper sweep toolbar in the sidebar (Run primary / Run both)."""
    if not ui_on("sidebar_paper_launch"):
        return

    st.markdown("### 🚀 Paper sweeps")
    st.caption(
        "Paired 1D sweeps · Fig. 3 (under-train) + Fig. 4 (label noise). "
        "If you enabled both uncertainty types in Step 3, the *missing* axis is mirrored (fixed) in each arm."
    )

    mode = workflow.get("uncertainty_config", {}).get("sweep_mode", "quick")
    summary = aligned_sweep_summary(mode)
    n_pts = len(summary["label_noise_percent"])

    auto_start = st.checkbox(
        "Start training immediately",
        value=True,
        key=f"{key_prefix}_autostart",
    )

    sweep_kind = workflow.get("uncertainty_config", {}).get("sweep_kind", "label_noise")
    primary_is_label_noise = sweep_kind == "label_noise"
    primary_label = "Fig. 4 (label noise)" if primary_is_label_noise else "Fig. 3 (under-train)"
    primary_launch = on_launch_alea if primary_is_label_noise else on_launch_epis

    col_primary, col_both = st.columns([2, 1])
    with col_primary:
        if st.button(
            f"▶️ Run {primary_label} ({n_pts} runs)",
            type="primary",
            key=f"{key_prefix}_primary",
            use_container_width=True,
        ):
            st.session_state.launch_result = primary_launch(auto_start)
            st.rerun()

    with col_both:
        if st.button(
            f"▶️ Run both ({2 * n_pts} runs)",
            key=f"{key_prefix}_both",
            use_container_width=True,
        ):
            st.session_state.launch_result = on_launch_both(auto_start)
            st.rerun()

    # Show current sweep mode (configured in Step 3)
    st.info(f"**Sweep mode**: `{mode}` (configure in Step 3)")
    grid = aligned_sweep_summary(mode)
    st.caption(f"Fig. 3 grid: `{grid['under_train_per_class']}`")
    st.caption(f"Fig. 4 grid: `{grid['label_noise_percent']}`")
