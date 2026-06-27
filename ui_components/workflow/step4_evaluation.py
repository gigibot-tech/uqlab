"""Step 4 — Evaluation setup (progressive workflow UI)."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from uqlab_orchestrator.config import TRAINING_CONFIG, get_sweep_mode
from uqlab_orchestrator.signal_facade import (
    default_selected_signals,
    get_signal_groups,
    normalize_signal_name,
)


def _default_signals(workflow: Dict[str, Any]) -> List[str]:
    eval_cfg = workflow.get("evaluation_config") or {}
    saved = eval_cfg.get("selected_signals")
    if saved:
        return [normalize_signal_name(s) for s in saved]
    tc = workflow.get("training_config") or {}
    dropout = float(tc.get("dropout") or 0.0)
    mc = int(eval_cfg.get("mc_passes") or TRAINING_CONFIG[get_sweep_mode(workflow)]["mc_passes"])
    base = [normalize_signal_name(s) for s in default_selected_signals()]
    if mc > 1 and dropout <= 0.0 and "mutual_info" in base:
        return [s for s in base if s != "mutual_info"]
    return base


def render_step4_evaluation(workflow: Dict[str, Any]) -> bool:
    """
    Render Step 4. Returns True if the progressive flow should stop here.
    """
    eval_cfg = workflow.get("evaluation_config") or {}
    sweep_mode = get_sweep_mode(workflow)
    preset_mc = TRAINING_CONFIG[sweep_mode]["mc_passes"]
    default_eval_per_group = int(eval_cfg.get("eval_per_group") or 100)
    default_mc = int(eval_cfg.get("mc_passes") or preset_mc)
    default_signals = _default_signals(workflow)

    resume_label = st.session_state.get("resume_source_label")
    if resume_label and workflow.get("training_config", {}).get("use_checkpoint"):
        st.info(
            f"**Inherited from checkpoint run** `{resume_label}` — "
            f"{default_mc} MC passes, {default_eval_per_group} samples/group. "
            "Change only if you want a different eval pool."
        )

    st.markdown("#### Evaluation Pool Configuration")
    dataset_stats = workflow.get("dataset_config", {}).get("stats", {})
    total_samples = dataset_stats.get("total_samples", 50_000)

    if workflow.get("uncertainty_config", {}).get("epistemic_enabled"):
        under_train = workflow["uncertainty_config"].get("under_train_per_class", 50)
        regular_train = workflow["uncertainty_config"].get("regular_train_per_class", 300)
        under_supported = workflow["uncertainty_config"].get("under_supported", "random:2")
        if str(under_supported).startswith("random:"):
            num_under = int(str(under_supported).split(":")[1])
        else:
            num_under = len(str(under_supported).split(","))
        num_regular = 10 - num_under
        estimated_train = (num_under * under_train) + (num_regular * regular_train)
    else:
        estimated_train = 2500

    available_for_eval = total_samples - estimated_train
    st.info(f"📊 Estimated available for evaluation: ~{available_for_eval:,} samples")

    col1, col2 = st.columns(2)
    with col1:
        eval_per_group = st.number_input(
            "Samples per evaluation group",
            min_value=50,
            max_value=500,
            value=default_eval_per_group,
            step=50,
            help="Number of samples to evaluate per group",
        )
        num_groups = 3 if workflow.get("uncertainty_config", {}).get("epistemic_enabled") else 2
        st.caption(f"Total evaluation samples: {eval_per_group * num_groups:,}")

    with col2:
        mc_passes = st.number_input(
            "MC Dropout passes",
            min_value=1,
            max_value=50,
            value=default_mc,
            help=f"Quick/full preset for mode `{sweep_mode}`: {preset_mc} passes",
        )

    dropout = float(workflow.get("training_config", {}).get("dropout") or 0.0)
    if mc_passes > 1 and dropout <= 0.0:
        st.warning(
            "MC passes > 1 with **dropout=0** — `mutual_info` will be zero. "
            "Enable dropout in Step 2 or reduce MC passes to 1."
        )

    st.markdown("#### Uncertainty Signals")
    st.caption(
        "DualXDA metrics run when selected. EK-FAC metrics (suffix ``_ek_fak``) require the "
        "optional ``kronfluence`` package and only run when their checkboxes are enabled."
    )
    groups = get_signal_groups()
    cols = st.columns(len(groups))
    selected: List[str] = []

    for col, (title, signal_ids) in zip(cols, groups.items()):
        group_key = title.lower().replace(" ", "_")
        with col:
            st.markdown(f"**{title}**")
            for sid in signal_ids:
                disabled = sid == "mutual_info" and mc_passes > 1 and dropout <= 0.0
                default_on = sid in default_signals and not disabled
                label = sid
                if sid.endswith("_ek_fak"):
                    label = f"{sid} (EK-FAC)"
                if st.checkbox(label, value=default_on, disabled=disabled, key=f"step4_sig_{group_key}_{sid}"):
                    selected.append(sid)

    all_signals = selected
    if not all_signals:
        st.warning("⚠️ Please select at least one uncertainty signal")
        st.stop()

    if st.button("✓ Review & Launch Experiment", type="primary", use_container_width=True):
        workflow["step4_complete"] = True
        workflow["evaluation_config"] = {
            "eval_per_group": int(eval_per_group),
            "mc_passes": int(mc_passes),
            "selected_signals": all_signals,
        }
        st.rerun()

    return True
