"""Streamlit helpers: resume training from a saved experiment checkpoint."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import streamlit as st

from uqlab_orchestrator.config import merge_workflow_defaults
from uqlab.ui_components.grouping.campaign_groups import group_experiments_for_results
from uqlab_orchestrator.experiment_registry import (
    build_resume_checkpoints_map,
    checkpoint_option_label,
    has_checkpoint,
    inherit_evaluation_from_config,
    load_experiment_config,
    workflow_from_run_yaml,
)


def _restore_sweep_uncertainty_from_experiments(
    workflow: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> None:
    """Re-enable sweep axes from completed runs in a campaign."""
    sweep_values_noise: List[float] = []
    sweep_values_under: List[int] = []
    for exp in experiments:
        cfg = load_experiment_config(str(exp["id"]))
        if not cfg:
            continue
        data = cfg.get("data") or {}
        alea = data.get("aleatoric_noise_percentage")
        if alea is not None and float(alea) >= 0:
            sweep_values_noise.append(float(alea))
        under = data.get("under_train_per_class")
        if under is not None:
            sweep_values_under.append(int(under))

    uc = workflow["uncertainty_config"]
    if sweep_values_noise and len(set(sweep_values_noise)) > 1:
        uc["sweep_enabled"] = True
        uc["sweep_kind"] = "label_noise"
        uc["aleatoric_sweep_values"] = sorted(set(sweep_values_noise))
        uc["aleatoric_enabled"] = True
        uc["epistemic_enabled"] = False
    elif sweep_values_under and len(set(sweep_values_under)) > 1:
        uc["sweep_enabled"] = True
        uc["sweep_kind"] = "dataset_size"
        uc["epistemic_sweep_values"] = sorted(set(sweep_values_under))
        uc["epistemic_enabled"] = True


def _default_campaign_index(
    campaign_groups: List[Dict[str, Any]],
    *,
    batch_id: Optional[str] = None,
    highlight_id: Optional[str] = None,
    pending_id: Optional[str] = None,
) -> int:
    if batch_id:
        for i, group in enumerate(campaign_groups):
            if group["id"] == batch_id:
                return i
    for needle in (pending_id, highlight_id):
        if not needle:
            continue
        for i, group in enumerate(campaign_groups):
            if any(str(e.get("id")) == str(needle) for e in group["experiments"]):
                return i
    return max(len(campaign_groups) - 1, 0)


def render_campaign_checkpoint_picker(
    experiments: List[Dict[str, Any]],
    workflow: Optional[Dict[str, Any]] = None,
    *,
    key_prefix: str = "step2_ckpt",
    pending_checkpoint_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Campaign-first checkpoint picker (matches sidebar smart selector grouping).

    Returns dict with keys: mode ('single'|'sweep'), pick, campaign_experiments,
    resume_map (sweep only).
    """
    campaign_groups, _config_groups = group_experiments_for_results(
        experiments,
        workflow=workflow,
    )
    if not campaign_groups:
        return None

    campaign_labels = [g["label"] for g in campaign_groups]
    default_campaign_idx = _default_campaign_index(
        campaign_groups,
        batch_id=st.session_state.get("uqlab_viz_batch_id"),
        highlight_id=st.session_state.get("highlight_experiment_id"),
        pending_id=pending_checkpoint_id,
    )
    campaign_choice = st.selectbox(
        "Campaign / batch",
        campaign_labels,
        index=default_campaign_idx,
        key=f"{key_prefix}_campaign",
        help="Same timestamp batches as the sidebar experiment selector.",
    )
    campaign_idx = campaign_labels.index(campaign_choice)
    campaign = campaign_groups[campaign_idx]
    campaign_exps = campaign["experiments"]
    st.session_state["uqlab_viz_batch_id"] = campaign["id"]

    completed = [e for e in campaign_exps if e.get("status") == "completed"]
    with_ckpt = [e for e in completed if has_checkpoint(str(e["id"]))]
    pick_pool = with_ckpt or completed
    if not pick_pool:
        st.warning("No completed runs in this campaign.")
        return None

    resume_map = build_resume_checkpoints_map(campaign_exps)
    sweep_mode = len(resume_map) >= 2
    resume_mode = "single"
    if sweep_mode:
        resume_mode = st.radio(
            "Resume scope",
            [
                f"Single run ({len(with_ckpt) or len(completed)} available)",
                f"Whole sweep ({len(resume_map)} checkpoints)",
            ],
            key=f"{key_prefix}_scope",
            horizontal=True,
        )
        if resume_mode.startswith("Whole"):
            return {
                "mode": "sweep",
                "pick": with_ckpt[0] if with_ckpt else completed[0],
                "campaign_experiments": campaign_exps,
                "resume_map": resume_map,
            }

    run_labels = [checkpoint_option_label(e) for e in pick_pool]
    default_run_idx = 0
    for needle in (pending_checkpoint_id, st.session_state.get("highlight_experiment_id")):
        if not needle:
            continue
        for i, exp in enumerate(pick_pool):
            if str(exp.get("id")) == str(needle):
                default_run_idx = i
                break
        else:
            continue
        break

    run_choice = st.selectbox(
        "Run in campaign",
        run_labels,
        index=default_run_idx,
        key=f"{key_prefix}_run",
        help="Checkpoints with on-disk weights only when available.",
    )
    pick = pick_pool[run_labels.index(run_choice)]
    return {
        "mode": "single",
        "pick": pick,
        "campaign_experiments": campaign_exps,
        "resume_map": None,
    }


def commit_step2_checkpoint_selection(
    workflow: Dict[str, Any],
    selection: Dict[str, Any],
    *,
    extra_epochs: int,
) -> None:
    """Apply a campaign picker result and mark Step 2 complete."""
    pick = selection["pick"]
    picked_id = str(pick["id"])
    picked_cfg = load_experiment_config(picked_id) or {}
    prior_epochs = int((picked_cfg.get("training") or {}).get("epochs") or 2)
    total_epochs = prior_epochs + int(extra_epochs)

    base_tc = dict(workflow.get("training_config") or {})
    if not base_tc.get("model_architecture"):
        restored = workflow_from_run_yaml(picked_cfg) if picked_cfg else {}
        base_tc.update(restored.get("training_config") or {})

    inherit_evaluation_from_config(workflow, picked_cfg)

    workflow["step2_complete"] = True
    workflow["training_config"] = {
        **base_tc,
        "use_checkpoint": True,
        "checkpoint_id": picked_id,
        "prior_epochs": prior_epochs,
        "additional_epochs": int(extra_epochs),
        "epochs": total_epochs,
    }

    resume_map = selection.get("resume_map")
    if selection.get("mode") == "sweep" and resume_map:
        _restore_sweep_uncertainty_from_experiments(
            workflow,
            selection.get("campaign_experiments") or [],
        )
        workflow["resume_checkpoints"] = resume_map
    else:
        workflow.pop("resume_checkpoints", None)

    st.session_state.pending_checkpoint_id = picked_id
    st.session_state.highlight_experiment_id = picked_id
    st.session_state.step2_force_checkpoint_mode = False


def apply_resume_to_workflow(
    exp_or_exps: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    extra_epochs: int = 10,
    mode: Literal["single", "sweep"] = "single",
) -> Optional[Dict[str, Any]]:
    """
    Restore workflow from one experiment or a sweep group.

    Sets checkpoint fields, inherits evaluation_config, and returns the workflow
    dict (caller should assign to session state).
    """
    if mode == "single":
        exp = exp_or_exps  # type: ignore[assignment]
        if not isinstance(exp, dict):
            return None
        exp_id = str(exp["id"])
        cfg = load_experiment_config(exp_id)
        if not cfg:
            return None

        workflow = merge_workflow_defaults(workflow_from_run_yaml(cfg))
        inherit_evaluation_from_config(workflow, cfg)
        prior_epochs = int((cfg.get("training") or {}).get("epochs") or 2)
        tc = workflow["training_config"]
        tc.update(
            {
                "use_checkpoint": True,
                "checkpoint_id": exp_id,
                "prior_epochs": prior_epochs,
                "additional_epochs": int(extra_epochs),
                "epochs": prior_epochs + int(extra_epochs),
            }
        )
        workflow.pop("resume_checkpoints", None)
        workflow["step2_complete"] = False
        workflow["step4_complete"] = True
        label = exp.get("name", exp_id[:8])
        return {
            "workflow": workflow,
            "pending_checkpoint_id": exp_id,
            "pending_checkpoint_label": label,
            "resume_source_label": label,
            "highlight_experiment_id": exp_id,
            "step2_force_checkpoint_mode": True,
        }

    experiments = exp_or_exps  # type: ignore[assignment]
    if not isinstance(experiments, list):
        return None
    completed = [e for e in experiments if e.get("status") == "completed"]
    if not completed:
        return None

    base_cfg = load_experiment_config(str(completed[0]["id"]))
    if not base_cfg:
        return None

    workflow = merge_workflow_defaults(workflow_from_run_yaml(base_cfg))
    inherit_evaluation_from_config(workflow, base_cfg)
    resume_map = build_resume_checkpoints_map(experiments)
    if not resume_map:
        return None

    _restore_sweep_uncertainty_from_experiments(workflow, experiments)

    prior_epochs = int((base_cfg.get("training") or {}).get("epochs") or 2)
    tc = workflow["training_config"]
    tc.update(
        {
            "use_checkpoint": True,
            "checkpoint_id": str(completed[0]["id"]),
            "prior_epochs": prior_epochs,
            "additional_epochs": int(extra_epochs),
            "epochs": prior_epochs + int(extra_epochs),
        }
    )
    workflow["resume_checkpoints"] = resume_map
    workflow["step2_complete"] = False
    workflow["step4_complete"] = True
    label = f"sweep ({len(resume_map)} points)"
    return {
        "workflow": workflow,
        "pending_checkpoint_id": str(completed[0]["id"]),
        "pending_checkpoint_label": label,
        "resume_source_label": label,
        "highlight_experiment_id": str(completed[-1]["id"]),
        "step2_force_checkpoint_mode": True,
    }


def _commit_resume_state(state: Dict[str, Any]) -> None:
    st.session_state.workflow = state["workflow"]
    st.session_state.pending_checkpoint_id = state["pending_checkpoint_id"]
    st.session_state.pending_checkpoint_label = state.get("pending_checkpoint_label")
    st.session_state.resume_source_label = state.get("resume_source_label")
    st.session_state.highlight_experiment_id = state["highlight_experiment_id"]
    st.session_state.step2_force_checkpoint_mode = state["step2_force_checkpoint_mode"]
    st.rerun()


def queue_single_checkpoint_resume(
    exp: Dict[str, Any],
    *,
    extra_epochs: int = 10,
) -> None:
    """Restore workflow from one experiment and open Step 2 in checkpoint mode."""
    exp_id = str(exp["id"])
    cfg = load_experiment_config(exp_id)
    if not cfg:
        st.error(f"No config.yaml found for experiment {exp_id[:8]}")
        return

    state = apply_resume_to_workflow(exp, extra_epochs=extra_epochs, mode="single")
    if not state:
        st.error(f"Could not restore workflow from experiment {exp_id[:8]}")
        return
    _commit_resume_state(state)


def queue_sweep_checkpoint_resume(
    experiments: List[Dict[str, Any]],
    *,
    extra_epochs: int = 10,
) -> None:
    """Restore workflow from sweep group; each point resumes its own checkpoint."""
    completed = [e for e in experiments if e.get("status") == "completed"]
    if not completed:
        st.warning("No completed runs in this sweep to resume from.")
        return

    state = apply_resume_to_workflow(experiments, extra_epochs=extra_epochs, mode="sweep")
    if not state:
        st.warning("No checkpoints found on disk for this sweep.")
        return
    _commit_resume_state(state)


def render_checkpoint_resume_controls(
    experiments: List[Dict[str, Any]],
    *,
    key_prefix: str,
    compact: bool = False,
) -> None:
    """UI block: pick a completed run and continue training on top."""
    completed = [e for e in experiments if e.get("status") == "completed"]
    if not completed:
        return

    st.markdown("#### 🔁 Continue training (checkpoint resume)")
    extra = st.number_input(
        "Additional epochs per resumed run",
        min_value=1,
        max_value=100,
        value=10,
        key=f"{key_prefix}_resume_extra_epochs",
    )

    if not compact and len(experiments) >= 2:
        if st.button(
            f"Resume whole sweep ({len(completed)} checkpoints)",
            key=f"{key_prefix}_resume_sweep",
            use_container_width=True,
        ):
            queue_sweep_checkpoint_resume(experiments, extra_epochs=int(extra))

    labels = [checkpoint_option_label(e) for e in completed]
    pick = st.selectbox(
        "Resume from checkpoint",
        options=labels,
        key=f"{key_prefix}_resume_pick",
        help="Loads that run's config into Steps 2–4 and trains additional epochs on top.",
    )
    if st.button("Apply → Step 2 (Use existing checkpoint)", key=f"{key_prefix}_resume_one"):
        idx = labels.index(pick)
        queue_single_checkpoint_resume(completed[idx], extra_epochs=int(extra))
