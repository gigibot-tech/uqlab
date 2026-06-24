"""
Launch Results Rendering

Components for displaying experiment launch results and existing run suggestions.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from uqlab.ui_components.grouping.campaign_groups import group_experiments_for_results
from uqlab.ui_components.results.checkpoint_resume import (
    apply_resume_to_workflow,
    commit_step2_checkpoint_selection,
)
from uqlab_orchestrator.experiment_registry import (
    find_duplicate_runs,
    suggest_epoch_continuation,
)
from uqlab.ui_components.ui_debug import ui_on


def render_launch_result() -> None:
    """
    Show last launch outcome (persists across Streamlit reruns).
    
    Displays success/error messages, created experiments, and start status.
    Gated by ui_on("launch_result_banner") debug flag.
    """
    if not ui_on("launch_result_banner"):
        return
    result = st.session_state.get("launch_result")
    if not result:
        return

    if not result.get("ok"):
        st.error(f"Launch failed: {result.get('error', 'Unknown error')}")
        if result.get("detail"):
            with st.expander("Error details"):
                st.code(result["detail"])
        return

    n = result.get("n_created", 1)
    axis = result.get("sweep_axis", "single")
    if n > 1:
        st.success(f"**Fast sweep launched:** {n} experiments ({axis}).")
        if result.get("errors"):
            st.warning("Some points failed to create — see details below.")
            with st.expander("Creation errors"):
                st.code("\n".join(result["errors"]))
        with st.expander("Created runs", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "name": r["name"],
                            "id": r["id"],
                            "under_train": r.get("under_train"),
                            "noise_%": r.get("aleatoric_noise_percentage"),
                            "started": r.get("started"),
                        }
                        for r in result.get("created_runs", [])
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
    else:
        created = result["created"]
        st.success(f"Experiment **{created.get('name', '?')}** created (ID `{result['experiment_id']}`).")

    if result.get("n_started", 0) == n and n > 0:
        st.info(
            "Training started for all runs. Scroll to **Results** below (§1 live status, "
            "§2 sweep plots when ≥2 runs complete)."
        )
        if st.session_state.get("scroll_to_results"):
            st.caption("↓ Results section is directly below Step 5.")
    elif result.get("n_failed_start"):
        st.warning(
            f"{result['n_failed_start']} run(s) failed to start — use **Queued → Start** above "
            "or fix the backend error."
        )
        if result.get("start_error"):
            st.code(result["start_error"])
    elif not result.get("started"):
        st.info("Experiments saved **without** starting training (checkbox was off).")


def _exp_by_id(experiments: List[Dict[str, Any]], exp_id: str) -> Optional[Dict[str, Any]]:
    for exp in experiments:
        if str(exp.get("id")) == str(exp_id):
            return exp
    return None


def render_existing_run_suggestions(
    *,
    candidate_cfgs: List[Dict[str, Any]],
    desired_epochs: int,
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    project_root: Path,
    workflow: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Before launching, warn if there are already runs with the same setup (except epochs).
    Offers one-click checkpoint resume when a matching run has fewer epochs than desired.
    """
    if not candidate_cfgs:
        return

    try:
        from uqlab.ui_components.progressive.api_client import fetch_experiments

        experiments = fetch_experiments(api_base_url, get_headers_func)
    except Exception:
        return

    hits = find_duplicate_runs(
        candidate_cfgs,
        experiments,
        project_root=project_root,
        desired_epochs=desired_epochs,
    )
    if not hits:
        return

    continuations = suggest_epoch_continuation(
        hits,
        desired_epochs,
        project_root=project_root,
    )
    if not continuations:
        return

    st.warning(
        "Found existing run(s) with the **same configuration** at fewer epochs. "
        f"Target is **{desired_epochs}** total — resume from checkpoint instead of retraining from scratch."
    )

    resumable = [c for c in continuations if c.get("can_resume")]
    if resumable:
        sweep_candidates = [
            _exp_by_id(experiments, c["existing_id"])
            for c in resumable
            if _exp_by_id(experiments, c["existing_id"])
        ]
        sweep_candidates = [e for e in sweep_candidates if e]
        if len(sweep_candidates) >= 2:
            _, config_groups = group_experiments_for_results(
                sweep_candidates,
                workflow=workflow,
            )
            campaign_exps = sweep_candidates
            if config_groups:
                campaign_exps = config_groups[0]["experiments"]
            delta = resumable[0]["delta_epochs"]
            if st.button(
                f"Resume whole sweep +{delta} epochs ({len(sweep_candidates)} checkpoints)",
                use_container_width=True,
                key="resume_sweep_delta",
            ):
                state = apply_resume_to_workflow(
                    campaign_exps,
                    extra_epochs=int(delta),
                    mode="sweep",
                )
                if state and workflow is not None:
                    wf = state["workflow"]
                    wf["step2_complete"] = True
                    wf["step4_complete"] = True
                    st.session_state.workflow = wf
                    st.session_state.pending_checkpoint_id = state["pending_checkpoint_id"]
                    st.session_state.resume_source_label = state.get("resume_source_label")
                    st.rerun()

    for i, cont in enumerate(resumable[:5]):
        exp = _exp_by_id(experiments, cont["existing_id"])
        if not exp:
            continue
        delta = int(cont["delta_epochs"])
        cols = st.columns([2, 1])
        with cols[0]:
            st.caption(
                f"`{cont['existing_name']}` — {cont['existing_epochs']}ep → "
                f"{desired_epochs}ep (+{delta})"
            )
        with cols[1]:
            if st.button(
                f"Resume +{delta}",
                key=f"resume_delta_{cont['existing_id']}_{i}",
                use_container_width=True,
            ):
                if workflow is not None:
                    selection = {
                        "mode": "single",
                        "pick": exp,
                        "campaign_experiments": [exp],
                        "resume_map": None,
                    }
                    commit_step2_checkpoint_selection(
                        workflow,
                        selection,
                        extra_epochs=delta,
                    )
                    st.session_state.workflow = workflow
                    st.session_state.resume_source_label = exp.get("name", str(exp["id"])[:8])
                    st.rerun()

    with st.expander("Existing matching runs", expanded=False):
        for h in hits[:10]:
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.write(f"`{h['existing_name']}`")
            with cols[1]:
                st.write(str(h.get("existing_status") or "—"))
            with cols[2]:
                st.write(f"epochs={h['existing_epochs']}")
            with cols[3]:
                if st.button("Open", key=f"open_existing_{h['existing_id']}"):
                    st.session_state.highlight_experiment_id = h["existing_id"]
                    st.rerun()
