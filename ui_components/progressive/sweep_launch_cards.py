"""
Shared launch cards for Step 5 and sidebar — perspective-first (primary + Run both).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal

import streamlit as st

from uqlab_orchestrator.uncertainty import (
    LaunchAction,
    SINGLE_SWEEP_TARGET,
    SWEEP_BOTH_TARGET,
    launch_button_labels,
    launch_mirror_preview,
    resolve_launch_actions,
)
from uqlab.ui_components.progressive.launch_session import render_launch_autostart_checkbox
from uqlab.ui_components.ui_debug import ui_on

LaunchFn = Callable[[bool], Dict[str, Any]]
RunBothFn = Callable[[bool, str], Dict[str, Any]]

MirrorMode = Literal["mirror_point", "sweep_other_axis"]


@dataclass(frozen=True)
class SweepLaunchCallbacks:
    """Launch handlers wired from streamlit_app_progressive / orchestrator."""

    on_launch_primary: LaunchFn
    on_launch_both: RunBothFn


def _launch_dialog_key(key_prefix: str) -> str:
    return f"{key_prefix}_launch_dialog"


def _action_for_kind(actions: tuple[LaunchAction, ...], kind: str) -> LaunchAction:
    for action in actions:
        if action.kind == kind:
            return action
    return actions[0]


def _arms_for_dialog(
    preview: dict[str, Any],
    action: LaunchAction,
) -> list[dict[str, Any]]:
    target = preview["sweep_target"]
    if action.kind == "primary":
        if target == SWEEP_BOTH_TARGET:
            return list(preview.get("run_both_arms") or [])
        return [preview["primary"]]
    if target == SINGLE_SWEEP_TARGET:
        return list(preview.get("mirror_arms") or [])
    return list(
        preview.get("run_both_arms")
        or preview.get("mirror_arms")
        or []
    )


def _render_launch_dialog(
    workflow: Dict[str, Any],
    callbacks: SweepLaunchCallbacks,
    *,
    key_prefix: str,
    action: LaunchAction,
) -> None:
    preview = launch_mirror_preview(workflow)
    target = preview["sweep_target"]
    dialog_key = _launch_dialog_key(key_prefix)

    st.markdown(f"#### Confirm launch — {action.label}")
    
    # Show detailed configuration for each arm
    for arm in _arms_for_dialog(preview, action):
        st.markdown(f"**{arm['campaign']}** — {arm['n_runs']} runs")
        st.markdown(f"- **Swept:** {arm['swept']}")
        st.markdown(f"- **Fixed:** {arm['fixed']}")
        
        # Add prominent warning for fixed parameters
        uc = workflow.get("uncertainty_config", {}) or {}
        if arm.get("profile") == "noise":  # Aleatoric sweep (Fig. 4)
            under_train = uc.get("under_train_per_class")
            under_supported = uc.get("under_supported", "random:2")
            if under_train:
                st.warning(
                    f"⚠️ **Training budget fixed:** {under_train} samples/class for under-supported classes ({under_supported}). "
                    f"This will be the SAME across all {arm['n_runs']} runs in this sweep."
                )
        elif arm.get("profile") == "under_train":  # Epistemic sweep (Fig. 3)
            if uc.get("aleatoric_enabled"):
                rate = uc.get("custom_noise_rate")
                if rate is not None:
                    st.warning(
                        f"⚠️ **Label noise fixed:** {int(float(rate) * 100)}% noise will be applied to ALL {arm['n_runs']} runs in this sweep."
                    )

    auto_start = render_launch_autostart_checkbox(
        widget_key=f"{key_prefix}_dialog_autostart",
    )

    mirror_mode: MirrorMode = "mirror_point"
    if action.kind == "run_both" and target == SINGLE_SWEEP_TARGET:
        n_mirror = len(preview.get("mirror_arms") or [])
        choice = st.radio(
            "Mirror mode",
            ["mirror_point", "sweep_other_axis"],
            index=0,
            format_func=lambda m: (
                f"Mirror all ({n_mirror}) — 1D sweep per other perspective"
                if m == "mirror_point"
                else f"Single run + sweep all other ({n_mirror}) perspectives"
            ),
            key=f"{key_prefix}_mirror_mode",
        )
        mirror_mode = choice  # type: ignore[assignment]

    col_ok, col_cancel = st.columns(2)
    with col_ok:
        if st.button("Confirm launch", type="primary", key=f"{key_prefix}_dialog_confirm"):
            if action.kind == "primary":
                st.session_state.launch_result = callbacks.on_launch_primary(auto_start)
            else:
                st.session_state.launch_result = callbacks.on_launch_both(
                    auto_start,
                    mirror_mode,
                )
            st.session_state["scroll_to_results"] = True
            st.session_state.pop(dialog_key, None)
            st.rerun()
    with col_cancel:
        if st.button("Cancel", key=f"{key_prefix}_dialog_cancel"):
            st.session_state.pop(dialog_key, None)
            st.rerun()


def _open_launch_dialog(key_prefix: str, action_kind: str) -> None:
    st.session_state[_launch_dialog_key(key_prefix)] = action_kind
    st.rerun()


def render_sweep_launch_cards(
    workflow: Dict[str, Any],
    callbacks: SweepLaunchCallbacks,
    *,
    compact: bool = False,
    key_prefix: str = "launch",
    blocked: bool = False,
) -> None:
    """
    Render perspective-first launch UI: primary + Run both (always confirm dialog).

    ``compact=True`` stacks vertically for the sidebar.
    """
    if not ui_on("step5_launch_cards"):
        return

    if compact:
        st.markdown("### Run benchmark")
    else:
        st.markdown("#### Run benchmark")
        st.caption(launch_button_labels(workflow)["caption"])

    actions = resolve_launch_actions(workflow)
    dialog_key = _launch_dialog_key(key_prefix)
    pending_kind = st.session_state.get(dialog_key)

    if blocked:
        st.caption("Launch disabled until preflight issues are resolved.")
        return

    if pending_kind:
        _render_launch_dialog(
            workflow,
            callbacks,
            key_prefix=key_prefix,
            action=_action_for_kind(actions, str(pending_kind)),
        )
        return

    render_launch_autostart_checkbox(widget_key=f"{key_prefix}_autostart")

    if len(actions) == 1:
        action = actions[0]
        if st.button(
            action.label,
            type="primary",
            key=f"{key_prefix}_primary",
            use_container_width=compact,
        ):
            _open_launch_dialog(key_prefix, action.kind)
        return

    if compact:
        for action in actions:
            btn_type = "primary" if action.kind == "primary" else "secondary"
            if st.button(
                action.label,
                type=btn_type,  # type: ignore[arg-type]
                key=f"{key_prefix}_{action.kind}",
                use_container_width=True,
            ):
                _open_launch_dialog(key_prefix, action.kind)
    else:
        cols = st.columns(len(actions))
        for col, action in zip(cols, actions):
            with col:
                btn_type = "primary" if action.kind == "primary" else "secondary"
                if st.button(
                    action.label,
                    type=btn_type,  # type: ignore[arg-type]
                    key=f"{key_prefix}_{action.kind}",
                    use_container_width=True,
                ):
                    _open_launch_dialog(key_prefix, action.kind)
