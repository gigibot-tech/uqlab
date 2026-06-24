"""Sidebar quick launch — thin wrapper over shared launch panel."""

from __future__ import annotations

from typing import Any, Callable, Dict

from uqlab_orchestrator.campaign_id import new_campaign_timestamp
from uqlab_orchestrator.launch_types import LaunchReadiness
from uqlab.ui_components.progressive.launch_panel import render_launch_panel
from uqlab.ui_components.progressive.sweep_launch_cards import (
    RunBothFn,
    SweepLaunchCallbacks,
)
from uqlab.ui_components.ui_debug import ui_on
from uqlab_orchestrator.uncertainty.arm_builder import build_arm_workflow

LaunchFn = Callable[[bool], Dict[str, Any]]


def build_paper_profile_workflow(
    workflow: Dict[str, Any],
    profile: str,
    mode: str,
    *,
    under_train_sweep: Callable[[str], list] | None = None,
    label_noise_sweep: Dict[str, list] | None = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for ``build_arm_workflow``."""
    return build_arm_workflow(
        workflow,
        profile,
        mode,
        under_train_sweep=under_train_sweep,
        label_noise_sweep=label_noise_sweep,
    )


def render_sidebar_paper_launch(
    workflow: Dict[str, Any],
    *,
    readiness: LaunchReadiness,
    on_launch_primary: LaunchFn,
    on_launch_both: RunBothFn,
    on_apply_launch: Callable[[Dict[str, Any], bool], None] | None = None,
    key_prefix: str = "sb_launch",
) -> None:
    """Quick launch in sidebar — same preflight + controls as Step 5."""
    if not ui_on("sidebar_paper_launch"):
        return

    render_launch_panel(
        workflow,
        SweepLaunchCallbacks(
            on_launch_primary=on_launch_primary,
            on_launch_both=on_launch_both,
        ),
        readiness,
        layout="compact",
        key_prefix=key_prefix,
        on_apply_launch=on_apply_launch,
    )
