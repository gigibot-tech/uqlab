"""
Results Section Rendering

Single entry for the post–Step 5 results area in the progressive UI.
"""

from __future__ import annotations

from typing import Callable

import streamlit as st

from uqlab.ui_components.progressive.sweep_analysis_section import render_sweep_analysis_hub
from uqlab.ui_components.results.experiment_results_panel import render_experiment_results_panel
from uqlab.ui_components.ui_debug import sync_results_auto_refresh, ui_on


def render_progressive_results_section(
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    *,
    on_apply_launch: Callable[[dict, bool], None] | None = None,
) -> None:
    """Render the structured results block (one title, visible stub when gated off)."""
    if not ui_on("results_section"):
        sync_results_auto_refresh()
        st.markdown("---")
        st.caption(
            "Results hidden (UI debug). Open sidebar **UI debug → Results defaults** to show "
            "§1–§4 below Step 5."
        )
        return

    st.markdown("---")
    st.markdown("## Results")
    st.caption(
        f"Live data from `{api_base_url}`. "
        "Launch from **Step 5** or sidebar **Quick launch**. "
        "Toggle sub-panels in **UI debug** (sidebar footer)."
    )

    if st.session_state.pop("scroll_to_results", False):
        st.info("Launch submitted — review progress and plots in the sections below.")

    def _sweep_hub(groups: list, *, key_prefix: str = "sweep_hub") -> None:
        render_sweep_analysis_hub(
            groups,
            key_prefix=key_prefix,
            on_apply_launch=on_apply_launch,
        )

    st.session_state.results_auto_refresh = render_experiment_results_panel(
        api_base_url,
        get_headers_func,
        st.session_state.results_auto_refresh,
        empty_message=(
            "No experiments yet. Use **Launch** in Step 5 or sidebar, then refresh."
        ),
        key_prefix="prog_",
        sweep_analysis_renderer=_sweep_hub,
    )
