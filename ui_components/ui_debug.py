"""
UI debug toggles for the progressive Streamlit app.

Each toggle uses ``st.session_state["ui_debug__<key>"]`` as the checkbox widget key
(single source of truth). Parent keys gate children automatically.

Agent checklist: see ``.cursor/skills/ui-debug/SKILL.md``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import streamlit as st

UI_DEBUG_DEFAULTS_VERSION = 6
UI_DEBUG_DEFAULTS_VERSION_KEY = "ui_debug_defaults_v6"

# Results sub-toggles that stay off when results are enabled by default.
RESULTS_DEFAULTS_OFF: frozenset[str] = frozenset({
    "results_experiment_details",
    "results_training_data",
    "results_running_progress",
    "results_auto_refresh_ui",
    "results_auto_refresh_schedule",
})

# key -> (label, default_on)
UI_DEBUG_REGISTRY: Dict[str, Tuple[str, bool]] = {
    # Main
    "page_title": ("Page title & caption", True),
    "step1_dataset": ("Step 1 · Dataset", True),
    "step2_training": ("Step 2 · Training", True),
    "step2_5_checkpoint_arsenal": ("Step 2.5 · Checkpoint arsenal (optional)", True),
    "step3_uncertainty": ("Step 3 · Uncertainty", True),
    "step4_evaluation": ("Step 4 · Evaluation", True),
    "step5_launch": ("Step 5 · Review summary", True),
    "step5_launch_cards": ("Step 5 · launch buttons", True),
    "plot_probe_suggestions": ("Step 5 / Results · plot probe redo suggestions", True),
    "launch_result_banner": ("Launch result banner", True),
    "footer": ("Footer", True),
    "signal_validation": ("Signal validation mode", True),
    # Sidebar (only rendered components)
    "sidebar_paper_launch": ("Sidebar · quick launch (same as Step 5)", True),
    "sidebar_debug_panel": ("Sidebar · debug panel", True),
    # Results — on by default except per-run details + training data
    "results_section": ("Results · entire section", True),
    "results_live_status": ("Results · §1 live status", True),
    "results_sweep_analysis": ("Results · §2 analysis & sweeps", True),
    "results_sweep_campaigns": ("Results · §3 campaigns (merged into §2)", True),
    "results_sweep_groups": ("Results · sweep groups (legacy parent)", True),
    "results_local_presets": ("Results · local preset sweeps", True),
    "results_local_viz": ("Results · local validation viz", True),
    "results_status_metrics": ("Footer · experiment status counts", True),
    "results_running_progress": ("Results · running progress bars", False),
    "results_auto_refresh_ui": ("Results · auto-refresh controls", False),
    "results_auto_refresh_schedule": ("Results · 5s auto-rerun (JS)", False),
    "results_bulk_delete": ("Results · bulk delete", True),
    "results_bulk_recover": ("Results · bulk recover", True),
    "results_standalone_table": ("Results · standalone table (merged into §2)", True),
    "results_experiment_details": ("Results · per-run details + bar charts", False),
    "results_disentanglement_score": ("Results · per-run disentanglement score", True),
    "results_four_region": ("Results · four-region validation", True),
    "results_training_data": ("Results · §3 training data inspection", False),
    "results_batch_ui": ("Results · UI batch experiments", True),
}

# child -> parent (semantic tree)
UI_DEBUG_PARENT: Dict[str, str] = {
    "step5_launch_cards": "step5_launch",
    "plot_probe_suggestions": "step5_launch",
    "results_live_status": "results_section",
    "results_sweep_analysis": "results_section",
    "results_sweep_campaigns": "results_section",
    "results_sweep_groups": "results_section",
    "results_local_presets": "results_section",
    "results_local_viz": "results_section",
    "results_running_progress": "results_live_status",
    "results_auto_refresh_ui": "results_live_status",
    "results_auto_refresh_schedule": "results_live_status",
    "results_bulk_delete": "results_live_status",
    "results_bulk_recover": "results_live_status",
    "results_standalone_table": "results_section",
    "results_training_data": "results_section",
    "results_batch_ui": "results_section",
    "results_experiment_details": "results_sweep_analysis",
    "results_disentanglement_score": "results_experiment_details",
    "results_four_region": "results_sweep_analysis",
    # results_status_metrics: independent footer toggle (still turned off by Results off)
}

UI_DEBUG_CHILDREN: Dict[str, List[str]] = {}
for _child, _parent in UI_DEBUG_PARENT.items():
    UI_DEBUG_CHILDREN.setdefault(_parent, []).append(_child)

UI_DEBUG_SECTIONS: List[Tuple[str, List[str]]] = [
    ("Main workflow", [
        "page_title",
        "step1_dataset",
        "step2_training",
        "step2_5_checkpoint_arsenal",
        "step3_uncertainty",
        "step4_evaluation",
        "step5_launch",
        "step5_launch_cards",
        "plot_probe_suggestions",
        "launch_result_banner",
        "signal_validation",
        "footer",
    ]),
    ("Sidebar", [
        "sidebar_paper_launch",
        "sidebar_debug_panel",
    ]),
    (
        "Results (per-run details & training data off by default)",
        [
            "results_section",
            "results_live_status",
            "results_running_progress",
            "results_auto_refresh_ui",
            "results_auto_refresh_schedule",
            "results_bulk_delete",
            "results_bulk_recover",
            "results_sweep_analysis",
            "results_four_region",
            "results_sweep_campaigns",
            "results_sweep_groups",
            "results_standalone_table",
            "results_experiment_details",
            "results_disentanglement_score",
            "results_training_data",
            "results_status_metrics",
            "results_batch_ui",
            "results_local_presets",
            "results_local_viz",
        ],
    ),
]


def widget_key(component_key: str) -> str:
    return f"ui_debug__{component_key}"


def _is_results_key(component_key: str) -> bool:
    return component_key.startswith("results_")


def _apply_results_defaults() -> None:
    """Enable results section + most children; keep per-run and training data off."""
    for key in UI_DEBUG_REGISTRY:
        if not _is_results_key(key):
            continue
        st.session_state[widget_key(key)] = key not in RESULTS_DEFAULTS_OFF
    st.session_state["results_auto_refresh"] = False


def _descendants(parent_key: str) -> List[str]:
    """All nested children of a parent toggle."""
    out: List[str] = []
    stack = list(UI_DEBUG_CHILDREN.get(parent_key, []))
    while stack:
        key = stack.pop()
        out.append(key)
        stack.extend(UI_DEBUG_CHILDREN.get(key, []))
    return out


def _ancestors_enabled(component_key: str) -> bool:
    parent = UI_DEBUG_PARENT.get(component_key)
    while parent:
        if not _enabled_raw(parent):
            return False
        parent = UI_DEBUG_PARENT.get(parent)
    return True


def _force_results_off() -> None:
    """Set every results toggle and auto-refresh off."""
    for key in UI_DEBUG_REGISTRY:
        if _is_results_key(key):
            st.session_state[widget_key(key)] = False
    st.session_state["results_auto_refresh"] = False
    if "sidebar_auto_refresh" in st.session_state:
        st.session_state["sidebar_auto_refresh"] = False
    if "prog_auto_refresh" in st.session_state:
        st.session_state["prog_auto_refresh"] = False


def _cascade_disable(parent_key: str) -> None:
    """Uncheck parent and every descendant in session state."""
    st.session_state[widget_key(parent_key)] = False
    for desc in _descendants(parent_key):
        st.session_state[widget_key(desc)] = False
    sync_results_auto_refresh()


def _apply_parent_cascade_rules() -> None:
    """If a parent is off, force all descendants off before widgets render."""
    from streamlit.runtime.state import get_session_state

    for parent_key in UI_DEBUG_CHILDREN:
        if not _enabled_raw(parent_key):
            for desc in _descendants(parent_key):
                wkey = widget_key(desc)
                try:
                    session_state = get_session_state()
                    if wkey not in session_state._new_widget_state:
                        st.session_state[wkey] = False
                except Exception:
                    pass


def init_ui_debug() -> None:
    if "ui_debug" in st.session_state and not isinstance(
        st.session_state.get("ui_debug"), dict
    ):
        del st.session_state["ui_debug"]
    legacy = st.session_state.get("ui_debug")
    if isinstance(legacy, dict):
        for key, val in legacy.items():
            wkey = widget_key(key)
            if wkey not in st.session_state:
                st.session_state[wkey] = bool(val)
        del st.session_state["ui_debug"]

    migrated_version = st.session_state.get(UI_DEBUG_DEFAULTS_VERSION_KEY)

    for key, (_, default_on) in UI_DEBUG_REGISTRY.items():
        wkey = widget_key(key)
        if wkey not in st.session_state:
            st.session_state[wkey] = default_on

    if migrated_version != UI_DEBUG_DEFAULTS_VERSION:
        _apply_results_defaults()
        st.session_state[widget_key("step5_launch_cards")] = True
        st.session_state[widget_key("results_live_status")] = True
        st.session_state[widget_key("results_sweep_analysis")] = True
        st.session_state[widget_key("results_sweep_campaigns")] = True
        st.session_state[UI_DEBUG_DEFAULTS_VERSION_KEY] = UI_DEBUG_DEFAULTS_VERSION

    _apply_parent_cascade_rules()


def _enabled_raw(component_key: str) -> bool:
    default = UI_DEBUG_REGISTRY.get(component_key, (None, False))[1]
    return bool(st.session_state.get(widget_key(component_key), default))


def ui_on(component_key: str) -> bool:
    """True when this component and all ancestors are enabled."""
    init_ui_debug()
    if component_key not in UI_DEBUG_REGISTRY:
        return True

    chain: List[str] = []
    key: Optional[str] = component_key
    while key:
        chain.append(key)
        key = UI_DEBUG_PARENT.get(key)

    for k in chain:
        if not _enabled_raw(k):
            return False
    return True


def sync_results_auto_refresh() -> None:
    """Stop JS auto-rerun when results section or schedule is disabled."""
    _apply_parent_cascade_rules()
    if not _enabled_raw("results_section") or not _enabled_raw("results_auto_refresh_schedule"):
        st.session_state["results_auto_refresh"] = False
    # Heavy per-run details + 5s JS rerun is a bad combo — keep schedule off.
    if ui_on("results_experiment_details") and _enabled_raw("results_auto_refresh_schedule"):
        st.session_state[widget_key("results_auto_refresh_schedule")] = False
        st.session_state["results_auto_refresh"] = False


def set_all(enabled: bool) -> None:
    init_ui_debug()
    for key in UI_DEBUG_REGISTRY:
        st.session_state[widget_key(key)] = enabled
    if not enabled:
        st.session_state["results_auto_refresh"] = False
    else:
        sync_results_auto_refresh()


def set_results_off() -> None:
    """Disable all results toggles and stop auto-refresh."""
    init_ui_debug()
    _force_results_off()
    _apply_parent_cascade_rules()
    sync_results_auto_refresh()


def _make_parent_change_handler(parent_key: str):
    def _handler() -> None:
        if not st.session_state.get(widget_key(parent_key), False):
            for desc in _descendants(parent_key):
                st.session_state[widget_key(desc)] = False
        sync_results_auto_refresh()

    return _handler


def _render_debug_checkbox(component_key: str, label: str) -> None:
    wkey = widget_key(component_key)
    is_parent = component_key in UI_DEBUG_CHILDREN
    has_parent = component_key in UI_DEBUG_PARENT

    if has_parent and not _ancestors_enabled(component_key):
        st.session_state[wkey] = False

    if is_parent:
        st.checkbox(
            label,
            key=wkey,
            on_change=_make_parent_change_handler(component_key),
        )
        return

    disabled = has_parent and not _ancestors_enabled(component_key)
    if disabled:
        st.session_state[wkey] = False
    st.checkbox(label, key=wkey, disabled=disabled)
    if disabled:
        parent_key = UI_DEBUG_PARENT[component_key]
        parent_label = UI_DEBUG_REGISTRY[parent_key][0]
        st.caption(f"Enable parent: {parent_label}")


def render_ui_debug_panel(*, in_sidebar: bool = True) -> None:
    """Categorized expandable checkboxes — call once per run before gated UI."""
    init_ui_debug()
    _apply_parent_cascade_rules()
    sync_results_auto_refresh()

    container = st.sidebar if in_sidebar else st
    with container.expander("🐛 UI debug — components", expanded=False):
        st.caption(
            "Per-run details and training data are **off by default**. "
            "Parent toggles cascade to children. Use **Results off** to hide all results."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("All on", key="ui_dbg_all_on", use_container_width=True):
                set_all(True)
                st.session_state["results_auto_refresh"] = False
                st.rerun()
                return
        with c2:
            if st.button("Results off", key="ui_dbg_results_off", use_container_width=True):
                set_results_off()
                st.rerun()
                return
        with c3:
            if st.button("Results defaults", key="ui_dbg_results_defaults", use_container_width=True):
                _apply_results_defaults()
                st.session_state[UI_DEBUG_DEFAULTS_VERSION_KEY] = UI_DEBUG_DEFAULTS_VERSION
                st.rerun()
                return

        disabled = sum(1 for k in UI_DEBUG_REGISTRY if not _enabled_raw(k))
        if disabled:
            st.info(f"{disabled} component(s) disabled.")

        for section_title, keys in UI_DEBUG_SECTIONS:
            with st.expander(section_title, expanded=(section_title.startswith("Results"))):
                for key in keys:
                    if key not in UI_DEBUG_REGISTRY:
                        continue
                    label, _ = UI_DEBUG_REGISTRY[key]
                    parent = UI_DEBUG_PARENT.get(key)
                    if parent:
                        label = f"↳ {label}"
                    _render_debug_checkbox(key, label)

        sync_results_auto_refresh()

