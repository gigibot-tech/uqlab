"""Shared Streamlit session keys for launch UI (sidebar + Step 5)."""

from __future__ import annotations

import streamlit as st

LAUNCH_AUTOSTART_KEY = "launch_autostart"


def ensure_launch_session() -> None:
    """Initialize launch-related session defaults."""
    if LAUNCH_AUTOSTART_KEY not in st.session_state:
        st.session_state[LAUNCH_AUTOSTART_KEY] = True


def get_launch_autostart() -> bool:
    """Global autostart flag — shared by sidebar, Step 5, and confirm dialog."""
    ensure_launch_session()
    return bool(st.session_state[LAUNCH_AUTOSTART_KEY])


def render_launch_autostart_checkbox(*, widget_key: str) -> bool:
    """
    Render autostart checkbox synced across surfaces.

    Each surface needs a unique Streamlit widget key; values sync via
    ``LAUNCH_AUTOSTART_KEY``.
    """
    ensure_launch_session()
    st.session_state[widget_key] = st.session_state[LAUNCH_AUTOSTART_KEY]
    checked = st.checkbox(
        "Start training immediately",
        key=widget_key,
    )
    st.session_state[LAUNCH_AUTOSTART_KEY] = checked
    return checked


__all__ = [
    "LAUNCH_AUTOSTART_KEY",
    "ensure_launch_session",
    "get_launch_autostart",
    "render_launch_autostart_checkbox",
]
