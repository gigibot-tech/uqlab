"""Sidebar helpers: paper launch + debug panel."""

from __future__ import annotations

from uqlab.ui_components.ui_debug import render_ui_debug_panel


def render_sidebar_footer_debug() -> None:
    """Debug checkboxes at bottom of sidebar."""
    render_ui_debug_panel(in_sidebar=True)
