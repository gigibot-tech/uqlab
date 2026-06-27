"""Shared Streamlit styling helpers for the progressive Results UI."""

from __future__ import annotations

import html
import streamlit as st

_CSS_INJECTED_KEY = "_uqlab_results_css_injected"


def inject_results_css() -> None:
    """Inject neutral pill/badge + section card styles once per session."""
    if st.session_state.get(_CSS_INJECTED_KEY):
        return
    st.session_state[_CSS_INJECTED_KEY] = True
    st.markdown(
        """
        <style>
        .uqlab-section {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 8px;
            padding: 0.75rem 1rem 0.5rem 1rem;
            margin: 0.5rem 0 1rem 0;
            background: rgba(128,128,128,0.04);
        }
        .uqlab-section-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin: 0 0 0.25rem 0;
            line-height: 1.3;
        }
        .uqlab-section-sub {
            font-size: 0.82rem;
            color: rgba(128,128,128,0.95);
            margin: 0 0 0.5rem 0;
            line-height: 1.4;
        }
        .uqlab-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.25rem 0 0.5rem 0;
        }
        .uqlab-pill {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 500;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            border: 1px solid rgba(128,128,128,0.35);
            background: rgba(128,128,128,0.08);
            color: inherit;
        }
        .uqlab-pill-queued { border-color: rgba(100,149,237,0.5); }
        .uqlab-pill-running { border-color: rgba(255,165,0,0.55); }
        .uqlab-pill-completed { border-color: rgba(60,179,113,0.5); }
        .uqlab-pill-failed { border-color: rgba(220,80,80,0.5); }
        .uqlab-pill-muted { opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, kind: str = "default") -> str:
    """Return HTML for a single status pill."""
    safe_label = html.escape(str(label))
    safe_kind = html.escape(kind)
    return f'<span class="uqlab-pill uqlab-pill-{safe_kind}">{safe_label}</span>'


def status_pills_row(counts: dict[str, int]) -> None:
    """Render a row of status count pills."""
    parts = []
    mapping = (
        ("queued", "queued"),
        ("running", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
    )
    for key, kind in mapping:
        n = counts.get(key, 0)
        if n:
            parts.append(status_pill(f"{key.title()} {n}", kind))
    total = sum(counts.values()) if counts else 0
    if total:
        parts.append(status_pill(f"Total {total}", "muted"))
    if parts:
        st.markdown(
            f'<div class="uqlab-pills">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def section_header(num: str, title: str, subtitle: str = "") -> None:
    """Render a numbered section header with optional subtitle inside a card."""
    safe_num = html.escape(str(num))
    safe_title = html.escape(str(title))
    sub_html = ""
    if subtitle:
        safe_sub = html.escape(str(subtitle))
        sub_html = f'<p class="uqlab-section-sub">{safe_sub}</p>'
    st.markdown(
        (
            f'<div class="uqlab-section">'
            f'<p class="uqlab-section-title">{safe_num} · {safe_title}</p>'
            f"{sub_html}"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )
