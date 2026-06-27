"""Streamlit export — campaign config timeline + sweep plot PDF."""

from __future__ import annotations

from typing import Any

import streamlit as st

from uqlab.evaluation.reporting.campaign_report import (
    CampaignExportBundle,
    PdfLayout,
    build_multi_campaign_report_pdf,
    campaign_report_filename,
)
from uqlab.runtime_paths import experiments_root, repository_root


def render_campaign_report_download(
    experiments: list[dict[str, Any]],
    *,
    export_bundles: list[CampaignExportBundle] | None = None,
    sweep_kind: str | None = None,
    facet_filters: dict[str, Any] | None = None,
    signal: str | None = None,
    title: str | None = None,
    key_prefix: str = "campaign_report",
) -> None:
    """Campaign PDF: slim config pages + sweep plots (by section or grouped by metric)."""
    bundles = export_bundles
    if bundles is None:
        if len(experiments) < 2:
            return
        bundles = [
            CampaignExportBundle(label=title or "Campaign", experiments=tuple(experiments))
        ]
    elif len(bundles) < 1:
        return

    layout_col, _ = st.columns([3, 2])
    with layout_col:
        layout: PdfLayout = st.radio(
            "PDF layout",
            ("by_section", "by_metric"),
            horizontal=True,
            format_func=lambda k: "By section" if k == "by_section" else "By metric",
            key=f"{key_prefix}_layout",
            help=(
                "By section: config then all signal plots per epistemic/aleatoric arm. "
                "By metric: one plot page per signal comparing sweeps across sections."
            ),
        )

    cache_key = f"{key_prefix}_pdf_{layout}"
    bundle_key = tuple((b.label, tuple(e.get("id") for e in b.experiments)) for b in bundles)

    build_col, dl_col = st.columns(2)
    cached = st.session_state.get(cache_key)
    ready = cached is not None and cached[2] == bundle_key

    with build_col:
        if st.button(
            "Build campaign PDF",
            key=f"{key_prefix}_build_{layout}",
            use_container_width=True,
        ):
            with st.spinner("Assembling report…"):
                try:
                    pdf_bytes, summary = build_multi_campaign_report_pdf(
                        list(bundles),
                        experiments_root(),
                        project_root=repository_root(),
                        sweep_kind=sweep_kind,
                        facet_filters=facet_filters or None,
                        signal=signal,
                        title=title,
                        include_all_signals=True,
                        layout=layout,
                    )
                    st.session_state[cache_key] = (pdf_bytes, summary, bundle_key)
                    st.rerun()
                except Exception as exc:
                    st.session_state.pop(cache_key, None)
                    st.error(str(exc))

    with dl_col:
        if ready:
            pdf_bytes, summary, _ = cached
            st.download_button(
                "Download campaign PDF",
                data=pdf_bytes,
                file_name=campaign_report_filename(summary),
                mime="application/pdf",
                key=f"{key_prefix}_dl_{layout}",
                use_container_width=True,
            )
        else:
            st.button(
                "Download campaign PDF",
                key=f"{key_prefix}_dl_disabled_{layout}",
                disabled=True,
                use_container_width=True,
            )

    if not ready:
        n_groups = len(bundles)
        st.caption(
            f"Exports {n_groups} campaign(s) · 2-page config (shared + sweep table) · "
            f"{'plots per section' if layout == 'by_section' else 'plots grouped by metric'} · "
            "facet slice applies to plots when set."
        )
