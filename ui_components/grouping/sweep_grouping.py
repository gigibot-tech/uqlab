"""Streamlit render layer for sweep groups — domain logic lives in ``uqlab_orchestrator.sweep_groups``."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from uqlab.ui_components.grouping.campaign_format import (
    campaign_date_label,
    representative_experiment_id,
)
from uqlab_orchestrator.sweep_groups import (
    group_experiments_by_config_similarity,
    group_experiments_by_metadata,
    group_experiments_by_name_pattern,
    group_experiments_intelligently,
)

__all__ = [
    "group_experiments_by_config_similarity",
    "group_experiments_by_metadata",
    "group_experiments_by_name_pattern",
    "group_experiments_intelligently",
    "render_sweep_group_summary",
]


def render_sweep_group_summary(
    group: Dict[str, Any],
    show_details: bool = False,
    api_base_url: Optional[str] = None,
    *,
    show_inline_plot: bool = False,
) -> None:
    """
    Render a summary card for a sweep group.

    Args:
        group: Sweep group dict from group_experiments_intelligently()
        show_details: Whether to show detailed experiment list
        api_base_url: Reserved for future API-backed metrics
    """
    del api_base_url

    experiments = group["experiments"]
    swept_param = group["swept_param"]
    values = group["values"]

    date_label = campaign_date_label(group, experiments=experiments)
    rep_id = representative_experiment_id(experiments)
    st.caption(f"**{date_label}** · `{rep_id}` · sweep `{group.get('sweep_group_id') or '—'}`")

    completed = sum(1 for e in experiments if e["status"] == "completed")
    running = sum(1 for e in experiments if e["status"] == "running")
    failed = sum(1 for e in experiments if e["status"] == "failed")

    aleatoric_scores = [e.get("aleatoric_auroc") for e in experiments if e.get("aleatoric_auroc")]
    epistemic_scores = [e.get("epistemic_auroc") for e in experiments if e.get("epistemic_auroc")]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", len(experiments))
    with col2:
        st.metric("Completed", f"{completed}/{len(experiments)}")
    with col3:
        if aleatoric_scores:
            st.metric("Best Aleatoric AUROC", f"{max(aleatoric_scores):.3f}")
        else:
            st.metric("Best Aleatoric AUROC", "N/A")
    with col4:
        if epistemic_scores:
            st.metric("Best Epistemic AUROC", f"{max(epistemic_scores):.3f}")
        else:
            st.metric("Best Epistemic AUROC", "N/A")

    st.markdown(f"**Swept Parameter:** `{swept_param}`")
    st.markdown(f"**Values:** {', '.join(str(v) for v in values)}")

    status_text = f"✅ {completed} completed"
    if running > 0:
        status_text += f" | 🔄 {running} running"
    if failed > 0:
        status_text += f" | ❌ {failed} failed"
    st.caption(status_text)

    if show_inline_plot:
        completed_with_artifacts = [e for e in experiments if e.get("status") == "completed"]
        if len(completed_with_artifacts) >= 2:
            st.markdown("---")
            st.caption("Plot: use **§2 Sweep analysis** above for signal + facet controls.")
            from uqlab.ui_components.visualization.sweeps.sweep_line_plot_viz import (
                render_sweep_line_plot,
            )

            group_key = group.get("sweep_group_id") or group.get("name") or swept_param
            render_sweep_line_plot(
                completed_with_artifacts,
                key_prefix=f"grp_{group_key}",
                show_signal_picker=False,
            )

    if show_details:
        st.markdown("---")
        st.markdown("**Experiment Details:**")

        data = []
        for exp, value in zip(experiments, values):
            data.append({
                "Value": value,
                "Name": exp["name"],
                "Status": exp["status"],
                "Aleatoric": f"{exp.get('aleatoric_auroc', 0):.3f}" if exp.get("aleatoric_auroc") else "N/A",
                "Epistemic": f"{exp.get('epistemic_auroc', 0):.3f}" if exp.get("epistemic_auroc") else "N/A",
            })

        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        st.markdown("**📊 View Detailed Metrics:**")
        completed_exps = [
            e for e in experiments
            if e["status"] == "completed" and e.get("best_signals_json")
        ]

        if completed_exps:
            from uqlab.ui_components.results.experiment_details import (
                render_experiment_details_with_metrics,
            )

            for exp in completed_exps:
                with st.expander(f"🔬 {exp['name']} - Detailed Metrics", expanded=False):
                    render_experiment_details_with_metrics(exp, show_explanation=False)
        else:
            st.info("💡 Detailed metrics will be available after experiments complete")
