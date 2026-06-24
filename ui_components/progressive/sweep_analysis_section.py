"""
Structured sweep analysis hub — paper plots first, pool-filtered diagnostics second.

**Primary (paper):** scores + global aleatoric + global epistemic vs Percentage (0–1).

**Secondary (uqlab):** 4D 3-line pool plot — X = swept param, Y = pool means + accuracy,
signal picker, facet slice.
"""

from __future__ import annotations

from typing import Any, Callable

import streamlit as st

from uqlab.ui_components.grouping.campaign_format import format_sweep_campaign_header
from uqlab.evaluation.pipeline.sweep_plot_pools import eval_pool_debug_rows
from uqlab.evaluation.pipeline.sweep_line_plot import (
    FACET_PARAM_LABELS,
    SWEEP_KIND_DATASET_SIZE,
    SWEEP_KIND_LABEL_NOISE,
    build_sweep_metrics_frame,
    detect_facet_columns,
    infer_sweep_kind,
    resolve_x_col,
    sweep_kind_from_group,
)
from uqlab_orchestrator.uncertainty import iter_perspectives
from uqlab.runtime_paths import experiments_root, repository_root
from uqlab.ui_components.progressive.plot_probe_panel import render_results_plot_status
from uqlab.ui_components.ui_debug import ui_on
from uqlab.ui_components.visualization.sweeps.paper_benchmark_plot_viz import (
    render_paper_benchmark_plot,
)
from uqlab.ui_components.visualization.sweeps.sweep_line_plot_viz import (
    render_sweep_line_plot,
    run_ids_for_experiments,
)
from uqlab.ui_components.visualization.thesis.thesis_diagram_viz import render_thesis_diagram_panel
from uqlab.ui_components.visualization.campaign.campaign_report_viz import render_campaign_report_download
from uqlab.evaluation.pipeline.campaign_report import CampaignExportBundle


def _completed_experiments(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in experiments if e.get("status") == "completed"]


def _experiments_for_perspective(
    experiments: list[dict[str, Any]],
    perspective_id: str,
) -> list[dict[str, Any]]:
    """Filter runs whose names match a registered perspective."""
    if perspective_id == "epistemic":
        return [
            e for e in experiments
            if "fast_epis" in str(e.get("name", "")) or "_under_" in str(e.get("name", ""))
        ]
    if perspective_id == "aleatoric":
        return [
            e for e in experiments
            if "fast_alea" in str(e.get("name", "")) or "_noise_" in str(e.get("name", ""))
        ]
    return []


def _arm_buckets(
    experiments: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition completed runs by perspective id (registry-driven)."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for p in iter_perspectives():
        runs = _experiments_for_perspective(experiments, p.id)
        if runs:
            buckets[p.id] = runs
    return buckets


def _campaign_label(group: dict[str, Any], experiments: list[dict[str, Any]]) -> str:
    buckets = _arm_buckets(experiments)
    parts = [
        f"{next(p for p in iter_perspectives() if p.id == pid).short_label} {len(runs)}"
        for pid, runs in buckets.items()
    ]
    suffix = " + ".join(parts) if parts else f"{len(experiments)} runs"
    return format_sweep_campaign_header(group, experiments=experiments, suffix=suffix)


def render_sweep_analysis_hub(
    sweep_groups: list[dict[str, Any]],
    *,
    key_prefix: str = "sweep_hub",
    on_apply_launch: Callable[[dict[str, Any], bool], None] | None = None,
) -> None:
    """
    Primary UI for sweep analysis: paper 3-curve plot, then pool-filtered diagnostic.
    """
    if not ui_on("results_sweep_analysis"):
        return
    plottable_groups: list[dict[str, Any]] = []
    for group in sweep_groups:
        completed = _completed_experiments(group.get("experiments") or [])
        if len(run_ids_for_experiments(completed)) >= 2:
            plottable_groups.append(group)

    if not plottable_groups:
        st.info(
            "Sweep plots appear here once a campaign has **≥2 completed runs** with "
            "`results.pt` on disk (pool means are not in `summary.json` alone)."
        )
        return

    st.caption(
        "**Paper plot (default):** Percentage (0–1) vs accuracy + global aleatoric/epistemic means. "
        "**Pool diagnostic (expander):** raw sweep param · pool-filtered signal · facet slice."
    )

    labels = [
        _campaign_label(g, g.get("experiments") or [])
        for g in plottable_groups
    ]
    pick_idx = st.selectbox(
        "Sweep campaign",
        range(len(plottable_groups)),
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_group",
    )
    group = plottable_groups[pick_idx]
    completed = _completed_experiments(group["experiments"])
    arm_buckets = _arm_buckets(completed)

    sweep_kind_for_perspective = {
        "epistemic": SWEEP_KIND_DATASET_SIZE,
        "aleatoric": SWEEP_KIND_LABEL_NOISE,
    }

    forced_kind: str | None = None
    if len(arm_buckets) > 1:
        st.caption(
            "Same launch id — **separate 1D sweeps** per perspective (not a 2D grid). "
            "Pick which arm to plot."
        )
        arm_ids = list(arm_buckets.keys())
        arm_tab = st.radio(
            "Sweep arm",
            arm_ids,
            horizontal=True,
            format_func=lambda pid: (
                f"{next(p for p in iter_perspectives() if p.id == pid).label} "
                f"({next(p for p in iter_perspectives() if p.id == pid).fig_label}) "
                f"— {len(arm_buckets[pid])} runs"
            ),
            key=f"{key_prefix}_arm",
        )
        plot_experiments = arm_buckets[arm_tab]
        forced_kind = sweep_kind_for_perspective.get(arm_tab)
    else:
        plot_experiments = completed

    run_ids = run_ids_for_experiments(plot_experiments)
    exp_dir = experiments_root()

    df = build_sweep_metrics_frame(run_ids, exp_dir)
    if df.empty:
        st.warning("No on-disk metrics for this campaign yet.")
        return

    group_hint = forced_kind or sweep_kind_from_group(group)
    auto_kind = infer_sweep_kind(df, hint=group_hint)

    x_axis_labels = {
        "auto": f"Auto — {auto_kind.replace('_', ' ')} (from campaign)",
        SWEEP_KIND_LABEL_NOISE: "Label noise (%) — Fig 4 / aleatoric axis",
        SWEEP_KIND_DATASET_SIZE: "Under-train per class — Fig 3 / epistemic axis",
    }
    x_pick = st.selectbox(
        "X axis (swept parameter)",
        ["auto", SWEEP_KIND_LABEL_NOISE, SWEEP_KIND_DATASET_SIZE],
        format_func=lambda k: x_axis_labels[k],
        key=f"{key_prefix}_x_axis",
    )
    sweep_kind = auto_kind if x_pick == "auto" else x_pick
    try:
        x_col = resolve_x_col(df, sweep_kind)
        facets = detect_facet_columns(df, x_col)
    except ValueError as exc:
        st.warning(str(exc))
        return

    if group_hint and x_pick == "auto" and group_hint != auto_kind:
        st.caption(
            f"Campaign metadata suggests **{group_hint.replace('_', ' ')}**; "
            f"metrics frame inferred **{auto_kind.replace('_', ' ')}**."
        )

    facet_filters: dict[str, Any] = {}
    if facets:
        st.markdown("**Facet slice (Z)** — hold constant so X is a clean 1D sweep")
        facet_cols = st.columns(min(len(facets), 3))
        for i, (col, values) in enumerate(facets.items()):
            label = FACET_PARAM_LABELS.get(col, col.replace("_", " ").title())
            with facet_cols[i % len(facet_cols)]:
                choice = st.selectbox(
                    label,
                    ["all", *values],
                    format_func=lambda v, _l=label: (
                        f"All {_l} (mixed)" if v == "all" else f"{v}"
                    ),
                    key=f"{key_prefix}_facet_{col}",
                )
                if choice != "all":
                    facet_filters[col] = choice
        if not facet_filters:
            st.caption(
                "Tip: if learning rate (or another hyperparameter) varies across runs, "
                "pick one value above — otherwise the lines mix incompatible training setups."
            )

    with st.expander("Debug: eval pool counts vs config", expanded=False):
        debug_rows = eval_pool_debug_rows(run_ids, exp_dir)
        if debug_rows:
            st.caption(
                "Group counts from `results.pt` · `epistemic_expected` / `aleatoric_expected` "
                "from config (same rules as split construction). At 100% noise, "
                "`n_epistemic_like` should be 0."
            )
            st.dataframe(debug_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No on-disk results to inspect.")

    with st.expander("Methods schematic", expanded=False):
        render_thesis_diagram_panel(
            experiments=plot_experiments,
            project_root=repository_root(),
            key_prefix=f"{key_prefix}_thesis",
            default_symbolic=True,
        )

    plot_key = f"{key_prefix}_{group.get('sweep_group_id', pick_idx)}"
    paper_shown = render_paper_benchmark_plot(
        plot_experiments,
        sweep_kind=sweep_kind,
        key_prefix=f"{plot_key}_paper",
        quiet=True,
    )
    if not paper_shown:
        st.info(
            "Paper plot unavailable for this slice — need global `{signal}_mean` columns in "
            "`results.pt` (re-run with fast-pilot eval). See pool diagnostic below."
        )

    with st.expander("Pool-filtered sweep plot (uqlab diagnostic)", expanded=not paper_shown):
        st.caption(
            "**4D model:** X = swept parameter · Y = primary pool mean (+ optional mirror) "
            "and accuracy · **Signal** = which uncertainty metric · **Z** = facet slice."
        )
        shown = render_sweep_line_plot(
            plot_experiments,
            key_prefix=plot_key,
            facet_filters=facet_filters or None,
            sweep_kind=sweep_kind,
            quiet=True,
        )

    plottable_completed = run_ids_for_experiments(completed)
    if len(plottable_completed) >= 2:
        import re

        safe_plot_key = re.sub(r"[^\w]", "_", plot_key)[:48]
        picked_signal = st.session_state.get(f"{safe_plot_key}_signal")
        export_pick = st.multiselect(
            "Campaigns for PDF export",
            range(len(plottable_groups)),
            default=[pick_idx],
            format_func=lambda i: labels[i],
            key=f"{key_prefix}_export_groups",
            help="One or more sweep campaigns in the same PDF.",
        )
        export_bundles = [
            CampaignExportBundle(
                label=labels[i],
                experiments=tuple(_completed_experiments(plottable_groups[i].get("experiments") or [])),
            )
            for i in export_pick
        ]
        render_campaign_report_download(
            completed,
            export_bundles=export_bundles,
            sweep_kind=None,
            facet_filters=facet_filters or None,
            signal=picked_signal,
            title=_campaign_label(group, completed),
            key_prefix=f"{key_prefix}_report_{group.get('sweep_group_id', pick_idx)}",
        )

    if not shown and not paper_shown:
        from uqlab_orchestrator.plot_probe import assess_outcome

        probe = assess_outcome(
            plot_experiments,
            sweep_kind=sweep_kind,
            facet_filters=facet_filters or None,
        )
        rep = plot_experiments[0] if plot_experiments else {}
        rep_id = str(rep.get("id") or "")
        source_label = str(rep.get("name") or rep_id or "campaign")
        render_results_plot_status(
            probe,
            source_label=source_label,
            key_prefix=f"{key_prefix}_plot_fail",
            insufficient_runs=len(run_ids) < 2,
        )
