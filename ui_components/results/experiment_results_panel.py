"""
Progressive-app experiment results panel (sweep groups, details, auto-refresh).

Each block respects :mod:`uqlab.ui_components.ui_debug` toggles.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import requests
import streamlit as st

from uqlab.ui_components.grouping import group_experiments_intelligently
from uqlab.ui_components.results.training_data_inspection import (
    parse_training_data_stats,
    render_training_data_stats,
)
from uqlab.ui_components.progressive.api_client import (
    bulk_delete_experiments_by_status,
    fetch_recoverability,
    recover_experiments_batch,
    summarize_recoverability_tiers,
)
from uqlab.ui_components.ui_debug import sync_results_auto_refresh, ui_on
from uqlab.ui_components.style import section_header, status_pill, status_pills_row


def _deferred_rerun_after(seconds: int) -> None:
    import streamlit.components.v1 as components

    components.html(
        f"""
        <script>
        setTimeout(function() {{
            window.parent.postMessage({{type: "streamlit:rerun"}}, "*");
        }}, {int(seconds) * 1000});
        </script>
        """,
        height=0,
    )


def _schedule_auto_refresh(
    *,
    enabled: bool,
    experiments: List[Dict[str, Any]],
) -> bool:
    running_statuses = {"pending", "running", "queued"}
    has_running = any(exp.get("status") in running_statuses for exp in experiments)

    if (
        not enabled
        or not ui_on("results_section")
        or not ui_on("results_auto_refresh_schedule")
    ):
        return False

    if not has_running:
        st.caption("✅ All experiments finished — auto-refresh off.")
        return False

    _deferred_rerun_after(5)
    st.caption("🔄 Auto-refresh every 5s while runs are queued/running.")
    return True


def render_experiment_stats_footer(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
) -> None:
    """Compact experiment counts at page footer (independent of main results panel)."""
    if not ui_on("results_status_metrics"):
        return

    sync_results_auto_refresh()

    try:
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers_func(),
            timeout=10,
        )
        response.raise_for_status()
        experiments = response.json()
    except requests.exceptions.RequestException:
        return

    counts: Dict[str, int] = {}
    for exp in experiments:
        status = exp.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    queued = counts.get("queued", 0)
    running = counts.get("running", 0)
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    total = len(experiments)

    st.markdown(
        f'<div class="uqlab-pills">{status_pill(f"Queued {queued}", "queued")}'
        f'{status_pill(f"Running {running}", "running")}'
        f'{status_pill(f"Completed {completed}", "completed")}'
        f'{status_pill(f"Failed {failed}", "failed")}'
        f'{status_pill(f"Total {total}", "muted")}</div>',
        unsafe_allow_html=True,
    )


def render_running_progress(experiments: List[Dict[str, Any]]) -> None:
    running = [e for e in experiments if e.get("status") == "running"]
    if not running:
        return
    st.markdown("#### ▶️ Running now")
    for exp in running[:8]:
        label = exp.get("name", str(exp.get("id", "?")))
        progress = float(exp.get("progress") or 0.0)
        st.progress(min(max(progress, 0.0), 1.0), text=f"{label} — {progress:.1%}")


def _experiment_status_counts(experiments: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for exp in experiments:
        status = exp.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _live_status_has_content(
    experiments: List[Dict[str, Any]],
    *,
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
) -> bool:
    if ui_on("results_running_progress") and any(
        e.get("status") == "running" for e in experiments
    ):
        return True
    if ui_on("results_auto_refresh_ui"):
        return True
    if ui_on("results_bulk_delete"):
        deletable = ("failed", "running", "queued")
        counts = _experiment_status_counts(experiments)
        if sum(counts.get(s, 0) for s in deletable) > 0:
            return True
    if ui_on("results_bulk_recover"):
        try:
            recoverability = fetch_recoverability(
                api_base_url,
                get_headers_func,
                status="failed",
            )
            tier_counts = summarize_recoverability_tiers(recoverability)
            if tier_counts.get("zwischen_finalize") or tier_counts.get("db_sync"):
                return True
        except requests.exceptions.RequestException:
            pass
    return False


def _render_per_run_details(
    experiments: List[Dict[str, Any]],
    *,
    key_prefix: str,
) -> None:
    if not ui_on("results_experiment_details"):
        return
    completed = [
        e for e in experiments
        if e.get("status") == "completed" and e.get("best_signals_json")
    ]
    if not completed:
        return
    st.markdown("#### Per-run details")
    st.caption(
        "Pick any completed run — sweep campaign, four-region, or standalone — "
        "for signal tables, bar charts, and disentanglement diagnostics."
    )
    from uqlab.ui_components.results.experiment_details import (
        render_experiment_details_with_metrics,
    )
    from uqlab.ui_components.results.disentanglement_score_viz import (
        render_disentanglement_score_panel,
    )

    options = {exp["name"]: exp for exp in completed}
    pick = st.selectbox(
        "Run",
        list(options.keys()),
        key=f"{key_prefix}per_run_pick",
    )
    if pick:
        render_experiment_details_with_metrics(options[pick], show_explanation=False)
        render_disentanglement_score_panel(
            options[pick],
            key_prefix=f"{key_prefix}per_run_disent",
        )


def _any_results_subpanel_on() -> bool:
    return any(
        ui_on(key)
        for key in (
            "results_live_status",
            "results_sweep_analysis",
            "results_experiment_details",
            "results_training_data",
        )
    )


def render_experiment_results_panel(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
    auto_refresh: bool,
    *,
    empty_message: str = "",
    key_prefix: str = "res_",
    sweep_analysis_renderer: Optional[Callable[..., None]] = None,
) -> bool:

    if not ui_on("results_section"):
        sync_results_auto_refresh()
        return False

    try:
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers_func(),
            timeout=10,
        )
        response.raise_for_status()
        experiments = response.json()
    except requests.exceptions.RequestException as exc:
        st.error(f"Failed to fetch experiments: {exc}")
        return auto_refresh

    if not experiments:
        st.info(
            empty_message
            or "No experiments in the database yet. Launch from Step 5."
        )
        return auto_refresh

    if not _any_results_subpanel_on():
        st.warning(
            "All Results sub-panels are off in **UI debug**. "
            "Enable **Results · §2 sweep analysis** or **Results defaults** in the sidebar."
        )
        return auto_refresh

    status_counts = _experiment_status_counts(experiments)
    status_pills_row(status_counts)

    # --- 1. Live status ---
    if ui_on("results_live_status"):
        has_live = _live_status_has_content(
            experiments,
            api_base_url=api_base_url,
            get_headers_func=get_headers_func,
        )
        if has_live:
            section_header(
                "1",
                "Live status",
                "Queued/running runs, refresh controls, bulk delete/recover.",
            )
        else:
            st.markdown(
                f'<div class="uqlab-pills">{status_pill("No live activity", "muted")}</div>',
                unsafe_allow_html=True,
            )

        if has_live and ui_on("results_running_progress"):
            render_running_progress(experiments)

        if not ui_on("results_auto_refresh_schedule"):
            auto_refresh = False

        if has_live and ui_on("results_auto_refresh_ui"):
            c1, c2, c3 = st.columns(3)
            with c1:
                auto_refresh = st.checkbox(
                    "Enable auto-refresh (5s)",
                    value=auto_refresh,
                    key=f"{key_prefix}auto_refresh",
                )
            with c2:
                if st.button("Refresh now", key=f"{key_prefix}refresh", use_container_width=True):
                    st.rerun()
            with c3:
                if st.button("Stop refresh", key=f"{key_prefix}stop", use_container_width=True):
                    auto_refresh = False
                    st.rerun()

        if has_live and ui_on("results_bulk_delete"):
            status_counts: Dict[str, int] = {}
            for exp in experiments:
                s = exp.get("status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1

            deletable_statuses = ("failed", "running", "queued")
            total_deletable = sum(status_counts.get(status, 0) for status in deletable_statuses)

            if total_deletable > 0:
                st.markdown("#### Bulk delete")
                for status in deletable_statuses:
                    n = status_counts.get(status, 0)
                    if n and st.button(f"Delete {n} {status}", key=f"{key_prefix}del_{status}"):
                        deleted = bulk_delete_experiments_by_status(
                            api_base_url,
                            get_headers_func,
                            experiments,
                            status,
                        )
                        st.success(f"Deleted {deleted} {status} experiment(s)")
                        st.rerun()

        if has_live and ui_on("results_bulk_recover"):
            try:
                recoverability = fetch_recoverability(
                    api_base_url,
                    get_headers_func,
                    status="failed",
                )
            except requests.exceptions.RequestException as exc:
                st.warning(f"Recoverability check unavailable: {exc}")
                recoverability = []

            tier_counts = summarize_recoverability_tiers(recoverability)
            n_failed = sum(
                1 for e in recoverability if e.get("status") == "failed"
            ) or sum(1 for e in experiments if e.get("status") == "failed")
            n_zwischen = tier_counts.get("zwischen_finalize", 0)
            n_db_sync = tier_counts.get("db_sync", 0)

            if n_zwischen or n_db_sync:
                st.markdown("#### Bulk recover")
                st.caption(
                    f"{n_failed} failed · {n_zwischen} recoverable from zwischen · "
                    f"{n_db_sync} need DB sync only"
                )
                if n_zwischen and st.button(
                    f"Recover {n_zwischen} failed (extract results)",
                    key=f"{key_prefix}recover_zwischen",
                ):
                    with st.spinner("Finalizing runs from disk (no re-training)…"):
                        result = recover_experiments_batch(
                            api_base_url,
                            get_headers_func,
                            status="failed",
                            tier="zwischen_finalize",
                        )
                    st.success(
                        f"Recovered {result.get('recovered', 0)} run(s); "
                        f"skipped {result.get('skipped', 0)}"
                    )
                    errors = result.get("errors") or []
                    if errors:
                        st.warning(f"{len(errors)} run(s) failed recovery")
                    st.rerun()
                if n_db_sync and st.button(
                    f"Sync {n_db_sync} completed metrics to DB",
                    key=f"{key_prefix}recover_db_sync",
                ):
                    with st.spinner("Syncing disk artifacts to database…"):
                        result = recover_experiments_batch(
                            api_base_url,
                            get_headers_func,
                            status="failed",
                            tier="db_sync",
                        )
                    st.success(
                        f"Synced {result.get('recovered', 0)} run(s); "
                        f"skipped {result.get('skipped', 0)}"
                    )
                    st.rerun()

    sweep_groups, _standalone = group_experiments_intelligently(experiments, min_group_size=3)

    show_analysis = (
        ui_on("results_sweep_analysis")
        or ui_on("results_experiment_details")
        or ui_on("results_sweep_campaigns")
        or ui_on("results_standalone_table")
    )
    if show_analysis:
        st.markdown("---")
        section_header(
            "2",
            "Analysis & sweeps",
            "Paper sweeps (Fig 3/4 pool means), four-region validation, and per-run metrics. "
            "AUROC = separability; pool-mean Y values = averages over eval pools.",
        )
        if ui_on("results_sweep_analysis") and sweep_groups and sweep_analysis_renderer:
            try:
                sweep_analysis_renderer(sweep_groups, key_prefix=f"{key_prefix}sweep_hub")
            except Exception as exc:
                st.warning(f"Sweep analysis unavailable: {exc}")
        elif ui_on("results_sweep_analysis") and not sweep_groups:
            st.info(
                "No sweep campaigns yet (need ≥3 similar runs). "
                "Four-region and standalone runs appear in **Per-run details** below."
            )
        _render_per_run_details(experiments, key_prefix=key_prefix)

    if ui_on("results_training_data"):
        st.markdown("---")
        section_header(
            "3",
            "Training data inspection",
            "On-disk training split stats per completed run.",
        )
        completed_with_data = [
            e for e in experiments
            if e.get("status") == "completed" and parse_training_data_stats(str(e.get("id")))
        ]
        if completed_with_data:
            options = {
                f"{e['name']} ({str(e['id'])[:8]})": e["id"]
                for e in completed_with_data
            }
            pick = st.selectbox(
                "Experiment",
                list(options.keys()),
                key=f"{key_prefix}train_pick",
            )
            if pick:
                render_training_data_stats(options[pick])
        else:
            st.caption("No training-data artifacts on disk yet.")

    return _schedule_auto_refresh(enabled=auto_refresh, experiments=experiments) or auto_refresh
