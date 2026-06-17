"""
Progressive-app experiment results panel (sweep groups, details, auto-refresh).

Each block respects :mod:`uqlab.ui_components.ui_debug` toggles.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from uqlab.ui_components.grouping import (
    group_experiments_intelligently,
    render_sweep_group_summary,
)
from uqlab.ui_components.results.training_data_inspection import (
    parse_training_data_stats,
    render_training_data_stats,
)
from uqlab.ui_components.ui_debug import sync_results_auto_refresh, ui_on


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
        (
            "<p style='font-size:0.72rem;color:#888;text-align:center;margin:4px 0 0 0'>"
            f"Queued {queued} · Running {running} · Completed {completed} · "
            f"Failed {failed} · Total {total}"
            "</p>"
        ),
        unsafe_allow_html=True,
    )


def render_status_metrics(experiments: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for exp in experiments:
        status = exp.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Queued", counts.get("queued", 0))
    with c2:
        st.metric("Running", counts.get("running", 0))
    with c3:
        st.metric("Completed", counts.get("completed", 0))
    with c4:
        st.metric("Failed", counts.get("failed", 0))
    with c5:
        st.metric("Total", len(experiments))


def render_running_progress(experiments: List[Dict[str, Any]]) -> None:
    running = [e for e in experiments if e.get("status") == "running"]
    if not running:
        return
    st.markdown("#### ▶️ Running now")
    for exp in running[:8]:
        label = exp.get("name", str(exp.get("id", "?")))
        progress = float(exp.get("progress") or 0.0)
        st.progress(min(max(progress, 0.0), 1.0), text=f"{label} — {progress:.1%}")


def render_experiment_results_panel(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
    auto_refresh: bool,
    *,
    caption: str = "",
    empty_message: str = "",
    highlight_experiment_id: Optional[str] = None,
    key_prefix: str = "res_",
    workflow: Optional[dict] = None,
    **kwargs: Any,
) -> bool:
    del highlight_experiment_id, workflow, kwargs

    if not ui_on("results_section"):
        sync_results_auto_refresh()
        return False

    if caption:
        st.caption(caption)

    if not ui_on("results_sweep_groups") and not ui_on("results_running_progress"):
        st.info("Results sub-panels disabled in UI debug.")
        return auto_refresh

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

    if ui_on("results_running_progress"):
        render_running_progress(experiments)

    if not ui_on("results_auto_refresh_schedule"):
        auto_refresh = False

    if ui_on("results_auto_refresh_ui"):
        c1, c2, c3 = st.columns(3)
        with c1:
            auto_refresh = st.checkbox(
                "🔄 Enable Auto-Refresh (5s)",
                value=auto_refresh,
                key=f"{key_prefix}auto_refresh",
            )
        with c2:
            if st.button("🔄 Refresh Now", key=f"{key_prefix}refresh", use_container_width=True):
                st.rerun()
        with c3:
            if st.button("🛑 Stop Refresh", key=f"{key_prefix}stop", use_container_width=True):
                auto_refresh = False
                st.rerun()

    if ui_on("results_bulk_delete"):
        st.markdown("---")
        st.markdown("### 🗑️ Bulk delete")
        status_counts: Dict[str, int] = {}
        for exp in experiments:
            s = exp.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

        for status in ("failed", "running", "queued"):
            n = status_counts.get(status, 0)
            if n and st.button(f"Delete {n} {status}", key=f"{key_prefix}del_{status}"):
                deleted = 0
                for exp in experiments:
                    if exp.get("status") != status:
                        continue
                    try:
                        r = requests.delete(
                            f"{api_base_url}/api/v1/experiments/no-auth/{exp['id']}",
                            headers=get_headers_func(),
                            timeout=10,
                        )
                        if r.status_code == 200:
                            deleted += 1
                    except Exception:
                        pass
                st.success(f"Deleted {deleted} {status} experiment(s)")
                st.rerun()

    st.markdown("---")
    st.markdown("### 🔬 Experiments & parameter sweeps")

    sweep_groups, standalone = group_experiments_intelligently(experiments, min_group_size=3)
    total_in_sweeps = sum(len(g["experiments"]) for g in sweep_groups)
    st.caption(
        f"{len(sweep_groups)} sweep group(s) ({total_in_sweeps} runs), "
        f"{len(standalone)} standalone"
    )

    if ui_on("results_sweep_groups") and sweep_groups:
        st.markdown("#### 📊 Parameter sweeps")
        for i, group in enumerate(sweep_groups):
            with st.expander(
                f"🔬 {group['name']} ({len(group['experiments'])} runs)",
                expanded=(i == 0 and ui_on("results_experiment_details")),
            ):
                if ui_on("results_sweep_summary"):
                    render_sweep_group_summary(
                        group,
                        show_details=ui_on("results_experiment_details"),
                    )

    if ui_on("results_standalone_table") and standalone:
        st.markdown("#### 🧪 Standalone experiments")
        rows = []
        for exp in standalone:
            rows.append({
                "ID": str(exp["id"])[:8],
                "Name": exp["name"],
                "Status": exp["status"],
                "Progress": f"{exp.get('progress', 0):.1%}",
                "Aleatoric": f"{exp['aleatoric_auroc']:.3f}" if exp.get("aleatoric_auroc") else "N/A",
                "Epistemic": f"{exp['epistemic_auroc']:.3f}" if exp.get("epistemic_auroc") else "N/A",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if ui_on("results_experiment_details"):
            completed = [
                e for e in standalone
                if e.get("status") == "completed" and e.get("best_signals_json")
            ]
            if completed:
                from uqlab.ui_components.results.experiment_details import (
                    render_experiment_details_with_metrics,
                )

                for i, exp in enumerate(completed):
                    with st.expander(
                        f"🔬 {exp['name']} — metrics",
                        expanded=(i == 0),
                    ):
                        render_experiment_details_with_metrics(exp, show_explanation=False)

    if ui_on("results_training_data"):
        st.markdown("---")
        st.markdown("### 📊 Training data inspection")
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

    if ui_on("results_batch_ui"):
        st.markdown("---")
        with st.expander("📦 Legacy: Batch experiments (UI-created)", expanded=False):
            st.caption("⚠️ **Legacy view** - Old batch experiment interface")
            try:
                from uqlab.ui_components.visualization.signals.signal_visualization import (
                    render_batch_results,
                )

                render_batch_results(api_base_url, get_headers_func)
            except Exception as exc:
                st.warning(f"Batch results UI unavailable: {exc}")

    return _schedule_auto_refresh(enabled=auto_refresh, experiments=experiments) or auto_refresh
