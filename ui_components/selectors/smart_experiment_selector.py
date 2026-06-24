"""
Sidebar experiment campaign/run picker for the progressive Streamlit app.

Groups API and on-disk runs by campaign timestamp, detects 1D/2D sweep shape,
and writes ``highlight_experiment_id`` for the main results panel.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from uqlab.ui_components.grouping.campaign_format import (
    campaign_date_from_batch,
    format_campaign_timestamp,
    representative_experiment_id,
)
from uqlab.experiment_config_flat import (
    ALEATORIC_PARAM,
    EPISTEMIC_PARAM,
    resolve_experiment_param,
)
from uqlab.runtime_paths import resolve_experiment_results_dir

_TS_RE = re.compile(r"(\d{8}_\d{6})")
_FAST_SWEEP_RE = re.compile(r"^fast_(alea|epis)_(\d{8}_\d{6})")


def _experiment_results_dir(
    experiment_id: str,
    *,
    results_path: str | None = None,
) -> Path:
    return Path(resolve_experiment_results_dir(experiment_id, results_path=results_path))


def _parse_config(exp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = exp.get("config") or exp.get("config_yaml") or {}
    if isinstance(cfg, str):
        try:
            return json.loads(cfg)
        except json.JSONDecodeError:
            return {}
    return cfg if isinstance(cfg, dict) else {}


def _get_sweep_values(experiments: List[Dict[str, Any]], param: str) -> List[float]:
    values: set[float] = set()
    for exp in experiments:
        val = resolve_experiment_param(exp, param)
        if val is not None:
            values.add(float(val))
    return sorted(values)


def index_campaign_batches(
    experiments: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Map campaign timestamp → {epis: [...], alea: [...]}."""
    by_ts: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for exp in experiments:
        m = _FAST_SWEEP_RE.match(exp.get("name") or "")
        if not m:
            continue
        arm, ts = m.group(1), m.group(2)
        by_ts.setdefault(ts, {"epis": [], "alea": []})
        by_ts[ts][arm].append(exp)
    return by_ts


def group_experiments_for_selection(
    experiments: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group by campaign timestamp (merges fast_epis + fast_alea arms)."""
    by_ts = index_campaign_batches(experiments)
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for ts, arms in by_ts.items():
        combined = list(arms.get("epis", [])) + list(arms.get("alea", []))
        if combined:
            groups[ts] = combined

    assigned = {id(e) for exps in groups.values() for e in exps}
    for exp in experiments:
        if id(exp) in assigned:
            continue
        name = exp.get("name", "") or ""
        m = _TS_RE.search(name)
        key = m.group(1) if m else f"run_{exp.get('id', 'unknown')}"
        groups.setdefault(key, []).append(exp)

    return groups


def detect_experiment_configuration(
    experiments: List[Dict[str, Any]],
    *,
    mode: str = "quick",
) -> Dict[str, Any]:
    """1D epistemic / 1D aleatoric / 2D grid / single_point + complement hints."""
    del mode  # reserved for future sweep presets
    epistemic_vals = _get_sweep_values(experiments, EPISTEMIC_PARAM)
    aleatoric_vals = _get_sweep_values(experiments, ALEATORIC_PARAM)
    n_epi = len(epistemic_vals)
    n_alea = len(aleatoric_vals)
    completed = [e for e in experiments if e.get("status") == "completed"]

    if n_epi >= 2 and n_alea >= 2:
        exp_type = "2d_grid"
        needs_complement = False
        complement_type = None
    elif n_epi >= 2 and n_alea <= 1:
        exp_type = "1d_epistemic"
        needs_complement = True
        complement_type = "aleatoric"
    elif n_alea >= 2 and n_epi <= 1:
        exp_type = "1d_aleatoric"
        needs_complement = True
        complement_type = "epistemic"
    else:
        exp_type = "single_point"
        needs_complement = True
        complement_type = "both"

    return {
        "type": exp_type,
        "epistemic_values": epistemic_vals,
        "aleatoric_values": aleatoric_vals,
        "n_epistemic": n_epi,
        "n_aleatoric": n_alea,
        "completed_count": len(completed),
        "total_count": len(experiments),
        "needs_complement": needs_complement,
        "complement_type": complement_type,
    }


@dataclass
class ExperimentVizAnalysis:
    experiment_id: str
    name: str
    status: str
    dimension: str
    campaign_ts: Optional[str] = None
    fast_arm: Optional[str] = None
    epis_batch: List[Dict[str, Any]] = field(default_factory=list)
    alea_batch: List[Dict[str, Any]] = field(default_factory=list)
    epis_sweep_values: List[float] = field(default_factory=list)
    alea_sweep_values: List[float] = field(default_factory=list)
    missing_arm: Optional[str] = None

    @property
    def dimension_label(self) -> str:
        labels = {
            "single_point": "single config",
            "1d_epistemic": "1D epistemic",
            "1d_aleatoric": "1D aleatoric",
            "2d_grid": "2D grid",
            "unknown": "unknown",
        }
        return labels.get(self.dimension, self.dimension)


def _fast_sweep_arm(exp: Dict[str, Any]) -> Optional[str]:
    m = _FAST_SWEEP_RE.match(exp.get("name") or "")
    return m.group(1) if m else None


def _campaign_timestamp_from_experiment(exp: Dict[str, Any]) -> Optional[str]:
    m = _FAST_SWEEP_RE.match(exp.get("name") or "")
    return m.group(2) if m else None


def analyze_experiment_for_viz(
    exp: Dict[str, Any],
    all_experiments: List[Dict[str, Any]],
) -> ExperimentVizAnalysis:
    """Lightweight inspection for sidebar run labels."""
    exp_id = str(exp.get("id", ""))
    name = exp.get("name") or "?"
    status = exp.get("status") or "unknown"
    ts = _campaign_timestamp_from_experiment(exp)
    arm = _fast_sweep_arm(exp)
    by_ts = index_campaign_batches(all_experiments)

    epis_batch: List[Dict[str, Any]] = []
    alea_batch: List[Dict[str, Any]] = []
    if ts and ts in by_ts:
        epis_batch = list(by_ts[ts].get("epis", []))
        alea_batch = list(by_ts[ts].get("alea", []))
    elif arm == "epis":
        epis_batch = [exp]
    elif arm == "alea":
        alea_batch = [exp]

    epis_vals = _get_sweep_values(epis_batch or [exp], EPISTEMIC_PARAM)
    alea_vals = _get_sweep_values(alea_batch or [exp], ALEATORIC_PARAM)
    n_epi = len(epis_vals)
    n_alea = len(alea_vals)

    if n_epi >= 2 and n_alea >= 2:
        dimension = "2d_grid"
    elif n_epi >= 2:
        dimension = "1d_epistemic"
    elif n_alea >= 2:
        dimension = "1d_aleatoric"
    elif arm == "epis":
        dimension = "1d_epistemic"
    elif arm == "alea":
        dimension = "1d_aleatoric"
    else:
        dimension = "single_point"

    missing_arm: Optional[str] = None
    if ts or arm:
        if not epis_batch and alea_batch:
            missing_arm = "epistemic"
        elif not alea_batch and epis_batch:
            missing_arm = "aleatoric"

    return ExperimentVizAnalysis(
        experiment_id=exp_id,
        name=name,
        status=status,
        dimension=dimension,
        campaign_ts=ts,
        fast_arm=arm,
        epis_batch=epis_batch,
        alea_batch=alea_batch,
        epis_sweep_values=epis_vals,
        alea_sweep_values=alea_vals,
        missing_arm=missing_arm,
    )


def _count_arms(batch_exps: List[Dict[str, Any]]) -> Tuple[int, int]:
    epis = sum(1 for e in batch_exps if _fast_sweep_arm(e) == "epis")
    alea = sum(1 for e in batch_exps if _fast_sweep_arm(e) == "alea")
    return epis, alea


def _short_run_name(name: str) -> str:
    """Human-readable run label from API experiment name."""
    for pattern, fmt in (
        (r"_noise_(\d+)$", "noise {}%"),
        (r"_under_(\d+)$", "under-train {}"),
        (r"aleatoric_noise_percentage_(\d+)$", "noise {}%"),
        (r"under_train_per_class_(\d+)$", "under-train {}"),
    ):
        m = re.search(pattern, name)
        if m:
            return fmt.format(m.group(1))
    if len(name) > 42:
        return name[:39] + "…"
    return name


def format_experiment_viz_option(
    exp: Dict[str, Any],
    analysis: ExperimentVizAnalysis,
    *,
    highlight: bool = False,
) -> str:
    star = "★ " if highlight else ""
    status = exp.get("status", "?")
    short = _short_run_name(analysis.name)
    eid = str(exp.get("id", ""))[:8]
    return f"{star}{short} ({status}) — {eid}"


def _batch_label_map(
    experiments: List[Dict[str, Any]],
    *,
    mode: str = "quick",
) -> Dict[str, Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
    groups = group_experiments_for_selection(experiments)

    def _newest_created(exps: List[Dict[str, Any]]) -> str:
        times = [e.get("created_at") or "" for e in exps]
        return max(times) if times else ""

    ordered = sorted(groups.items(), key=lambda item: _newest_created(item[1]), reverse=True)

    batch_labels: Dict[str, Tuple[str, List[Dict[str, Any]], Dict[str, Any]]] = {}
    for batch_id, batch_exps in ordered:
        cfg = detect_experiment_configuration(batch_exps, mode=mode)
        n_epi_runs, n_alea_runs = _count_arms(batch_exps)

        if n_epi_runs and n_alea_runs:
            type_short = f"Fig 3+4 ({n_epi_runs}+{n_alea_runs} runs)"
        elif n_epi_runs:
            type_short = f"Fig 3 ({n_epi_runs} runs)"
        elif n_alea_runs:
            type_short = f"Fig 4 ({n_alea_runs} runs)"
        else:
            type_short = {
                "1d_epistemic": "under-train sweep",
                "1d_aleatoric": "noise sweep",
                "2d_grid": "mixed axes",
                "single_point": "single run",
            }.get(cfg["type"], cfg["type"])

        flag = ""
        if n_epi_runs and not n_alea_runs:
            flag = " · missing Fig 4 arm"
        elif n_alea_runs and not n_epi_runs:
            flag = " · missing Fig 3 arm"
        elif cfg["needs_complement"] and not (n_epi_runs or n_alea_runs):
            flag = f" · incomplete ({cfg.get('complement_type', '?')})"

        date_label = campaign_date_from_batch(batch_id, batch_exps)
        rep_id = representative_experiment_id(batch_exps)
        label = (
            f"{date_label} · {rep_id} · {type_short} · "
            f"{cfg['completed_count']}/{cfg['total_count']} done"
            f"{flag}"
        )
        batch_labels[label] = (batch_id, batch_exps, cfg)
    return batch_labels


def _resolve_selection(
    experiments: List[Dict[str, Any]],
    *,
    highlight_experiment_id: Optional[str],
    batch_label: Optional[str],
    batch_labels: Dict[str, Tuple[str, List[Dict[str, Any]], Dict[str, Any]]],
    run_key: str,
) -> tuple[Optional[Dict[str, Any]], str, List[Dict[str, Any]], Dict[str, Any]]:
    if not batch_label or batch_label not in batch_labels:
        return None, "", [], {}

    batch_id, batch_exps, config = batch_labels[batch_label]
    analyses = {str(e.get("id")): analyze_experiment_for_viz(e, experiments) for e in batch_exps}

    def _run_index() -> int:
        if highlight_experiment_id:
            for i, e in enumerate(batch_exps):
                if str(e.get("id")) == str(highlight_experiment_id):
                    return i
        for i, e in enumerate(batch_exps):
            if e.get("status") == "completed":
                return i
        return 0

    pick = st.selectbox(
        "Run",
        options=batch_exps,
        index=_run_index(),
        format_func=lambda e: format_experiment_viz_option(
            e,
            analyses[str(e.get("id"))],
            highlight=bool(
                highlight_experiment_id and str(e.get("id")) == str(highlight_experiment_id)
            ),
        ),
        key=run_key,
    )
    return pick, batch_id, batch_exps, config


def render_sidebar_experiment_selector(
    experiments: List[Dict[str, Any]],
    workflow: Optional[Dict[str, Any]],
    *,
    key_prefix: str = "sb",
) -> Optional[Dict[str, Any]]:
    """
    Left sidebar: browse past API runs by launch timestamp.

    Sets ``highlight_experiment_id`` (used by checkpoint resume). Main sweep
    plots live in **Results §2–§3** below Step 5 — this picker is optional.
    """
    st.markdown("### Experiments")
    st.caption(
        "Browse past launches grouped by timestamp. "
        "**Results §2–§3** below is the main place for sweep plots."
    )
    if not experiments:
        st.caption("No API runs in the database yet.")
        return None

    if st.button("🔄 Refresh list", key=f"{key_prefix}_refresh", use_container_width=True):
        st.rerun()

    mode = (workflow or {}).get("uncertainty_config", {}).get("sweep_mode", "quick")
    batch_labels = _batch_label_map(experiments, mode=mode)
    if not batch_labels:
        st.caption("No experiments to list.")
        return None

    highlight = st.session_state.get("highlight_experiment_id")
    default_batch_idx = len(batch_labels) - 1
    if highlight:
        for i, (_, exps, _) in enumerate(batch_labels.values()):
            if any(str(e.get("id")) == str(highlight) for e in exps):
                default_batch_idx = i
                break

    batch_label = st.selectbox(
        "Launch group",
        options=list(batch_labels.keys()),
        index=default_batch_idx,
        key=f"{key_prefix}_batch",
        help="Runs that share the same launch timestamp (e.g. Fig 3+4 from **Run both**).",
    )
    pick, batch_id, _batch_exps, config = _resolve_selection(
        experiments,
        highlight_experiment_id=highlight,
        batch_label=batch_label,
        batch_labels=batch_labels,
        run_key=f"{key_prefix}_run",
    )
    if not pick:
        return None

    st.session_state["highlight_experiment_id"] = str(pick.get("id"))
    st.session_state["uqlab_viz_batch_id"] = batch_id

    type_labels = {
        "1d_epistemic": "Under-train sweep (Fig 3 style)",
        "1d_aleatoric": "Label-noise sweep (Fig 4 style)",
        "2d_grid": "Both axes vary (unusual)",
        "single_point": "Single run",
    }
    st.caption(f"**{type_labels.get(config['type'], config['type'])}**")
    n_epi_runs, n_alea_runs = _count_arms(_batch_exps)
    st.caption(f"Fig 3 runs: {n_epi_runs} · Fig 4 runs: {n_alea_runs}")
    if n_epi_runs and not n_alea_runs:
        st.caption("Only Fig 3 arm in this group — launch Fig 4 for the paired paper plot.")
    elif n_alea_runs and not n_epi_runs:
        st.caption("Only Fig 4 arm in this group — launch Fig 3 for the paired paper plot.")

    on_disk = _experiment_results_dir(
        str(pick.get("id")),
        results_path=pick.get("results_path"),
    )
    if not (on_disk / "summary.json").is_file() and not (on_disk / "results.pt").is_file():
        st.caption("⚠️ No on-disk results (check `/tmp` vs `data/` path)")

    if st.button("Checkpoint arsenal", key=f"{key_prefix}_arsenal", use_container_width=True):
        from uqlab.ui_components.results.checkpoint_arsenal_viz import open_checkpoint_arsenal_dialog

        open_checkpoint_arsenal_dialog(experiments, workflow, key_prefix=f"{key_prefix}_arsenal")

    return pick
