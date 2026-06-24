"""Unified campaign grouping for results panel and sidebar selector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from uqlab.ui_components.grouping.campaign_format import (
    campaign_date_from_batch,
    representative_experiment_id,
)
from uqlab_orchestrator.sweep_groups import group_experiments_intelligently
from uqlab.ui_components.selectors.smart_experiment_selector import (
    detect_experiment_configuration,
    group_experiments_for_selection,
)


def group_experiments_for_results(
    experiments: List[Dict[str, Any]],
    *,
    workflow: Optional[Dict[str, Any]] = None,
    min_config_group_size: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (campaign_groups, config_sweep_groups).

    campaign_groups: timestamp / paper-batch oriented (smart selector)
    config_sweep_groups: config-similarity fallback for legacy run names
    """
    mode = (workflow or {}).get("uncertainty_config", {}).get("sweep_mode", "quick")
    by_ts = group_experiments_for_selection(experiments)

    campaign_groups: List[Dict[str, Any]] = []
    assigned_ids: set[str] = set()

    for ts, batch_exps in sorted(by_ts.items(), key=lambda kv: kv[0]):
        if not batch_exps:
            continue
        config_meta = detect_experiment_configuration(batch_exps, mode=mode)
        type_labels = {
            "1d_epistemic": "Fig. 3 epistemic",
            "1d_aleatoric": "Fig. 4 aleatoric",
            "2d_grid": "2D grid",
            "single_point": "Single run",
        }
        rep_id = representative_experiment_id(batch_exps)
        date_label = campaign_date_from_batch(ts, batch_exps)
        label = f"{date_label} · {rep_id} · {type_labels.get(config_meta['type'], config_meta['type'])}"
        campaign_groups.append(
            {
                "id": ts,
                "label": label,
                "type": config_meta["type"],
                "experiments": batch_exps,
                "config_meta": config_meta,
            }
        )
        for exp in batch_exps:
            assigned_ids.add(str(exp.get("id")))

    remaining = [e for e in experiments if str(e.get("id")) not in assigned_ids]
    config_sweep_groups, _standalone = group_experiments_intelligently(
        remaining,
        min_group_size=min_config_group_size,
    )
    return campaign_groups, config_sweep_groups


def selected_campaign_experiments(
    campaign_groups: List[Dict[str, Any]],
    *,
    highlight_experiment_id: Optional[str] = None,
    batch_id: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Resolve experiment list for the active sidebar campaign."""
    if batch_id:
        for group in campaign_groups:
            if group["id"] == batch_id:
                return group["experiments"]
    if highlight_experiment_id:
        for group in campaign_groups:
            if any(str(e.get("id")) == str(highlight_experiment_id) for e in group["experiments"]):
                return group["experiments"]
    if campaign_groups:
        return campaign_groups[-1]["experiments"]
    return None
