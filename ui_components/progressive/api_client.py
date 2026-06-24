"""Shared API helpers for the progressive Streamlit app."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import requests


def fetch_experiments(
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    *,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """Load all experiments from the no-auth API endpoint."""
    response = requests.get(
        f"{api_base_url}/api/v1/experiments/no-auth",
        headers=get_headers_func(),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def bulk_delete_experiments_by_status(
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    experiments: List[Dict[str, Any]],
    status: str,
    *,
    timeout: int = 10,
) -> int:
    """Delete all experiments matching *status*; returns count deleted."""
    deleted = 0
    for exp in experiments:
        if exp.get("status") != status:
            continue
        try:
            response = requests.delete(
                f"{api_base_url}/api/v1/experiments/no-auth/{exp['id']}",
                headers=get_headers_func(),
                timeout=timeout,
            )
            if response.status_code == 200:
                deleted += 1
        except Exception:
            pass
    return deleted


def fetch_recoverability(
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    *,
    status: str | None = None,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """Load recoverability assessments from the API."""
    params: Dict[str, Any] = {}
    if status:
        params["status"] = status
    response = requests.get(
        f"{api_base_url}/api/v1/experiments/no-auth/recoverability",
        headers=get_headers_func(),
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def recover_experiments_batch(
    api_base_url: str,
    get_headers_func: Callable[[], dict],
    *,
    status: str = "failed",
    tier: str = "zwischen_finalize",
    timeout: int = 600,
) -> Dict[str, Any]:
    """POST recover-batch; returns {recovered, skipped, errors}."""
    response = requests.post(
        f"{api_base_url}/api/v1/experiments/no-auth/recover-batch",
        headers=get_headers_func(),
        json={"status": status, "tier": tier},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def summarize_recoverability_tiers(
    entries: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Count recoverability tiers from API entries."""
    counts: Dict[str, int] = {}
    for entry in entries:
        tier = str(entry.get("tier") or "none")
        counts[tier] = counts.get(tier, 0) + 1
    return counts
