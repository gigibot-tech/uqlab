"""Human-readable sweep campaign labels (date, short id, sweep summary)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

_CAMPAIGN_TS_RE = re.compile(r"(\d{8}_\d{6})")
_TS_PARTS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")


def short_experiment_id(exp_id: Any, *, length: int = 8) -> str:
    """First ``length`` hex chars of a UUID (no dashes)."""
    raw = str(exp_id or "").replace("-", "")
    return raw[:length] if raw else "?"


def format_campaign_timestamp(ts: str) -> str:
    """
    Format ``YYYYMMDD_HHMMSS`` as ``Mon DD · HH:MM`` (no year).

    Falls back to the raw string when parsing fails.
    """
    m = _TS_PARTS_RE.match((ts or "").strip())
    if not m:
        return ts or "?"
    year, month, day, hour, minute, _sec = (int(x) for x in m.groups())
    try:
        dt = datetime(year, month, day, hour, minute)
        return dt.strftime("%b %d · %H:%M")
    except ValueError:
        return ts


def extract_campaign_timestamp(*, sweep_group_id: str = "", name: str = "") -> Optional[str]:
    """Pull ``YYYYMMDD_HHMMSS`` from sweep group id or experiment name."""
    for text in (sweep_group_id, name):
        m = _CAMPAIGN_TS_RE.search(text or "")
        if m:
            return m.group(1)
    return None


def _format_created_at(created_at: Any) -> Optional[str]:
    if created_at is None:
        return None
    if isinstance(created_at, datetime):
        dt = created_at
    elif isinstance(created_at, str):
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt.strftime("%b %d · %H:%M")


def campaign_date_label(
    group: Dict[str, Any],
    *,
    experiments: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Best-effort campaign date without year."""
    exps = experiments if experiments is not None else (group.get("experiments") or [])
    ts = extract_campaign_timestamp(sweep_group_id=str(group.get("sweep_group_id") or ""))
    if not ts and exps:
        for exp in exps:
            ts = extract_campaign_timestamp(name=str(exp.get("name") or ""))
            if ts:
                break
    if ts:
        return format_campaign_timestamp(ts)
    created = group.get("created_at")
    if created is None and exps:
        times = [e.get("created_at") for e in exps if e.get("created_at")]
        created = min(times) if times else None
    formatted = _format_created_at(created)
    return formatted or "?"


def representative_experiment_id(experiments: List[Dict[str, Any]]) -> str:
    if not experiments:
        return "?"
    anchor = min(experiments, key=lambda e: str(e.get("created_at") or ""))
    return short_experiment_id(anchor.get("id"))


def campaign_date_from_batch(
    batch_id: str,
    experiments: List[Dict[str, Any]],
) -> str:
    """Date label from launch timestamp key or earliest experiment ``created_at``."""
    if _TS_PARTS_RE.match((batch_id or "").strip()):
        return format_campaign_timestamp(batch_id)
    times = [e.get("created_at") for e in experiments if e.get("created_at")]
    if times:
        formatted = _format_created_at(min(times))
        if formatted:
            return formatted
    return batch_id or "?"


def format_sweep_campaign_header(
    group: Dict[str, Any],
    *,
    experiments: Optional[List[Dict[str, Any]]] = None,
    suffix: str = "",
) -> str:
    """
    ``Jun 15 · a1b2c3d4 · Sweep: noise`` (+ optional suffix).

    Used in Results §2–§3 expanders and sweep analysis picker.
    """
    exps = experiments if experiments is not None else (group.get("experiments") or [])
    date = campaign_date_label(group, experiments=exps)
    cid = representative_experiment_id(exps)
    base = group.get("name", "Sweep")
    parts = [date, cid, base]
    if suffix:
        parts.append(suffix)
    return " · ".join(parts)
