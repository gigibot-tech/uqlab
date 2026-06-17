"""
Discover completed training runs on disk when the API database has no rows.

API experiments live in SQLite; artifacts live under ``data/experiments/`` or
legacy ``/tmp/uqlab_experiments/``. Local preset sweeps use ``results/validation/``
and are not included here.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from uqlab.runtime_paths import experiments_root


def _has_result_artifacts(results_dir: Path) -> bool:
    return (
        (results_dir / "summary.json").is_file()
        or (results_dir / "results.pt").is_file()
        or (results_dir / "per_sample_signals.csv").is_file()
    )


def _stable_experiment_id(results_dir: Path) -> str:
    parent_name = results_dir.parent.name
    try:
        uuid.UUID(parent_name)
        return parent_name
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(results_dir.resolve())))


def _infer_name(results_dir: Path, summary: dict[str, Any]) -> str:
    cfg_file = (summary.get("config") or {}).get("config_file")
    if cfg_file:
        return Path(str(cfg_file)).stem
    return results_dir.parent.name


def _record_from_results_dir(results_dir: Path) -> dict[str, Any] | None:
    if not _has_result_artifacts(results_dir):
        return None

    summary_path = results_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    eid = _stable_experiment_id(results_dir)
    name = _infer_name(results_dir, summary)
    status = "completed" if summary_path.is_file() else "running"

    ts_source = summary_path if summary_path.is_file() else results_dir
    mtime = ts_source.stat().st_mtime
    created_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return {
        "id": eid,
        "name": name,
        "status": status,
        "progress": 1.0 if status == "completed" else 0.0,
        "created_at": created_at,
        "started_at": None,
        "completed_at": created_at if status == "completed" else None,
        "error_message": None,
        "aleatoric_auroc": None,
        "epistemic_auroc": None,
        "results_path": str(results_dir.resolve()),
        "best_signals_json": None,
        "_source": "disk",
    }


def _scan_experiment_root(root: Path, records: dict[str, dict[str, Any]]) -> None:
    if not root.is_dir():
        return

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        if child.name.startswith("batch_"):
            batch_runs = child / "experiments"
            if not batch_runs.is_dir():
                continue
            for run_dir in sorted(batch_runs.iterdir()):
                if not run_dir.is_dir():
                    continue
                results_dir = run_dir / "results"
                if not results_dir.is_dir():
                    results_dir = run_dir
                rec = _record_from_results_dir(results_dir)
                if rec and rec["id"] not in records:
                    records[rec["id"]] = rec
            continue

        results_dir = child / "results"
        if not results_dir.is_dir():
            continue
        rec = _record_from_results_dir(results_dir)
        if rec and rec["id"] not in records:
            records[rec["id"]] = rec


def discover_experiments_from_disk() -> list[dict[str, Any]]:
    """Build API-shaped experiment dicts from on-disk ``results/`` folders."""
    records: dict[str, dict[str, Any]] = {}
    _scan_experiment_root(experiments_root(), records)
    _scan_experiment_root(Path("/tmp/uqlab_experiments"), records)
    return list(records.values())


def fetch_experiments_for_ui(
    api_base_url: str,
    get_headers_func,
    *,
    timeout: int = 10,
) -> list[dict[str, Any]] | None:
    """
    Merge API experiments with on-disk runs missing from SQLite.

    Returns ``None`` if the API is unreachable.
    """
    import requests

    try:
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers_func(),
            timeout=timeout,
        )
        response.raise_for_status()
        experiments = list(response.json())
    except Exception:
        return None

    try:
        api_ids = {str(e.get("id")) for e in experiments}
        for record in discover_experiments_from_disk():
            if str(record.get("id")) not in api_ids:
                experiments.append(record)
    except Exception:
        pass

    experiments.sort(key=lambda e: e.get("created_at") or "", reverse=True)
    return experiments
