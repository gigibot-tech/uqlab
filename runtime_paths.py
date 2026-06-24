"""
Persistent data locations for UQLab (experiments, SQLite, caches).

Override with env ``UQLAB_DATA_DIR`` (absolute path recommended).

Default: ``<repo>/data/`` — survives backend/Streamlit restarts (unlike ``/tmp``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parents[2]


def repository_root() -> Path:
    """Checkout root (directory that contains ``src/``)."""
    if _REPO_ROOT.name == "src":
        return _REPO_ROOT.parent
    return _REPO_ROOT


def data_root() -> Path:
    """Root directory for DB, experiment outputs, and optional local artifacts."""
    raw = os.environ.get("UQLAB_DATA_DIR", "").strip()
    root = Path(raw).expanduser() if raw else _REPO_ROOT / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def sqlite_db_path() -> Path:
    return data_root() / "uqlab.db"


def experiments_root() -> Path:
    root = data_root() / "experiments"
    root.mkdir(parents=True, exist_ok=True)
    return root


def experiment_dir(experiment_id: Union[str, UUID]) -> Path:
    return experiments_root() / str(experiment_id)


def experiment_results_dir(experiment_id: Union[str, UUID]) -> Path:
    return experiment_dir(experiment_id) / "results"


def resolve_experiment_results_dir(
    experiment_id: Union[str, UUID],
    *,
    results_path: str | Path | None = None,
) -> Path:
    """
    Find on-disk results for an API experiment id.

    Checks DB ``results_path``, then ``data/experiments/<id>/results``,
    then legacy ``/tmp/uqlab_experiments/<id>/results``.
    """
    eid = str(experiment_id)
    candidates: list[Path] = []

    if results_path:
        raw = Path(results_path)
        candidates.append(raw if raw.name == "results" else raw / "results")

    candidates.append(experiment_results_dir(eid))
    candidates.append(Path("/tmp/uqlab_experiments") / eid / "results")

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and (
            (path / "summary.json").is_file()
            or (path / "results.pt").is_file()
            or (path / "per_sample_signals.csv").is_file()
        ):
            return path

    return experiment_results_dir(eid)


def batch_root(batch_id: Union[str, UUID]) -> Path:
    return experiments_root() / f"batch_{batch_id}"
