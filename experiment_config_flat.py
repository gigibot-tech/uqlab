"""
Flatten grouped experiment YAML and resolve sweep parameters for plotting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

ALEATORIC_PARAM = "aleatoric_noise_percentage"
EPISTEMIC_PARAM = "under_train_per_class"


def flatten_experiment_config(cfg: Any) -> dict[str, Any]:
    """Merge top-level keys with nested ``data`` / ``model`` / … sections."""
    if not cfg:
        return {}
    if isinstance(cfg, str):
        try:
            cfg = yaml.safe_load(cfg)
        except Exception:
            return {}
    if not isinstance(cfg, dict):
        return {}

    flat = dict(cfg)
    for section in ("data", "model", "training", "evaluation", "paths"):
        block = cfg.get(section)
        if isinstance(block, dict):
            for key, value in block.items():
                flat.setdefault(key, value)
    return flat


def normalize_noise_percent(value: Any) -> Optional[float]:
    """Map 0–1 fractions to 0–100 percent (training script expects percent)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if 0 < v <= 1.0:
        return v * 100.0
    return v


def _summary_config_from_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return flatten_experiment_config(summary.get("config"))


def resolve_experiment_param(
    exp: dict[str, Any],
    param: str,
    *,
    parse_from_name: Callable[[str, str], Optional[int]] | None = None,
) -> Optional[float]:
    """
    Read a sweep parameter from flat config, grouped YAML, ``summary.json``, or name.
    """
    flat: dict[str, Any] = {}
    for key in ("config", "config_yaml"):
        raw = exp.get(key)
        if raw:
            flat = flatten_experiment_config(raw)
            if flat:
                break

    if not flat and exp.get("id"):
        from uqlab.runtime_paths import resolve_experiment_results_dir

        run_dir = resolve_experiment_results_dir(
            str(exp["id"]),
            results_path=exp.get("results_path"),
        )
        flat = _summary_config_from_run(run_dir)

    val = flat.get(param)
    if val is not None:
        if param == ALEATORIC_PARAM:
            return normalize_noise_percent(val)
        try:
            return float(val)
        except (TypeError, ValueError):
            pass

    name = exp.get("name") or ""
    if parse_from_name and name:
        parsed = parse_from_name(name, param)
        if parsed is not None:
            return float(parsed)

    return None
