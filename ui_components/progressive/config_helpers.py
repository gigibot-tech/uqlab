"""
Configuration Helper Functions

Utilities for loading and comparing experiment configurations.
Delegates to experiment_registry for a single source of truth.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from uqlab_orchestrator.experiment_registry import (
    load_experiment_config,
    match_key_from_run_yaml,
)


def load_local_config_yaml(
    experiment_id: str,
    project_root: Path,
) -> Optional[Dict[str, Any]]:
    """Load config.yaml under data/experiments/<id>/ (backward-compatible wrapper)."""
    return load_experiment_config(experiment_id, project_root=project_root)


__all__ = ["load_local_config_yaml", "match_key_from_run_yaml", "load_experiment_config"]
