"""
Shared utilities package.

Merges general helpers (``core``), classification experiment helpers, and tracking.
Replaces the old split between ``shared/utils.py`` and ``shared/utils/`` that broke
``from uqlab.shared.utils import Timer`` (the directory shadowed the module).
"""

from .core import (
    Timer,
    batch_to_device,
    ensure_dir,
    format_number,
    format_time,
    get_device,
    get_file_hash,
    get_logger,
    get_string_hash,
    load_json,
    load_pickle,
    load_yaml,
    retry,
    safe_execute,
    save_json,
    save_pickle,
    save_yaml,
    set_seed,
    setup_logging,
    timeit,
    to_numpy,
    to_tensor,
    truncate_string,
    validate_non_negative,
    validate_positive,
    validate_range,
)
from .classification import auto_device, dino_transform
from .tracking import ExperimentTracker

__all__ = [
    "Timer",
    "ExperimentTracker",
    "auto_device",
    "batch_to_device",
    "dino_transform",
    "ensure_dir",
    "format_number",
    "format_time",
    "get_device",
    "get_file_hash",
    "get_logger",
    "get_string_hash",
    "load_json",
    "load_pickle",
    "load_yaml",
    "retry",
    "safe_execute",
    "save_json",
    "save_pickle",
    "save_yaml",
    "set_seed",
    "setup_logging",
    "timeit",
    "to_numpy",
    "to_tensor",
    "truncate_string",
    "validate_non_negative",
    "validate_positive",
    "validate_range",
]
