"""
Data layer — datasets, splits, loaders, and experiment data setup.

Import submodules directly to avoid eager torch imports. Key modules:
``dataset_registry``, ``experiment_loader``, ``class_regions``, ``setup``, ``packs``.

Layering: see ``data/README.md``.
"""

from __future__ import annotations

_LAZY = {
    "ExperimentDataContext": ("setup", "ExperimentDataContext"),
    "PilotDataRequest": ("setup", "PilotDataRequest"),
    "parse_pilot_data_request": ("setup", "parse_pilot_data_request"),
    "prepare_experiment_data": ("setup", "prepare_experiment_data"),
    "prepare_run_data_context": ("packs", "prepare_run_data_context"),
    "get_data_loading_mode": ("packs", "get_data_loading_mode"),
    "prepare_eval_tensors": ("packs", "prepare_eval_tensors"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    from importlib import import_module

    mod = import_module(f"{__name__}.{mod_name}")
    return getattr(mod, attr)
