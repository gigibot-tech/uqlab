"""EK-FAK signal pipeline: catalog (light) → registry/sources (heavy, explicit import)."""

from __future__ import annotations

from .catalog import *  # noqa: F403
from .formulas import *  # noqa: F403

_ARCHIVED = (
    "SignalCalculator was moved to archive/legacy_src/evaluation/signal_calculator.py. "
    "Use METRICS and sources instead."
)


def __getattr__(name: str):
    if name == "SignalCalculator":
        raise ImportError(_ARCHIVED)
    for mod_name in ("registry", "primitives", "sources", "attribution"):
        from importlib import import_module

        mod = import_module(f"{__name__}.{mod_name}")
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
