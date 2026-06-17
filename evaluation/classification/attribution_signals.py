"""Shim: ``uqlab.classification.attribution_signals`` → ``uqlab.evaluation.signals.attribution``."""

import importlib

_attribution = importlib.import_module("uqlab.evaluation.signals.attribution")

build_fast_pilot_signal_table = _attribution.build_fast_pilot_signal_table
compute_attribution_structure_signals = _attribution.compute_attribution_structure_signals

__all__ = [
    "build_fast_pilot_signal_table",
    "compute_attribution_structure_signals",
]
