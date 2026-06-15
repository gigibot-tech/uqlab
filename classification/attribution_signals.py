"""Shim: ``uqlab.classification.attribution_signals`` → ``uqlab.4_evaluation.signals.attribution``."""

import importlib

_attribution = importlib.import_module("uqlab.4_evaluation.signals.attribution")

build_fast_pilot_signal_table = _attribution.build_fast_pilot_signal_table
compute_attribution_structure_signals = _attribution.compute_attribution_structure_signals

__all__ = [
    "build_fast_pilot_signal_table",
    "compute_attribution_structure_signals",
]
