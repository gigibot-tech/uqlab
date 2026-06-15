"""Shim: ``uq_classification.signal_formula_specs`` → ``uqlab.4_evaluation.signals.formulas``."""

import importlib

_formulas = importlib.import_module("uqlab.4_evaluation.signals.formulas")

build_signal_formula_manifest = _formulas.build_signal_formula_manifest
fast_pilot_signal_formula_specs = _formulas.fast_pilot_signal_formula_specs

__all__ = ["build_signal_formula_manifest", "fast_pilot_signal_formula_specs"]
