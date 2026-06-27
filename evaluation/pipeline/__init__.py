"""Backward-compat shim — prefer direct imports from new homes.

- Data setup: ``uqlab.data.setup``
- Runner phases: ``uqlab.runner.phases``
- Reporting: ``uqlab.evaluation.reporting``
- Metrics: ``uqlab.evaluation.metrics``
"""

from __future__ import annotations

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ExperimentDataContext": ("uqlab.data.setup", "ExperimentDataContext"),
    "PilotDataRequest": ("uqlab.data.setup", "PilotDataRequest"),
    "parse_pilot_data_request": ("uqlab.data.setup", "parse_pilot_data_request"),
    "prepare_experiment_data": ("uqlab.data.setup", "prepare_experiment_data"),
    "RunConfigView": ("uqlab.runner.phases.config_view", "RunConfigView"),
    "apply_data_context": ("uqlab.runner.phases.config_view", "apply_data_context"),
    "extract_run_config": ("uqlab.runner.phases.config_view", "extract_run_config"),
    "print_dataset_loaded": ("uqlab.runner.phases.config_view", "print_dataset_loaded"),
    "print_experiment_configuration": ("uqlab.runner.phases.config_view", "print_experiment_configuration"),
    "require_complete_config": ("uqlab.runner.phases.config_view", "require_complete_config"),
    "validate_eval_splits": ("uqlab.runner.phases.config_view", "validate_eval_splits"),
    "EvalSignalConfig": ("uqlab.runner.phases.eval_signal_config", "EvalSignalConfig"),
    "collect_uncertainty_signals": ("uqlab.runner.phases.eval", "collect_uncertainty_signals"),
    "score_uncertainty_signals": ("uqlab.runner.phases.eval", "score_uncertainty_signals"),
    "RecoverabilityReport": ("uqlab.runner.phases.recovery", "RecoverabilityReport"),
    "assess_run_recovery": ("uqlab.runner.phases.recovery", "assess_run_recovery"),
    "finalize_run_from_zwischen": ("uqlab.runner.phases.recovery", "finalize_run_from_zwischen"),
    "recover_run_on_disk": ("uqlab.runner.phases.recovery", "recover_run_on_disk"),
    "sync_run_from_disk": ("uqlab.runner.phases.recovery", "sync_run_from_disk"),
    "SweepLinePlotPayload": ("uqlab.evaluation.reporting.sweep_line_plot", "SweepLinePlotPayload"),
    "build_sweep_line_plot": ("uqlab.evaluation.reporting.sweep_line_plot", "build_sweep_line_plot"),
    "default_signal_for_sweep": ("uqlab.evaluation.reporting.sweep_line_plot", "default_signal_for_sweep"),
    "list_plottable_signals": ("uqlab.evaluation.reporting.sweep_line_plot", "list_plottable_signals"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    from importlib import import_module

    mod = import_module(mod_name)
    return getattr(mod, attr)
