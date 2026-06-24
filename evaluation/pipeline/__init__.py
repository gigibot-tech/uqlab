"""Fast-pilot experiment pipeline stages (import submodules directly; many need torch)."""

from __future__ import annotations

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ExperimentDataContext": ("data_setup", "ExperimentDataContext"),
    "PilotDataRequest": ("data_setup", "PilotDataRequest"),
    "parse_pilot_data_request": ("data_setup", "parse_pilot_data_request"),
    "prepare_experiment_data": ("data_setup", "prepare_experiment_data"),
    "RunConfigView": ("experiment_setup", "RunConfigView"),
    "apply_data_context": ("experiment_setup", "apply_data_context"),
    "extract_run_config": ("experiment_setup", "extract_run_config"),
    "print_dataset_loaded": ("experiment_setup", "print_dataset_loaded"),
    "print_experiment_configuration": ("experiment_setup", "print_experiment_configuration"),
    "require_complete_config": ("experiment_setup", "require_complete_config"),
    "validate_eval_splits": ("experiment_setup", "validate_eval_splits"),
    "EvalSignalConfig": ("eval_signal_config", "EvalSignalConfig"),
    "collect_uncertainty_signals": ("experiment_eval", "collect_uncertainty_signals"),
    "score_uncertainty_signals": ("experiment_eval", "score_uncertainty_signals"),
    "RecoverabilityReport": ("run_recovery", "RecoverabilityReport"),
    "assess_run_recovery": ("run_recovery", "assess_run_recovery"),
    "finalize_run_from_zwischen": ("run_recovery", "finalize_run_from_zwischen"),
    "recover_run_on_disk": ("run_recovery", "recover_run_on_disk"),
    "sync_run_from_disk": ("run_recovery", "sync_run_from_disk"),
    "SweepLinePlotPayload": ("sweep_line_plot", "SweepLinePlotPayload"),
    "build_sweep_line_plot": ("sweep_line_plot", "build_sweep_line_plot"),
    "default_signal_for_sweep": ("sweep_line_plot", "default_signal_for_sweep"),
    "list_plottable_signals": ("sweep_line_plot", "list_plottable_signals"),
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = spec
    from importlib import import_module

    mod = import_module(f"{__name__}.{mod_name}")
    return getattr(mod, attr)
