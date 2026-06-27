"""Plot payloads, campaign PDF assembly, and run output writers."""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "SweepLinePlotPayload": ("sweep_line_plot", "SweepLinePlotPayload"),
    "build_sweep_line_plot": ("sweep_line_plot", "build_sweep_line_plot"),
    "default_signal_for_sweep": ("sweep_line_plot", "default_signal_for_sweep"),
    "list_plottable_signals": ("sweep_line_plot", "list_plottable_signals"),
    "build_results_markdown": ("result_writers", "build_results_markdown"),
    "save_per_sample_csv": ("result_writers", "save_per_sample_csv"),
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
