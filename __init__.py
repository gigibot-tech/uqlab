"""UQLab unified UQ validation and paper benchmark package."""

__version__ = "0.1.0"

__all__ = [
    "SIGNAL_NAMES",
    "UNIFIED_COLUMNS",
    "UnifiedRow",
    "adapt_pytorch_metrics_csv",
    "append_paper_row",
    "append_pytorch_row",
    "load_unified_metrics",
]


def __getattr__(name: str):
    if name in __all__:
        from uqlab import results_io

        return getattr(results_io, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
