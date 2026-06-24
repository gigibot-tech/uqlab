"""Selection UI components (dataset, model, experiment campaigns)."""

from __future__ import annotations

from typing import Any

__all__ = ["render_sidebar_experiment_selector"]


def __getattr__(name: str) -> Any:
    if name == "render_sidebar_experiment_selector":
        from .smart_experiment_selector import render_sidebar_experiment_selector

        return render_sidebar_experiment_selector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
