"""Vendored disentanglement_error metric (see UPSTREAM.md).

Heavy imports (torch, full error metric) are lazy so Streamlit/orchestrator can
import ``Config`` without pulling the ML stack.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Config",
    "DisentanglingModel",
    "RunResults",
    "calculate_disentanglement_error",
    "calculate_disentanglement_error_torch",
    "json_results_to_df",
]


def __getattr__(name: str) -> Any:
    if name == "Config":
        from uqlab.vendor.disentanglement_error.util import Config

        return Config
    if name == "RunResults":
        from uqlab.vendor.disentanglement_error.util import RunResults

        return RunResults
    if name == "json_results_to_df":
        from uqlab.vendor.disentanglement_error.util import json_results_to_df

        return json_results_to_df
    if name == "DisentanglingModel":
        from uqlab.vendor.disentanglement_error.disentangling_model import DisentanglingModel

        return DisentanglingModel
    if name == "calculate_disentanglement_error":
        from uqlab.vendor.disentanglement_error.error_metric import calculate_disentanglement_error

        return calculate_disentanglement_error
    if name == "calculate_disentanglement_error_torch":
        from uqlab.vendor.disentanglement_error.error_metric import calculate_disentanglement_error_torch

        return calculate_disentanglement_error_torch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
