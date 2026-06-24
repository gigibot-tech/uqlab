"""Vendored disentanglement_error metric (see UPSTREAM.md)."""

from uqlab.vendor.disentanglement_error.disentangling_model import DisentanglingModel
from uqlab.vendor.disentanglement_error.error_metric import (
    calculate_disentanglement_error,
    calculate_disentanglement_error_torch,
)
from uqlab.vendor.disentanglement_error.util import Config, RunResults, json_results_to_df

__all__ = [
    "Config",
    "DisentanglingModel",
    "RunResults",
    "calculate_disentanglement_error",
    "calculate_disentanglement_error_torch",
    "json_results_to_df",
]
