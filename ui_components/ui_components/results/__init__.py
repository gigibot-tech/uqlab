"""
Results Management Package

Re-exports all functions from results.py for backward compatibility.
"""

from .results import (
    render_experiment_results,
    _render_experiment_detail,
    _render_experiment_results_data,
    _render_start_training_buttons,
    _format_best_metric,
)

__all__ = [
    "render_experiment_results",
    "_render_experiment_detail",
    "_render_experiment_results_data",
    "_render_start_training_buttons",
    "_format_best_metric",
]

# Made with Bob
