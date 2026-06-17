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

from .training_data_inspection import (
    render_training_data_stats,
    parse_training_data_stats,
    generate_training_stats_from_config,
    CIFAR10_CLASSES,
)

__all__ = [
    "render_experiment_results",
    "_render_experiment_detail",
    "_render_experiment_results_data",
    "_render_start_training_buttons",
    "_format_best_metric",
    "render_training_data_stats",
    "parse_training_data_stats",
    "generate_training_stats_from_config",
    "CIFAR10_CLASSES",
]

# Made with Bob
