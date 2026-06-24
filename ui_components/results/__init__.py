"""
Results Management Package

Progressive app uses ``experiment_results_panel``; legacy ``results.py`` is archived
under ``dead_code/src/uqlab/ui_components/results/``.
"""

from .training_data_inspection import (
    CIFAR10_CLASSES,
    generate_training_stats_from_config,
    parse_training_data_stats,
    render_training_data_stats,
)

__all__ = [
    "render_training_data_stats",
    "parse_training_data_stats",
    "generate_training_stats_from_config",
    "CIFAR10_CLASSES",
]
