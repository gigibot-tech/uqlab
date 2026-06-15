"""
Legacy UI Components

These components support the legacy batch experiment system which remains
functional for production parameter sweeps. While deprecated for new development,
they provide stable, tested batch experiment functionality.

Use the Unified Builder for new single experiments and quick tests.
Use these legacy components for systematic parameter sweeps and batch experiments.
"""

from .batch_config import (
    render_batch_sweep_config,
    render_batch_base_config,
)

from .batch_2d_sweep import (
    render_2d_sweep_config,
    render_2d_heatmap,
    render_2d_results_analysis,
)

__all__ = [
    # 1D Batch Experiments
    'render_batch_sweep_config',
    'render_batch_base_config',
    
    # 2D Batch Experiments
    'render_2d_sweep_config',
    'render_2d_heatmap',
    'render_2d_results_analysis',
]

# Made with Bob
