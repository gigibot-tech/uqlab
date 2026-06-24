"""Method comparison utilities (Plotly path is live; matplotlib helpers archived)."""

from .method_comparison_plotly import (
    create_method_uncertainty_comparison_figure,
    display_plotly_figure,
    plot_method_uncertainty_comparison,
)

__all__ = [
    "create_method_uncertainty_comparison_figure",
    "display_plotly_figure",
    "plot_method_uncertainty_comparison",
]
