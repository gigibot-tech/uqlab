"""
UI Components Package

This package contains modular UI components for the Streamlit Uncertainty
Quantification application. All components are re-exported here for backward
compatibility with existing imports.

New Directory Structure:
- config/: Configuration components (config_types, experiment_config, experiment_validation)
- selectors/: Selection components (dataset, model_selector)
- visualization/: All visualization components
  - analysis/: Analysis visualizations (correlation, data_overlap, uq_benchmarks)
  - sweeps/: Sweep visualizations (heatmap)
  - validation/: Validation visualizations (hypothesis, validation_viz)
  - signals/: Signal visualizations (signal_diagnostic, signal_viz, per_sample)
- orchestration/: Orchestration components (unified_builder, validation_runner)
- results/: Results display components
- legacy/: Legacy batch configuration components
- utils.py: Helper functions and utilities
"""

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in (_here, *_here.parents):
    if (_p / "pyproject.toml").is_file() and (_p / "scripts").is_dir():
        _root = _p
        break
else:
    _root = _here.parents[3]

_src = _root / "src"
for _entry in (str(_src), str(_root)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Configuration type definitions
from .config.config_types import (
    ValidationConfig,
    SweepConfig,
    EpistemicConfig,
    AleatoricConfig,
    UnifiedBuilderConfig,
)

# Dataset components
from .selectors.dataset import (
    render_dataset_selection,
    render_dataset_comparison,
)

# Experiment configuration components
from .config.experiment_config import (
    render_epistemic_config,
    render_epistemic_strength,
    render_aleatoric_config,
    render_aleatoric_strength,
    render_model_config,
    render_training_config,
    render_evaluation_config,
    render_evaluation_strategy,
    build_base_experiment_config,
)

# Legacy batch configuration components (moved to legacy/ folder)
from .legacy import (
    render_batch_sweep_config,
    render_batch_base_config,
    render_2d_sweep_config,
    render_2d_heatmap,
    render_2d_results_analysis,
)

# Results visualization components
from .results import (
    render_experiment_results,
)

# Signal visualization components
from .visualization.signals.signal_visualization import (
    render_batch_results,
)

# Model selector components
from .selectors.model_selector import (
    render_model_selector,
    render_model_inference_panel,
)

# Data overlap analysis
from .visualization.analysis.data_overlap_analysis import (
    render_data_overlap_analysis,
)

# Experiment validation components
from .config.experiment_validation import (
    render_experiment_type_validation,
    render_validation_summary,
    get_validation_badge,
    validate_sweep_configuration,
)

# Unified builder components (Phase A)
from .orchestration.unified_builder import (
    render_unified_builder,
    render_experiment_execution_panel,
)

# Correlation analysis components (Phase 2)
from .visualization.analysis.correlation_analysis import (
    CorrelationResult,
    ValidationResult,
    calculate_correlation,
    analyze_epistemic_sweep,
    analyze_aleatoric_sweep,
    analyze_2d_grid,
)

# Validation visualization components (Phase 2 & 3)
from .visualization.validation.validation_visualization import (
    render_correlation_scatter,
    render_validation_summary,
    render_correlation_details,
    render_full_validation_report,
    render_compliance_dashboard,
    render_ude_score_badge,
    render_condition_grid,
    render_condition_card,
    render_detailed_statistics,
    render_recommendations,
)

# Hypothesis validation components
from .visualization.validation.hypothesis_validation import (
    render_hypothesis_validation_tab,
)

# Utility components
from .utils import (
    render_configuration_progress,
    render_roc_explanation,
)

# Define what gets exported with "from ui_components import *"
__all__ = [
    # Configuration types
    'ValidationConfig',
    'SweepConfig',
    'EpistemicConfig',
    'AleatoricConfig',
    'UnifiedBuilderConfig',
    # Dataset
    'render_dataset_selection',
    'render_dataset_comparison',
    # Experiment config
    'render_epistemic_config',
    'render_epistemic_strength',
    'render_aleatoric_config',
    'render_aleatoric_strength',
    'render_model_config',
    'render_training_config',
    'render_evaluation_config',
    'render_evaluation_strategy',
    'build_base_experiment_config',
    # Batch config
    'render_batch_sweep_config',
    'render_batch_base_config',
    # 2D Batch sweep
    'render_2d_sweep_config',
    'render_2d_heatmap',
    'render_2d_results_analysis',
    # Results
    'render_experiment_results',
    'render_batch_results',
    # Model selector
    'render_model_selector',
    'render_model_inference_panel',
    # Data overlap analysis
    'render_data_overlap_analysis',
    # Experiment validation
    'render_experiment_type_validation',
    'render_validation_summary',
    'get_validation_badge',
    'validate_sweep_configuration',
    # Unified builder
    'render_unified_builder',
    'render_experiment_execution_panel',
    # Correlation analysis (Phase 2)
    'CorrelationResult',
    'ValidationResult',
    'calculate_correlation',
    'analyze_epistemic_sweep',
    'analyze_aleatoric_sweep',
    'analyze_2d_grid',
    # Validation visualization (Phase 2 & 3)
    'render_correlation_scatter',
    'render_validation_summary',
    'render_correlation_details',
    'render_full_validation_report',
    'render_compliance_dashboard',
    'render_ude_score_badge',
    'render_condition_grid',
    'render_condition_card',
    'render_detailed_statistics',
    'render_recommendations',
    # Hypothesis validation
    'render_hypothesis_validation_tab',
    # Utils
    'render_configuration_progress',
    'render_roc_explanation',
]

# Made with Bob
