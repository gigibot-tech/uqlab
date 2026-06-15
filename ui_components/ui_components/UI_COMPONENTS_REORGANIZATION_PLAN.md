# UI Components Reorganization Plan

## Current Structure (Flat - 24 files)

```
src/uqlab/ui_components/
├── __init__.py
├── config_types.py
├── correlation_analysis.py
├── data_overlap_analysis.py
├── dataset.py
├── experiment_config.py
├── experiment_sweep_context.py
├── experiment_validation.py
├── experiment_viz_inspector.py
├── heatmap_visualization.py
├── hypothesis_validation.py
├── model_selector.py
├── paper_sweep_viz.py
├── per_sample_signals_viz.py
├── results.py
├── signal_diagnostic_viz.py
├── signal_sweep_paper_viz.py
├── signal_visualization.py
├── smart_experiment_selector.py
├── sweep_campaign.py
├── unified_builder.py
├── uq_benchmarks.py
├── utils.py
├── validation_runner.py
├── validation_visualization.py
└── legacy/
```

## Proposed Structure (Organized by Function)

```
src/uqlab/ui_components/
├── __init__.py                          # Re-exports for backward compatibility
├── utils.py                             # Shared utilities (keep at root)
│
├── selectors/                           # 🎯 Selection & Navigation Components
│   ├── __init__.py
│   ├── smart_experiment_selector.py    # Smart batch/run selection with 1D/2D detection
│   ├── experiment_viz_inspector.py     # Experiment inspection and navigation
│   ├── model_selector.py               # Model architecture selection
│   └── dataset.py                      # Dataset selection and configuration
│
├── visualization/                       # 📊 Visualization Components
│   ├── __init__.py
│   ├── signals/                        # Signal-specific visualizations
│   │   ├── __init__.py
│   │   ├── signal_visualization.py     # Core signal visualization
│   │   ├── signal_diagnostic_viz.py    # Per-signal diagnostic panels
│   │   ├── signal_sweep_paper_viz.py   # Signal sweep paper-style plots
│   │   └── per_sample_signals_viz.py   # Per-sample signal analysis
│   │
│   ├── sweeps/                         # Sweep visualizations
│   │   ├── __init__.py
│   │   ├── paper_sweep_viz.py          # Paper-style sweep plots
│   │   ├── heatmap_visualization.py    # 2D heatmap visualizations
│   │   └── sweep_campaign.py           # Paired sweep campaign visualization
│   │
│   ├── analysis/                       # Analysis visualizations
│   │   ├── __init__.py
│   │   ├── correlation_analysis.py     # Correlation analysis plots
│   │   ├── data_overlap_analysis.py    # Data overlap visualization
│   │   └── uq_benchmarks.py            # UQ benchmark comparisons
│   │
│   └── validation/                     # Validation visualizations
│       ├── __init__.py
│       ├── validation_visualization.py # Validation result visualization
│       └── hypothesis_validation.py    # Hypothesis validation UI
│
├── config/                              # ⚙️ Configuration Components
│   ├── __init__.py
│   ├── experiment_config.py            # Experiment configuration builder
│   ├── config_types.py                 # Configuration type definitions
│   └── experiment_validation.py        # Configuration validation
│
├── orchestration/                       # 🎭 Orchestration & Execution
│   ├── __init__.py
│   ├── experiment_sweep_context.py     # Sweep context and classification
│   ├── validation_runner.py            # Validation execution runner
│   └── unified_builder.py              # Unified experiment builder
│
└── results/                             # 📋 Results & Data Management
    ├── __init__.py
    └── results.py                      # Results fetching and display
```

## File Categorization

### 🎯 Selectors (4 files)
**Purpose**: User interaction for selecting experiments, models, datasets
- `smart_experiment_selector.py` - Smart batch/run selection with 1D/2D detection
- `experiment_viz_inspector.py` - Experiment inspection and navigation
- `model_selector.py` - Model architecture selection
- `dataset.py` - Dataset selection and configuration

### 📊 Visualization (11 files)
**Purpose**: All visualization and plotting components

#### Signals (4 files)
- `signal_visualization.py` - Core signal visualization
- `signal_diagnostic_viz.py` - Per-signal diagnostic panels
- `signal_sweep_paper_viz.py` - Signal sweep paper-style plots
- `per_sample_signals_viz.py` - Per-sample signal analysis

#### Sweeps (3 files)
- `paper_sweep_viz.py` - Paper-style sweep plots
- `heatmap_visualization.py` - 2D heatmap visualizations
- `sweep_campaign.py` - Paired sweep campaign visualization

#### Analysis (3 files)
- `correlation_analysis.py` - Correlation analysis plots
- `data_overlap_analysis.py` - Data overlap visualization
- `uq_benchmarks.py` - UQ benchmark comparisons

#### Validation (2 files)
- `validation_visualization.py` - Validation result visualization
- `hypothesis_validation.py` - Hypothesis validation UI

### ⚙️ Config (3 files)
**Purpose**: Configuration building and validation
- `experiment_config.py` - Experiment configuration builder
- `config_types.py` - Configuration type definitions
- `experiment_validation.py` - Configuration validation

### 🎭 Orchestration (3 files)
**Purpose**: Experiment execution and coordination
- `experiment_sweep_context.py` - Sweep context and classification
- `validation_runner.py` - Validation execution runner
- `unified_builder.py` - Unified experiment builder

### 📋 Results (1 file)
**Purpose**: Results fetching and display
- `results.py` - Results fetching and display

### 🔧 Utilities (1 file - stays at root)
- `utils.py` - Shared utilities

## Migration Strategy

### Phase 1: Create Directory Structure
```bash
mkdir -p src/uqlab/ui_components/{selectors,visualization/{signals,sweeps,analysis,validation},config,orchestration,results}
```

### Phase 2: Move Files to New Locations
Move each file to its designated subdirectory while maintaining git history.

### Phase 3: Update __init__.py Files
Create `__init__.py` in each subdirectory with appropriate re-exports.

### Phase 4: Update Root __init__.py
Update root `__init__.py` to re-export from new locations for backward compatibility.

### Phase 5: Update Internal Imports
Update relative imports within moved files to reflect new structure.

### Phase 6: Verify and Test
- Verify all imports work
- Test Streamlit apps
- Ensure backward compatibility

## Backward Compatibility

The root `__init__.py` will re-export all components:

```python
# src/uqlab/ui_components/__init__.py

# Selectors
from .selectors.smart_experiment_selector import *
from .selectors.experiment_viz_inspector import *
from .selectors.model_selector import *
from .selectors.dataset import *

# Visualization - Signals
from .visualization.signals.signal_visualization import *
from .visualization.signals.signal_diagnostic_viz import *
# ... etc

# This ensures existing imports still work:
# from uqlab.ui_components import smart_experiment_selector
```

## Benefits

### 1. **Improved Discoverability**
- Clear categorization makes it easy to find components
- Logical grouping by function

### 2. **Better Maintainability**
- Related files are co-located
- Easier to understand component relationships
- Reduced cognitive load

### 3. **Scalability**
- Easy to add new components to appropriate categories
- Clear patterns for where new files belong

### 4. **Cleaner Imports**
- More explicit import paths
- Better IDE autocomplete
- Easier to understand dependencies

### 5. **Backward Compatible**
- Existing code continues to work
- Gradual migration possible
- No breaking changes

## Implementation Priority

### High Priority (Core Organization)
1. ✅ Create directory structure
2. ✅ Move selector files (most cohesive group)
3. ✅ Move visualization files (largest group)
4. ✅ Update __init__.py files

### Medium Priority (Polish)
5. ⏳ Update internal imports
6. ⏳ Add subdirectory documentation
7. ⏳ Update main documentation

### Low Priority (Optional)
8. ⏳ Further split large files if needed
9. ⏳ Add type hints to __init__.py exports
10. ⏳ Create component usage examples

## Example: Before vs After

### Before (Flat Structure)
```python
from uqlab.ui_components import smart_experiment_selector
from uqlab.ui_components import signal_visualization
from uqlab.ui_components import experiment_config
```

### After (Organized Structure)
```python
# New explicit imports (preferred)
from uqlab.ui_components.selectors import smart_experiment_selector
from uqlab.ui_components.visualization.signals import signal_visualization
from uqlab.ui_components.config import experiment_config

# Old imports still work (backward compatible)
from uqlab.ui_components import smart_experiment_selector
from uqlab.ui_components import signal_visualization
from uqlab.ui_components import experiment_config
```

## Next Steps

1. **Review and approve** this organization plan
2. **Execute Phase 1**: Create directory structure
3. **Execute Phase 2**: Move files systematically
4. **Execute Phase 3-6**: Update imports and verify
5. **Document** the new structure in README

---

**Status**: 📋 Plan Ready for Review  
**Estimated Effort**: 2-3 hours  
**Risk**: Low (backward compatible)  
**Impact**: High (better organization)