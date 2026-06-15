o# 6_ui Directory Migration Analysis

## Current Structure

```
src/uqlab/6_ui/
├── __init__.py
├── api_client.py              # API client for backend
├── app.py                     # Main Streamlit app
├── batch_builder.py           # Batch experiment builder
├── correlation_viz.py         # Correlation visualization
├── experiment_builder.py      # Experiment builder
├── results_viewer.py          # Results viewer
├── signal_viewer.py           # Signal viewer
├── sweep_planner.py           # Sweep planner (already updated imports)
├── visualizations.py          # General visualizations
├── apps/
│   ├── __init__.py
│   └── classification_viz.py # Classification visualization app
└── visualization/
    ├── __init__.py
    └── decision_boundaries.py # Decision boundary visualization
```

## Analysis

### Purpose of 6_ui
The `6_ui` directory appears to be an **older/alternative UI implementation** that predates the current `ui_components` structure. It contains:
- Standalone Streamlit apps (`app.py`, `apps/classification_viz.py`)
- UI builders (`experiment_builder.py`, `batch_builder.py`)
- Viewers (`results_viewer.py`, `signal_viewer.py`)
- Visualizations (`visualizations.py`, `correlation_viz.py`, `decision_boundaries.py`)
- API client (`api_client.py`)

### Comparison with ui_components

| Feature | 6_ui | ui_components |
|---------|------|---------------|
| **Purpose** | Standalone apps | Modular components |
| **Organization** | Flat + 2 subdirs | Hierarchical (9 subdirs) |
| **Usage** | Direct app execution | Imported by apps |
| **Maturity** | Older implementation | Current/active |
| **Duplication** | Some overlap | Comprehensive |

### Recommendation: **DO NOT MIGRATE**

**Reasons**:
1. **Different Purpose**: `6_ui` contains standalone apps, while `ui_components` contains reusable components
2. **Potential Duplication**: Moving would create confusion about which implementation to use
3. **Active Development**: `ui_components` is the current, actively maintained structure
4. **Backward Compatibility**: `6_ui` might be used by legacy code or scripts

### Better Approach: **Deprecation Path**

Instead of migrating, follow this deprecation strategy:

#### Phase 1: Document Status (Immediate)
1. Add deprecation notice to `6_ui/__init__.py`
2. Document which `ui_components` replace which `6_ui` files
3. Create migration guide for any active users

#### Phase 2: Identify Active Usage (Next)
1. Search codebase for imports from `6_ui`
2. Identify which files are actively used
3. Create replacement mapping

#### Phase 3: Gradual Migration (Future)
1. Update active code to use `ui_components` instead
2. Mark unused `6_ui` files as deprecated
3. Eventually remove when no longer referenced

## File-by-File Analysis

### Standalone Apps (Keep in 6_ui)
- `app.py` - Main Streamlit app (standalone)
- `apps/classification_viz.py` - Classification app (standalone)

**Reason**: These are complete applications, not components

### Builders (Potential Duplication)
- `experiment_builder.py` - Similar to `ui_components/config/experiment_config.py`
- `batch_builder.py` - Similar to `ui_components/legacy/batch_config.py`

**Action**: Document equivalents, deprecate if unused

### Viewers (Potential Duplication)
- `results_viewer.py` - Similar to `ui_components/results/results.py`
- `signal_viewer.py` - Similar to `ui_components/visualization/signals/`

**Action**: Document equivalents, deprecate if unused

### Visualizations (Potential Duplication)
- `visualizations.py` - Generic visualizations
- `correlation_viz.py` - Similar to `ui_components/visualization/analysis/correlation_analysis.py`
- `visualization/decision_boundaries.py` - Unique feature

**Action**: 
- Keep `decision_boundaries.py` (unique)
- Deprecate others if duplicated

### Utilities
- `api_client.py` - Similar to `uqlab_orchestrator/api_client.py`
- `sweep_planner.py` - Already imports from `uqlab_orchestrator.config`

**Action**: Document equivalents, deprecate if unused

## Recommended Actions

### 1. Add Deprecation Notice

Create `src/uqlab/6_ui/DEPRECATED.md`:
```markdown
# DEPRECATED: 6_ui Directory

This directory contains legacy UI implementations that are being phased out
in favor of the modular `ui_components` package.

## Migration Guide

| Old (6_ui) | New (ui_components) |
|------------|---------------------|
| experiment_builder.py | config/experiment_config.py |
| batch_builder.py | legacy/batch_config.py |
| results_viewer.py | results/results.py |
| signal_viewer.py | visualization/signals/ |
| correlation_viz.py | visualization/analysis/correlation_analysis.py |
| api_client.py | uqlab_orchestrator/api_client.py |

## Status
- ⚠️ **Deprecated**: Do not use for new development
- 📦 **Maintained**: Bug fixes only
- 🗑️ **Removal**: Planned for v3.0
```

### 2. Update 6_ui/__init__.py

Add deprecation warning:
```python
"""
DEPRECATED: Legacy UI implementations

This module is deprecated. Use `uqlab.ui_components` instead.
See DEPRECATED.md for migration guide.
"""
import warnings

warnings.warn(
    "uqlab.6_ui is deprecated. Use uqlab.ui_components instead.",
    DeprecationWarning,
    stacklevel=2
)
```

### 3. Keep sweep_planner.py Import Fix

The `sweep_planner.py` file already has the correct import:
```python
from uqlab_orchestrator.config import (
    LABEL_NOISE_SWEEP,
    TRAINING_CONFIG,
    aligned_under_train_sweep,
)
```

This is correct and should remain as-is.

## Conclusion

**DO NOT MIGRATE 6_ui to ui_components**

Instead:
1. ✅ Document deprecation status
2. ✅ Add migration guide
3. ✅ Keep existing imports working (already done for sweep_planner.py)
4. ⏳ Plan gradual phase-out
5. ⏳ Remove in future major version

The `6_ui` directory serves a different purpose (standalone apps) than `ui_components` (reusable components), and migrating would create confusion rather than clarity.