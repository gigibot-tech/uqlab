# 6_ui to ui_components/legacy Migration Plan

## Proposal

Move `src/uqlab/6_ui/` → `src/uqlab/ui_components/legacy/6_ui/` for better organization.

## Rationale

1. **Clearer Organization**: Groups all legacy UI code under `ui_components/legacy/`
2. **Consistent Structure**: Aligns with existing `ui_components/legacy/batch_config.py`
3. **Easier Deprecation**: Clear path for eventual removal
4. **Better Discovery**: Developers can find all legacy code in one place

## Current Usage Analysis

### Files in 6_ui
```
src/uqlab/6_ui/
├── __init__.py
├── api_client.py              # Replaced by: uqlab_orchestrator/api_client.py
├── app.py                     # Standalone app (no replacement)
├── batch_builder.py           # Replaced by: ui_components/legacy/batch_config.py
├── correlation_viz.py         # Replaced by: ui_components/visualization/analysis/correlation_analysis.py
├── experiment_builder.py      # Replaced by: ui_components/config/experiment_config.py
├── results_viewer.py          # Replaced by: ui_components/results/results.py
├── signal_viewer.py           # Replaced by: ui_components/visualization/signals/
├── sweep_planner.py           # Uses: uqlab_orchestrator.config (already updated)
├── visualizations.py          # Generic visualizations
├── apps/
│   ├── __init__.py
│   └── classification_viz.py # Standalone app
└── visualization/
    ├── __init__.py
    └── decision_boundaries.py # Unique feature (no replacement)
```

### Import Search Results

**No active imports found** - searched for:
- `from uqlab.6_ui`
- `import.*6_ui`

**Conclusion**: The `6_ui` directory is **not actively used** by any current code.

## Migration Steps

### Step 1: Move Directory
```bash
mkdir -p src/uqlab/ui_components/legacy/6_ui
mv src/uqlab/6_ui/* src/uqlab/ui_components/legacy/6_ui/
rmdir src/uqlab/6_ui
```

### Step 2: Update Internal Imports

Files that may need import updates within `6_ui`:
- `app.py` - Check for relative imports
- `apps/classification_viz.py` - Check for relative imports
- Any files importing from other `6_ui` modules

### Step 3: Add Deprecation Notice

Create `src/uqlab/ui_components/legacy/6_ui/README.md`:
```markdown
# Legacy 6_ui Directory

⚠️ **DEPRECATED** - This directory contains legacy standalone Streamlit apps.

## Status
- **Moved from**: `src/uqlab/6_ui/`
- **Moved on**: 2026-06-08
- **Reason**: Consolidation of legacy UI code
- **Replacement**: Use `uqlab.ui_components` for new development

## Migration Guide

| Legacy File | Modern Replacement |
|-------------|-------------------|
| `api_client.py` | `uqlab_orchestrator.api_client` |
| `experiment_builder.py` | `uqlab.ui_components.config.experiment_config` |
| `batch_builder.py` | `uqlab.ui_components.legacy.batch_config` |
| `results_viewer.py` | `uqlab.ui_components.results.results` |
| `signal_viewer.py` | `uqlab.ui_components.visualization.signals` |
| `correlation_viz.py` | `uqlab.ui_components.visualization.analysis.correlation_analysis` |

## Unique Features (No Replacement)
- `app.py` - Legacy standalone app
- `apps/classification_viz.py` - Legacy classification app
- `visualization/decision_boundaries.py` - Decision boundary visualization

## Removal Plan
- **Phase 1** (Current): Moved to legacy, documented
- **Phase 2** (v2.1): Add deprecation warnings
- **Phase 3** (v3.0): Remove entirely
```

### Step 4: Update ui_components/legacy/__init__.py

Add reference to 6_ui:
```python
"""
Legacy UI Components

This package contains deprecated UI implementations that are being phased out.

Subpackages:
- batch_config: Legacy batch experiment configuration
- 6_ui/: Legacy standalone Streamlit apps (moved from src/uqlab/6_ui)
"""
```

## Benefits of Migration

1. ✅ **Clearer Organization**: All legacy code in one place
2. ✅ **Easier Maintenance**: Clear deprecation path
3. ✅ **Better Documentation**: Centralized legacy code documentation
4. ✅ **Consistent Structure**: Aligns with project architecture
5. ✅ **No Breaking Changes**: No active imports to update

## Risks

1. ⚠️ **Potential Hidden Usage**: Some scripts might use `6_ui` directly
2. ⚠️ **Documentation Updates**: Need to update any docs referencing `6_ui`
3. ⚠️ **Path Changes**: Any hardcoded paths will break

## Recommendation

✅ **PROCEED WITH MIGRATION**

Since no active imports were found, the migration is safe and will improve code organization.

## Implementation Checklist

- [ ] Create backup of `src/uqlab/6_ui/`
- [ ] Move directory to `src/uqlab/ui_components/legacy/6_ui/`
- [ ] Update internal imports within moved files
- [ ] Add README.md with deprecation notice
- [ ] Update `ui_components/legacy/__init__.py`
- [ ] Test that no imports are broken
- [ ] Update project documentation
- [ ] Commit changes with clear message

## Alternative: Keep Separate

If migration is deemed too risky, alternative is to:
1. Keep `6_ui` where it is
2. Add deprecation notice in place
3. Document relationship with `ui_components/legacy`
4. Plan removal for v3.0

However, given **no active usage**, migration is the cleaner approach.