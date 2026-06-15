# UI Components Comprehensive Analysis & Reorganization Plan

**Date**: 2026-06-07  
**Status**: Analysis Complete - Ready for Migration  
**Goal**: Consolidate two `ui_components` directories and properly separate UI rendering from orchestration logic

---

## 🔍 Current State: Duplicate Directory Problem

### Two `ui_components` Directories Exist:

1. **Root Level**: `walaris-cen/ui_components/` (24 files)
   - Contains `api_sweep_launch.py` ✅
   - Legacy location, used by `streamlit_app.py`
   
2. **Package Level**: `walaris-cen/src/uqlab/ui_components/` (23 files)
   - Missing `api_sweep_launch.py` ❌
   - Proper package location, used by `streamlit_app_progressive.py`
   - Has imports expecting `api_sweep_launch` to exist

### Critical Issue:
Files in `src/uqlab/ui_components/` import from `.api_sweep_launch` but the file doesn't exist there:
- `smart_experiment_selector.py` line 18
- `sweep_campaign.py` line 15

---

## 📊 File-by-File Analysis

### Category 1: API Orchestration Logic (Should Move to `uqlab_orchestrator`)

#### 1. `api_sweep_launch.py` ⚠️ **CRITICAL - MISSING FROM PACKAGE**
**Location**: Only in `walaris-cen/ui_components/`  
**Purpose**: API client for launching sweep experiments  
**Functions**:
- `normalize_dinov2_model()` - Config normalization
- `experiment_name_for_point()` - Naming convention
- `build_experiment_payload()` - API payload construction
- `create_and_start_one()` - Single experiment API call
- `launch_api_sweep()` - Batch sweep launch
- `build_default_sweep_points()` - Sweep point generation
- `launch_paired_sweep_arm()` - Paired sweep orchestration
- `launch_paired_both_sweeps()` - Full paired sweep

**Decision**: ✅ **MOVE to `src/uqlab_orchestrator/api_client.py`**
- Pure orchestration logic, no UI rendering
- Used by UI components but not part of UI
- Should be in orchestrator package

**Migration Steps**:
1. Copy to `src/uqlab_orchestrator/api_client.py`
2. Update imports in `smart_experiment_selector.py`
3. Update imports in `sweep_campaign.py`
4. Add to `src/uqlab_orchestrator/__init__.py` exports
5. Delete from root `ui_components/`

---

### Category 2: Configuration Builders (Hybrid - Need Splitting)

#### 2. `experiment_config.py`
**Purpose**: Single experiment configuration UI + config building  
**Functions**:
- `render_epistemic_config()` - ✅ UI rendering
- `render_epistemic_strength()` - ✅ UI rendering
- `render_aleatoric_config()` - ✅ UI rendering
- `render_aleatoric_strength()` - ✅ UI rendering
- `render_model_config()` - ✅ UI rendering
- `render_training_config()` - ✅ UI rendering
- `render_evaluation_config()` - ✅ UI rendering
- `render_evaluation_strategy()` - ✅ UI rendering
- `build_nested_experiment_config()` - ⚠️ **Config building logic**
- `build_base_experiment_config()` - ⚠️ **Config building logic**

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/experiment_config.py`
- Move `build_*` functions to `src/uqlab/shared/config/experiment_builder.py`

#### 3. `unified_builder.py`
**Purpose**: Unified experiment builder UI  
**Functions**:
- `render_unified_builder()` - ✅ UI rendering (main entry)
- `render_dataset_config_compact()` - ✅ UI rendering
- `render_epistemic_with_sweep()` - ✅ UI rendering
- `render_aleatoric_with_sweep()` - ✅ UI rendering
- `build_unified_config()` - ⚠️ **Config building logic**
- `detect_experiment_type()` - ⚠️ **Analysis logic**
- `render_experiment_type_summary()` - ✅ UI rendering
- `handle_unified_builder_submission()` - ⚠️ **Orchestration logic**
- `render_completed_experiments_with_visualizations()` - ✅ UI rendering
- `render_experiment_execution_panel()` - ✅ UI rendering

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/unified_builder.py`
- Move `build_unified_config()` to `src/uqlab/shared/config/experiment_builder.py`
- Move `detect_experiment_type()` to `src/uqlab_orchestrator/experiment_analyzer.py`
- Move `handle_unified_builder_submission()` to `src/uqlab_orchestrator/submission_handler.py`

---

### Category 3: Pure UI Rendering (Stay in `ui_components`)

#### 4. `dataset.py` ✅
**Purpose**: Dataset selection and comparison UI  
**Decision**: **KEEP** - Pure UI rendering

#### 5. `utils.py` ✅
**Purpose**: UI utility functions (progress, ROC explanation)  
**Decision**: **KEEP** - Pure UI helpers

#### 6. `model_selector.py` ✅
**Purpose**: Model selection and inference UI  
**Decision**: **KEEP** - Pure UI rendering

#### 7. `results.py` ✅
**Purpose**: Experiment results display UI  
**Decision**: **KEEP** - Pure UI rendering with proper separation

#### 8. `signal_visualization.py` ✅
**Purpose**: Signal visualization UI  
**Decision**: **KEEP** - Pure UI rendering

#### 9. `per_sample_signals_viz.py` ✅
**Purpose**: Per-sample signal visualization  
**Decision**: **KEEP** - Pure UI rendering

#### 10. `signal_diagnostic_viz.py` ✅
**Purpose**: Signal diagnostic visualization  
**Decision**: **KEEP** - Pure UI rendering

#### 11. `paper_sweep_viz.py` ✅
**Purpose**: Paper-style sweep visualization  
**Decision**: **KEEP** - Pure UI rendering

#### 12. `signal_sweep_paper_viz.py` ✅
**Purpose**: Signal sweep paper visualization  
**Decision**: **KEEP** - Pure UI rendering

#### 13. `heatmap_visualization.py` ✅
**Purpose**: Heatmap visualization UI  
**Decision**: **KEEP** - Pure UI rendering

#### 14. `validation_visualization.py` ✅
**Purpose**: Validation results visualization  
**Decision**: **KEEP** - Pure UI rendering

#### 15. `experiment_viz_inspector.py` ✅
**Purpose**: Experiment visualization inspector  
**Decision**: **KEEP** - Pure UI rendering

#### 16. `data_overlap_analysis.py` ✅
**Purpose**: Data overlap analysis UI  
**Decision**: **KEEP** - Pure UI rendering

---

### Category 4: Analysis & Context (Move to Orchestrator)

#### 17. `experiment_sweep_context.py`
**Purpose**: Experiment sweep context analysis  
**Functions**:
- `new_campaign_timestamp()` - Utility
- `index_campaign_batches()` - Analysis logic
- `campaign_timestamp_from_experiment()` - Parsing logic
- `fast_sweep_arm()` - Parsing logic
- `distinct_param_values()` - Analysis logic
- `count_completed()` - Analysis logic
- `ExperimentVizAnalysis` - Data class
- `analyze_experiment_for_viz()` - Analysis logic
- `format_experiment_viz_option()` - Formatting logic
- `build_points_for_missing_arm()` - Config building logic

**Decision**: ✅ **MOVE to `src/uqlab_orchestrator/sweep_analyzer.py`**
- Pure analysis and context logic
- No UI rendering
- Used by UI but not part of UI

#### 18. `correlation_analysis.py`
**Purpose**: Correlation analysis for validation  
**Classes**: `CorrelationResult`, `ValidationResult`  
**Functions**: Analysis logic only

**Decision**: ✅ **MOVE to `src/uqlab/analysis/correlation.py`**
- Pure analysis logic
- Data classes for results
- No UI rendering

---

### Category 5: Validation & Execution (Move to Orchestrator)

#### 19. `validation_runner.py`
**Purpose**: Local validation experiment execution  
**Functions**:
- `_resolve_walaris_cen_root()` - Path resolution
- `_subprocess_env()` - Environment setup
- `_stream_subprocess()` - Process execution
- `run_validation_experiments()` - Orchestration
- `_execute_sweep_ui()` - UI + execution hybrid
- `render_local_validation_viz()` - UI rendering
- `render_preset_validation_sweeps()` - UI rendering

**Decision**: ✅ **SPLIT**
- Move execution logic to `src/uqlab_orchestrator/validation_executor.py`
- Keep `render_*` functions in `ui_components/validation_runner.py`

#### 20. `experiment_validation.py`
**Purpose**: Experiment validation UI  
**Functions**: Mostly UI rendering with some validation logic

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/`
- Move `validate_sweep_configuration()` to `src/uqlab/shared/config/validation.py`

---

### Category 6: Smart Selectors (Hybrid - Need Splitting)

#### 21. `smart_experiment_selector.py`
**Purpose**: Unified experiment visualization with smart selection  
**Functions**:
- `group_experiments_for_selection()` - ⚠️ Analysis logic
- `detect_experiment_configuration()` - ⚠️ Analysis logic
- `render_experiment_type_badge()` - ✅ UI rendering
- `render_sweep_summary_table()` - ✅ UI rendering
- `render_sweep_launch_toolbar()` - ✅ UI rendering
- `render_sweep_launch_controls()` - ✅ UI rendering
- `_launch_complement()` - ⚠️ Orchestration logic
- `render_complementary_sweep_creator()` - ✅ UI rendering
- `render_sidebar_experiment_selector()` - ✅ UI rendering
- `render_unified_experiment_visualization()` - ✅ UI rendering
- `render_smart_experiment_selector()` - ✅ UI rendering (main entry)

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/`
- Move `group_experiments_for_selection()` to `src/uqlab_orchestrator/experiment_grouper.py`
- Move `detect_experiment_configuration()` to `src/uqlab_orchestrator/experiment_analyzer.py`
- Move `_launch_complement()` to `src/uqlab_orchestrator/api_client.py`

#### 22. `sweep_campaign.py`
**Purpose**: Sweep campaign panel UI  
**Functions**: Mix of UI rendering and orchestration

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/`
- Move `_launch_missing_arm()` to `src/uqlab_orchestrator/api_client.py`

---

### Category 7: Benchmarking (Move to Orchestrator)

#### 23. `uq_benchmarks.py`
**Purpose**: UQ benchmarking UI + API calls  
**Functions**: Mix of API calls and visualization

**Decision**: ✅ **SPLIT**
- Keep `render_*` and `plot_*` functions in `ui_components/`
- Move `fetch_*` functions to `src/uqlab_orchestrator/benchmark_client.py`

---

### Category 8. `hypothesis_validation.py`
**Purpose**: Hypothesis validation tab UI  
**Functions**: Mix of analysis and UI

**Decision**: ✅ **SPLIT**
- Keep `render_*` functions in `ui_components/`
- Move `analyze_validation_results()` to `src/uqlab/analysis/hypothesis.py`
- Move `load_sweep_metrics()` to `src/uqlab/data_loading/metrics_loader.py`

---

### Category 9: Type Definitions (Move to Shared)

#### 24. `config_types.py`
**Purpose**: Configuration dataclasses  
**Classes**: `ValidationConfig`, `SweepConfig`, `EpistemicConfig`, `AleatoricConfig`, `UnifiedBuilderConfig`

**Decision**: ✅ **MOVE to `src/uqlab/shared/config/types.py`**
- Pure type definitions
- No logic or UI
- Should be in shared config

---

## 🎯 Migration Plan Summary

### Phase 1: Critical Fix (Immediate)
**Goal**: Fix broken imports in package-level `ui_components`

1. ✅ Copy `api_sweep_launch.py` to `src/uqlab_orchestrator/api_client.py`
2. ✅ Update imports in `src/uqlab/ui_components/smart_experiment_selector.py`
3. ✅ Update imports in `src/uqlab/ui_components/sweep_campaign.py`
4. ✅ Test that imports work

### Phase 2: Consolidate Directories
**Goal**: Single source of truth for UI components

1. ✅ Verify all files in `src/uqlab/ui_components/` are up-to-date
2. ✅ Delete root-level `walaris-cen/ui_components/` directory
3. ✅ Update `streamlit_app.py` imports to use `uqlab.ui_components`
4. ✅ Test all Streamlit apps

### Phase 3: Split Hybrid Files
**Goal**: Separate UI rendering from logic

1. ✅ Split `experiment_config.py`:
   - Keep `render_*` in `ui_components/`
   - Move `build_*` to `src/uqlab/shared/config/experiment_builder.py`

2. ✅ Split `unified_builder.py`:
   - Keep `render_*` in `ui_components/`
   - Move `build_unified_config()` to `experiment_builder.py`
   - Move `detect_experiment_type()` to `uqlab_orchestrator/experiment_analyzer.py`
   - Move `handle_unified_builder_submission()` to `uqlab_orchestrator/submission_handler.py`

3. ✅ Split `smart_experiment_selector.py`:
   - Keep `render_*` in `ui_components/`
   - Move analysis functions to `uqlab_orchestrator/`

4. ✅ Split `validation_runner.py`:
   - Keep `render_*` in `ui_components/`
   - Move execution logic to `uqlab_orchestrator/validation_executor.py`

5. ✅ Split `uq_benchmarks.py`:
   - Keep `render_*` and `plot_*` in `ui_components/`
   - Move `fetch_*` to `uqlab_orchestrator/benchmark_client.py`

### Phase 4: Move Pure Logic Files
**Goal**: Proper package organization

1. ✅ Move `experiment_sweep_context.py` → `uqlab_orchestrator/sweep_analyzer.py`
2. ✅ Move `correlation_analysis.py` → `src/uqlab/analysis/correlation.py`
3. ✅ Move `config_types.py` → `src/uqlab/shared/config/types.py`

### Phase 5: Update All Imports
**Goal**: Fix all import paths

1. ✅ Update all Streamlit apps
2. ✅ Update all UI component files
3. ✅ Update all orchestrator files
4. ✅ Run tests to verify

### Phase 6: Documentation
**Goal**: Clear architecture documentation

1. ✅ Update `FILE_ORGANIZATION_ANALYSIS.md`
2. ✅ Create `UI_COMPONENTS_ARCHITECTURE.md`
3. ✅ Update package `README.md` files

---

## 📁 Final Directory Structure

```
walaris-cen/
├── src/
│   ├── uqlab/
│   │   ├── ui_components/          # Pure UI rendering only
│   │   │   ├── dataset.py
│   │   │   ├── experiment_config.py (render_* only)
│   │   │   ├── unified_builder.py (render_* only)
│   │   │   ├── smart_experiment_selector.py (render_* only)
│   │   │   ├── results.py
│   │   │   ├── signal_visualization.py
│   │   │   ├── heatmap_visualization.py
│   │   │   ├── validation_visualization.py
│   │   │   └── ... (all other pure UI files)
│   │   │
│   │   ├── shared/
│   │   │   └── config/
│   │   │       ├── types.py (from config_types.py)
│   │   │       ├── experiment_builder.py (build_* functions)
│   │   │       └── validation.py
│   │   │
│   │   └── analysis/
│   │       ├── correlation.py (from correlation_analysis.py)
│   │       └── hypothesis.py
│   │
│   └── uqlab_orchestrator/
│       ├── api_client.py (from api_sweep_launch.py)
│       ├── sweep_analyzer.py (from experiment_sweep_context.py)
│       ├── experiment_analyzer.py
│       ├── experiment_grouper.py
│       ├── submission_handler.py
│       ├── validation_executor.py
│       └── benchmark_client.py
│
└── streamlit_app.py (imports from uqlab.ui_components)
```

---

## ✅ Benefits of This Reorganization

1. **Single Source of Truth**: One `ui_components` directory
2. **Clear Separation**: UI rendering vs orchestration vs analysis
3. **Better Testability**: Logic separated from UI can be unit tested
4. **Reusability**: Orchestration logic can be used outside Streamlit
5. **Maintainability**: Clear boundaries between concerns
6. **No Broken Imports**: All imports properly resolved

---

## 🚨 Critical Dependencies to Preserve

### Files that import `api_sweep_launch`:
- `smart_experiment_selector.py` (line 18)
- `sweep_campaign.py` (line 15)

### Files that use config builders:
- `streamlit_app.py`
- `streamlit_app_progressive.py`
- `unified_builder.py`

### Files that use analysis logic:
- All visualization components
- Smart selectors
- Validation runners

---

## 📝 Testing Checklist

After migration, verify:

- [ ] All Streamlit apps launch without import errors
- [ ] Single experiment creation works
- [ ] Batch sweep creation works
- [ ] Complementary sweep creation works
- [ ] Experiment visualization works
- [ ] Validation sweeps execute
- [ ] All UI components render correctly
- [ ] No duplicate code between directories
- [ ] All imports use correct paths
- [ ] Package structure is clean

---

## 🎓 Lessons Learned

1. **Duplicate directories cause confusion**: Should have been caught earlier
2. **Hybrid files are problematic**: Mix of UI and logic makes refactoring hard
3. **Import paths matter**: Package-level imports are cleaner than relative
4. **Separation of concerns is critical**: UI, orchestration, and analysis should be separate

---

**Next Step**: Execute Phase 1 (Critical Fix) to resolve broken imports immediately.