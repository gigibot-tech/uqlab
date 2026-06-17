# UI Components Migration Status

**Date**: 2026-06-07  
**Phase 1 Status**: ✅ COMPLETE  
**Overall Progress**: 20% (Phase 1 of 6 complete)

---

## ✅ Phase 1: Critical Import Fix (COMPLETE)

### Problem Identified
Files in `src/uqlab/ui_components/` were importing from `.api_sweep_launch` but the file didn't exist in that directory.

### Solution Implemented
Fixed imports to use `uqlab_orchestrator.api_client` instead:

1. ✅ **Fixed `smart_experiment_selector.py`**
   - Changed: `from .api_sweep_launch import` 
   - To: `from uqlab_orchestrator.api_client import`
   - Functions: `launch_api_sweep`, `launch_paired_both_sweeps`, `launch_paired_sweep_arm`

2. ✅ **Fixed `sweep_campaign.py`**
   - Changed: `from .api_sweep_launch import`
   - To: `from uqlab_orchestrator.api_client import`
   - Functions: `launch_api_sweep`, `launch_paired_sweep_arm`

### Verification
- ✅ No more broken relative imports in `src/uqlab/ui_components/`
- ✅ API client functions properly located in `uqlab_orchestrator`
- ✅ Imports now use correct package paths

---

## 📊 Current Architecture

### Directory Structure
```
uqlab-streamlit/
├── ui_components/                    # ROOT LEVEL (Legacy)
│   ├── __init__.py                   # Re-exports from local files
│   ├── experiment_config.py          # Actual implementation
│   ├── smart_experiment_selector.py  # Actual implementation
│   ├── sweep_campaign.py             # Actual implementation
│   └── ... (21 more files)
│
└── src/
    ├── uqlab/
    │   └── ui_components/            # PACKAGE LEVEL (Should be source of truth)
    │       ├── __init__.py
    │       ├── experiment_config.py
    │       ├── smart_experiment_selector.py  # ✅ Fixed imports
    │       ├── sweep_campaign.py             # ✅ Fixed imports
    │       └── ... (20 more files)
    │
    └── uqlab_orchestrator/
        └── api_client.py             # ✅ Contains API orchestration logic
```

### Import Patterns
- **streamlit_app.py**: `from ui_components import ...` (root level)
- **streamlit_app_progressive.py**: `from ui_components.X import ...` (root level)
- **Package files**: Now use `from uqlab_orchestrator.api_client import ...` ✅

---

## 🎯 Remaining Phases (Not Yet Started)

### Phase 2: Directory Consolidation
**Goal**: Single source of truth for UI components

**Tasks**:
1. Compare files in both directories
2. Ensure `src/uqlab/ui_components/` has latest versions
3. Convert root `ui_components/__init__.py` to re-export from `uqlab.ui_components`
4. Update Streamlit apps to import from `uqlab.ui_components`
5. Delete root-level `.py` files (keep only `__init__.py` as shim)

**Status**: ⏳ Not started

### Phase 3: Split Hybrid Files
**Goal**: Separate UI rendering from logic

**Files to Split**:
1. `experiment_config.py` - Keep `render_*`, move `build_*` to `experiment_builder.py`
2. `unified_builder.py` - Keep `render_*`, move logic to orchestrator
3. `smart_experiment_selector.py` - Keep `render_*`, move analysis to orchestrator
4. `validation_runner.py` - Keep `render_*`, move execution to orchestrator
5. `uq_benchmarks.py` - Keep `render_*`, move `fetch_*` to orchestrator

**Status**: ⏳ Not started

### Phase 4: Move Pure Logic Files
**Goal**: Proper package organization

**Files to Move**:
1. `experiment_sweep_context.py` → `uqlab_orchestrator/sweep_analyzer.py`
2. `correlation_analysis.py` → `src/uqlab/analysis/correlation.py`
3. `config_types.py` → `src/uqlab/shared/config/types.py`

**Status**: ⏳ Not started

### Phase 5: Update All Imports
**Goal**: Fix all import paths after reorganization

**Tasks**:
1. Update all Streamlit apps
2. Update all UI component files
3. Update all orchestrator files
4. Run tests to verify

**Status**: ⏳ Not started

### Phase 6: Documentation & Cleanup
**Goal**: Clear architecture documentation

**Tasks**:
1. Update `FILE_ORGANIZATION_ANALYSIS.md`
2. Create `UI_COMPONENTS_ARCHITECTURE.md`
3. Update package `README.md` files
4. Remove deprecated files

**Status**: ⏳ Not started

---

## 🚨 Key Decisions Made

### 1. API Client Location
**Decision**: Keep in `uqlab_orchestrator/api_client.py`  
**Rationale**: Pure orchestration logic, not UI rendering

### 2. Import Strategy
**Decision**: Use absolute imports (`uqlab_orchestrator.api_client`)  
**Rationale**: Clearer dependencies, easier to refactor

### 3. Backward Compatibility
**Decision**: Keep root `ui_components/__init__.py` as re-export shim  
**Rationale**: Don't break existing Streamlit apps during migration

---

## 📝 Next Steps

### Immediate (Phase 2)
1. Verify all files in `src/uqlab/ui_components/` are up-to-date
2. Check for any differences between root and package versions
3. Create migration script for Phase 2

### Short-term (Phases 3-4)
1. Identify all hybrid files that mix UI and logic
2. Create new files in appropriate packages
3. Split functionality systematically

### Long-term (Phases 5-6)
1. Update all imports across codebase
2. Run comprehensive tests
3. Document final architecture

---

## ✅ Testing Checklist

After each phase, verify:

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

## 📚 Related Documents

- `UI_COMPONENTS_COMPREHENSIVE_ANALYSIS.md` - Full file-by-file analysis
- `FILE_ORGANIZATION_ANALYSIS.md` - Original analysis of `api_sweep_launch.py`
- `ENTERPRISE_VALIDATION_COMPLETE.md` - Configuration validation implementation

---

**Last Updated**: 2026-06-07  
**Next Review**: After Phase 2 completion