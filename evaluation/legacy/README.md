# Legacy Folder Reorganization Plan

**Date**: 2026-06-17  
**Goal**: Consolidate legacy code into `4_evaluation/legacy/` subfolder

---

## 📊 Current Structure

```
src/uqlab/
├── triage/                    # 1 file: dualxda_axioms.py
├── legacy_metrics/            # 5 files: acquisition_functions, integrity_score, etc.
├── legacy_experiments/        # 2 files: dualxda_stream, risk_coverage_report
└── 4_evaluation/
    ├── evaluator.py
    ├── metrics.py
    ├── signals.py
    ├── validators.py
    ├── benchmarks/
    └── signals/
```

---

## 🎯 Proposed Structure

```
src/uqlab/
└── 4_evaluation/
    ├── evaluator.py
    ├── metrics.py
    ├── signals.py
    ├── validators.py
    ├── benchmarks/
    ├── signals/
    └── legacy/                 # NEW: Consolidated legacy code
        ├── __init__.py
        ├── README.md           # Documentation
        ├── triage/             # From: /triage
        │   ├── __init__.py
        │   └── dualxda_axioms.py
        ├── metrics/            # From: /legacy_metrics
        │   ├── __init__.py
        │   ├── acquisition_functions.py
        │   ├── integrity_score.py
        │   ├── standard_uq_metrics.py
        │   ├── surgical_score.py
        │   └── uncertainty_suite.py
        └── experiments/        # From: /legacy_experiments
            ├── __init__.py
            ├── dualxda_stream.py
            └── risk_coverage_report.py
```

---

## 🔗 Import Dependencies

### Current Imports (5 locations):

1. **`4_evaluation/signals/attribution.py`** (2 imports):
   ```python
   from uqlab.triage.dualxda_axioms import DualXDATracer
   from uqlab.triage.dualxda_axioms import infer_classifier_layer_name
   ```

2. **`legacy_metrics/acquisition_functions.py`** (3 imports):
   ```python
   from .surgical_score import SurgicalScoreCalculator
   from uqlab.legacy_experiments.dualxda_stream import compute_dualxda_scores_streaming
   from uqlab.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name
   ```

3. **`legacy_metrics/uncertainty_suite.py`** (3 imports):
   ```python
   from uqlab.legacy_metrics.surgical_score import batch_surgical_score
   from uqlab.legacy_experiments.dualxda_stream import compute_dualxda_scores_streaming
   from uqlab.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name
   ```

4. **`legacy_experiments/risk_coverage_report.py`** (1 import):
   ```python
   from uqlab.legacy_metrics.standard_uq_metrics import StandardUQMetrics
   ```

### New Imports (after reorganization):

1. **`4_evaluation/signals/attribution.py`**:
   ```python
   from uqlab.4_evaluation.legacy.triage.dualxda_axioms import DualXDATracer
   from uqlab.4_evaluation.legacy.triage.dualxda_axioms import infer_classifier_layer_name
   ```

2. **`4_evaluation/legacy/metrics/acquisition_functions.py`**:
   ```python
   from .surgical_score import SurgicalScoreCalculator
   from uqlab.4_evaluation.legacy.experiments.dualxda_stream import compute_dualxda_scores_streaming
   from uqlab.4_evaluation.legacy.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name
   ```

3. **`4_evaluation/legacy/metrics/uncertainty_suite.py`**:
   ```python
   from uqlab.4_evaluation.legacy.metrics.surgical_score import batch_surgical_score
   from uqlab.4_evaluation.legacy.experiments.dualxda_stream import compute_dualxda_scores_streaming
   from uqlab.4_evaluation.legacy.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name
   ```

4. **`4_evaluation/legacy/experiments/risk_coverage_report.py`**:
   ```python
   from uqlab.4_evaluation.legacy.metrics.standard_uq_metrics import StandardUQMetrics
   ```

---

## 📝 Migration Steps

### Step 1: Create New Structure
```bash
mkdir -p src/uqlab/4_evaluation/legacy/{triage,metrics,experiments}
```

### Step 2: Move Files
```bash
# Move triage
mv src/uqlab/triage/dualxda_axioms.py src/uqlab/4_evaluation/legacy/triage/

# Move legacy_metrics
mv src/uqlab/legacy_metrics/*.py src/uqlab/4_evaluation/legacy/metrics/

# Move legacy_experiments
mv src/uqlab/legacy_experiments/*.py src/uqlab/4_evaluation/legacy/experiments/
```

### Step 3: Create __init__.py Files
```bash
touch src/uqlab/4_evaluation/legacy/__init__.py
touch src/uqlab/4_evaluation/legacy/triage/__init__.py
touch src/uqlab/4_evaluation/legacy/metrics/__init__.py
touch src/uqlab/4_evaluation/legacy/experiments/__init__.py
```

### Step 4: Update Imports (5 files)
1. `4_evaluation/signals/attribution.py` (2 imports)
2. `4_evaluation/legacy/metrics/acquisition_functions.py` (3 imports)
3. `4_evaluation/legacy/metrics/uncertainty_suite.py` (3 imports)
4. `4_evaluation/legacy/experiments/risk_coverage_report.py` (1 import)

### Step 5: Remove Old Directories
```bash
rmdir src/uqlab/triage
rmdir src/uqlab/legacy_metrics
rmdir src/uqlab/legacy_experiments
```

### Step 6: Create Documentation
Create `src/uqlab/4_evaluation/legacy/README.md` explaining:
- What legacy code is
- Why it's separated
- Migration path for users
- Deprecation timeline (if any)

---

## ✅ Benefits

1. **Better Organization**: All legacy code in one place under evaluation
2. **Clear Separation**: Legacy vs current code is obvious
3. **Easier Maintenance**: Can deprecate/remove entire legacy folder later
4. **Logical Grouping**: Legacy evaluation code lives with evaluation code
5. **Cleaner Root**: Removes 3 top-level folders

---

## ⚠️ Considerations

1. **Import Changes**: 5 files need import updates (manageable)
2. **External Users**: If anyone imports these modules, their code will break
   - Solution: Add deprecation warnings in old locations (optional)
3. **Testing**: Need to verify all imports work after migration
4. **Documentation**: Update any docs that reference old paths

---

## 🔄 Alternative: Backward Compatibility

If external users exist, we can keep old locations as re-exports:

**`src/uqlab/triage/__init__.py`**:
```python
"""
DEPRECATED: This module has moved to uqlab.4_evaluation.legacy.triage
Import from the new location instead.
"""
import warnings
warnings.warn(
    "uqlab.triage is deprecated. Use uqlab.4_evaluation.legacy.triage instead.",
    DeprecationWarning,
    stacklevel=2
)

from uqlab.4_evaluation.legacy.triage.dualxda_axioms import *
```

Similar for `legacy_metrics` and `legacy_experiments`.

---

## 🚀 Execution Plan

**Option A: Clean Break** (Recommended if no external users)
- Move files
- Update imports
- Delete old folders
- Commit and push

**Option B: Gradual Migration** (If external users exist)
- Move files
- Update imports
- Keep old folders with deprecation warnings
- Remove old folders in future release

---

## 📊 Impact Summary

**Files to Move**: 8 files total
- 1 from `triage/`
- 5 from `legacy_metrics/`
- 2 from `legacy_experiments/`

**Files to Update**: 4 files (5 import locations)
- `4_evaluation/signals/attribution.py`
- `legacy/metrics/acquisition_functions.py`
- `legacy/metrics/uncertainty_suite.py`
- `legacy/experiments/risk_coverage_report.py`

**Folders to Remove**: 3 folders
- `triage/`
- `legacy_metrics/`
- `legacy_experiments/`

**New Folders**: 4 folders
- `4_evaluation/legacy/`
- `4_evaluation/legacy/triage/`
- `4_evaluation/legacy/metrics/`
- `4_evaluation/legacy/experiments/`

---

## ✨ Next Steps

1. Review this plan
2. Choose Option A (clean) or Option B (gradual)
3. Execute migration
4. Test imports
5. Update documentation
6. Commit and push

**Estimated Time**: 15-20 minutes for clean migration