# Experiment Grouping Module

## Overview

This module provides intelligent grouping of experiments into parameter sweeps by analyzing experiment metadata, names, and configurations.

## Architecture

```
grouping/
├── __init__.py           # Public API exports
├── sweep_grouping.py     # Core grouping logic + rendering
└── README.md            # This file
```

## Core Concepts

### Sweep Group

A **sweep group** is a collection of experiments that differ by exactly one parameter. For example:

```python
{
    'name': 'Sweep: mc_passes',
    'swept_param': 'evaluation.mc_passes',
    'experiments': [exp1, exp2, exp3, ...],
    'values': [10, 20, 30, ...],
    'created_at': datetime,
    'sweep_group_id': 'optional_id'
}
```

### Detection Strategies

The module uses **3 strategies** in order of reliability:

1. **Metadata-based** (Option 1): Explicit `sweep_group_id` in database
2. **Name pattern-based**: Detects `prefix_timestamp_param_value` patterns
3. **Config-based** (Option 2): Analyzes YAML config differences

## Public API

### Main Entry Point

```python
from uqlab.ui_components.grouping import group_experiments_intelligently

sweep_groups, standalone = group_experiments_intelligently(
    experiments,
    min_group_size=3
)
```

**Parameters:**
- `experiments`: List of experiment dicts from API
- `min_group_size`: Minimum experiments to form a sweep (default: 3)

**Returns:**
- `sweep_groups`: List of sweep group dicts (sorted by creation time, newest first)
- `standalone`: List of experiments not in any sweep

### Rendering Function

```python
from uqlab.ui_components.grouping import render_sweep_group_summary

render_sweep_group_summary(
    group,
    show_details=False,
    api_base_url=None
)
```

**Parameters:**
- `group`: Sweep group dict from `group_experiments_intelligently()`
- `show_details`: Whether to show detailed experiment list (default: False)
- `api_base_url`: Base URL for API (optional, for fetching detailed metrics)

**Renders:**
- Summary metrics (total runs, completed, best AUROC scores)
- Swept parameter and values
- Status breakdown
- Optional: Detailed experiment table with "View Details" buttons

## Integration with Results Module

The grouping module is designed to be used by the `results/` module:

```python
# In results/experiment_results_panel.py
from uqlab.ui_components.grouping import (
    group_experiments_intelligently,
    render_sweep_group_summary,
)

# Fetch experiments from API
experiments = fetch_experiments()

# Group into sweeps
sweep_groups, standalone = group_experiments_intelligently(experiments)

# Render sweep groups
for group in sweep_groups:
    with st.expander(f"🔬 {group['name']}"):
        render_sweep_group_summary(group, show_details=True)
```

## UI Debug Integration

The grouping module respects the `results_sweep_groups` toggle in `ui_debug.py`:

```python
# In ui_debug.py
"results_sweep_groups": ("Results · sweep groups (expanders + summary cards)", True)
```

When this toggle is **ON**:
- Sweep group expanders are rendered
- Summary cards are shown inside expanders
- Experiment details are available (if `results_experiment_details` is also ON)

When this toggle is **OFF**:
- No sweep groups are displayed
- Only standalone experiments table is shown (if enabled)

## Strategy Details

### 1. Metadata-based Grouping

**When to use:** When experiments have explicit sweep metadata in the database

**Required fields:**
- `sweep_group_id`: Unique identifier for the sweep
- `swept_parameter`: Name of the parameter being swept
- `swept_value`: Value of the parameter for this experiment
- `sweep_index`: Position in the sweep (for sorting)

**Example:**
```python
experiment = {
    'id': '123',
    'name': 'exp_1',
    'sweep_group_id': 'sweep_abc',
    'swept_parameter': 'mc_passes',
    'swept_value': 20,
    'sweep_index': 1,
    ...
}
```

### 2. Name Pattern-based Grouping

**When to use:** When experiments follow a naming convention

**Pattern:** `{prefix}_{date}_{time}_{param}_{value}`

**Example:**
```
fast_alea_20260615_174459_noise_100
fast_alea_20260615_174459_noise_75
fast_alea_20260615_174459_noise_50
```

**Detection:**
- Groups experiments with same `prefix_date_time`
- Extracts parameter name from second-to-last part
- Extracts value from last part
- Sorts by value (numeric if possible, otherwise string)

### 3. Config-based Grouping

**When to use:** When experiments have similar configs but differ by one parameter

**Algorithm:**
1. Parse all experiment YAML configs
2. Flatten nested dicts (e.g., `model.hidden_dim` → `model.hidden_dim`)
3. Compare each pair of configs
4. Group experiments that differ by exactly ONE parameter
5. Sort by swept parameter value

**Example:**
```yaml
# Experiment 1
evaluation:
  mc_passes: 10

# Experiment 2
evaluation:
  mc_passes: 20

# Difference: evaluation.mc_passes (10 vs 20)
```

## Modularity & Separation of Concerns

### Pure Logic (grouping/)

- ✅ Grouping algorithms (3 strategies)
- ✅ Data structure definitions
- ✅ Sorting and filtering logic

### UI Rendering (grouping/ + results/)

- ⚠️ `render_sweep_group_summary()` currently in `grouping/sweep_grouping.py`
- ⚠️ Creates circular dependency with `results/experiment_details.py`

### Recommended Refactoring

**Option A:** Move `render_sweep_group_summary()` to `results/sweep_results.py`
- Keeps pure grouping logic in `grouping/`
- All rendering in `results/`
- Eliminates circular dependencies

**Option B:** Keep current structure but document the interface
- Accept that rendering is part of the grouping module's public API
- Document the dependency on `results/experiment_details`
- Ensure `results/` module doesn't import from `grouping/` except for public API

## Testing

To test the grouping logic:

```python
# Create test experiments
experiments = [
    {'id': '1', 'name': 'exp_1', 'config_yaml': '...', ...},
    {'id': '2', 'name': 'exp_2', 'config_yaml': '...', ...},
    ...
]

# Test grouping
sweep_groups, standalone = group_experiments_intelligently(experiments)

# Verify results
assert len(sweep_groups) == expected_groups
assert len(standalone) == expected_standalone
assert sweep_groups[0]['swept_param'] == 'expected_param'
```

## Future Enhancements

1. **Database migration** for Option 1 (explicit sweep metadata)
2. **Sweep comparison** across multiple groups
3. **Sweep visualization** (line charts showing parameter vs AUROC)
4. **Sweep export** to CSV/JSON for analysis
5. **Sweep templates** for common parameter sweeps

## Related Files

- `results/experiment_results_panel.py` - Main consumer of this module
- `results/experiment_details.py` - Detailed experiment metrics (imported by `render_sweep_group_summary`)
- `ui_debug.py` - UI toggle configuration
- `backend/app/models.py` - Database schema (for Option 1 metadata)

---

**Made with Bob** 🤖