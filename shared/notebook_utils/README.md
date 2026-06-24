# notebook_support

Modular helpers for `notebooks/validation/`. **Do not paste plotting functions into notebook cells** — import from here.

## Setup cell (copy into notebook)

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from notebook_support import (
    ARCHITECTURE_STYLES,
    ensure_columns,
    find_project_root,
    get_row3_signals,
    load_sweep_metrics,
    plot_individual_signals,
    plot_method_uncertainty_comparison,
    run_validation_experiments,
    summarize_trends,
    validation_dir,
)

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

project_root = find_project_root()
```

Run notebooks with cwd = `notebooks/validation/` (or any cwd — `find_project_root()` locates the repo).

## Modules

| Module | Purpose |
|--------|---------|
| `constants.py` | Architecture styles, sweep x-columns, signal display names |
| `signals.py` | `get_row3_signals()`, AUROC helpers |
| `data_utils.py` | Load metrics, run validation script, trend summaries |
| `plotting.py` | Per-signal AUROC grid, dual-axis single architecture |
| `method_comparison.py` | **Matplotlib** 3×4 paper-style figure (notebooks) |
| `method_comparison_plotly.py` | **Plotly** same layout (Streamlit only) |

## Row 3 signals

Fixed: `inverse_coherence`, `inverse_dominance`, `inverse_mass`.  
Fourth column: best mean AUROC among `msp_uncertainty`, `predictive_entropy`, `mutual_info`, `inverse_logit_magnitude`.

## Streamlit

`ui_components/hypothesis_validation.py` imports `notebook_support.method_comparison_plotly` (Plotly) and re-exports the same `get_row3_signals` logic.

## Regenerate notebooks

```bash
cd notebooks/validation
python repair_validation_notebooks.py
```
