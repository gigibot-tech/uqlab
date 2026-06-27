# Vendored libraries

Third-party code **copied into the repo** under `src/uqlab/vendor/` (not pip dependencies). We vendor to pin versions, patch for PyTorch/UQLab, and keep paper benchmarks reproducible offline.

## What is a vendor port?

The vendored library defines an abstract API (e.g. [`DisentanglingModel`](disentanglement_error/disentangling_model.py): `fit` + `predict_disentangling`). UQLab implements the **read path** in evaluation:

- **Adapter:** [`evaluation/benchmarks/disentangling/experiment.py`](../evaluation/benchmarks/disentangling/experiment.py) (`ExperimentDisentanglingModel` reads `results.pt`)
- **Uncertainty data:** [`evaluation/artifacts.py`](../evaluation/artifacts.py) (`signal_table` in `results.pt`)

Training runs via `pipeline.run` + `ExperimentConfig`. Disentanglement scoring is post-hoc analysis.

Flow: [`docs/UQLAB_FLOW.md`](../../docs/UQLAB_FLOW.md)

## `disentanglement_error/`

| | |
|--|--|
| **Upstream** | [ivopascal/disentanglement_error](https://github.com/ivopascal/disentanglement_error) |
| **Purpose** | Paper metric — aleatoric vs label noise, epistemic vs dataset size |
| **Patches** | [`disentanglement_error/UPSTREAM.md`](disentanglement_error/UPSTREAM.md) |

### Public imports

```python
from uqlab.vendor.disentanglement_error import (
    DisentanglingModel,
    calculate_disentanglement_error,
    Config,
    json_results_to_df,
)
```

### Post-hoc analysis (recommended)

```bash
# After a run completes:
PYTHONPATH=src python scripts/analysis/disentanglement_error.py score \
  --results-dir data/experiments/<id>/results --mode paper
```

See [`scripts/analysis/disentanglement_error.py`](../../scripts/analysis/disentanglement_error.py).

## See also

- [`docs/UQLAB_FLOW.md`](../../docs/UQLAB_FLOW.md) — execution flow
- [`docs/features/disentanglement-benchmark.md`](../../docs/features/disentanglement-benchmark.md) — launch, analysis, tests
- [`docs/features/registries.md`](../../docs/features/registries.md) — METRICS / `signal_table` columns
