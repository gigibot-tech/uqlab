# Vendored libraries

Third-party code **copied into the repo** under `src/uqlab/vendor/` (not pip dependencies). We vendor to pin versions, patch for PyTorch/UQLab, and keep paper benchmarks reproducible offline.

## What is a vendor port?

The vendored library defines an abstract API (e.g. [`DisentanglingModel`](disentanglement_error/disentangling_model.py): `fit` + `predict_disentangling`). UQLab implements that API in **evaluation**, not here:

- **Adapter:** [`evaluation/benchmarks/disentangling/fast_pilot.py`](../evaluation/benchmarks/disentangling/fast_pilot.py)
- **Uncertainty data:** [`evaluation/artifacts.py`](../evaluation/artifacts.py) (`signal_table` in `results.pt`)

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

### CLI example

```python
from uqlab.evaluation.benchmarks.disentangling import (
    FastPilotDisentanglingModel,
    calculate_disentanglement_error,
    collect_cifar10_arrays,
)

X, y = collect_cifar10_arrays()
score, _, _ = calculate_disentanglement_error(
    X, y, FastPilotDisentanglingModel.from_workflow_defaults(), kw_config={"n_runs": 1}
)
```

See [`scripts/runners/run_disentanglement_benchmark.py`](../../scripts/runners/run_disentanglement_benchmark.py).

## See also

- [`docs/UQLAB_FLOW.md`](../../docs/UQLAB_FLOW.md) — why `fit` vs `predict_disentangling`
- [`docs/features/disentanglement-benchmark.md`](../../docs/features/disentanglement-benchmark.md) — Streamlit launch, tests
- [`docs/features/registries.md`](../../docs/features/registries.md) — METRICS / `signal_table` columns
