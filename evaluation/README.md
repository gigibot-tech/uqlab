# Evaluation module

**Flow:** [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)  
**Package layout:** [`docs/architecture/PACKAGE_REDESIGN.md`](../../../docs/architecture/PACKAGE_REDESIGN.md)

## Layout (2026-06)

```
evaluation/
├── signals/          # Per-sample signal computation (registry, sources, attribution backends)
├── metrics/          # Pure scoring (scoring.py) + results.pt contract (artifacts.py)
├── reporting/        # Plot payloads, campaign PDFs, CSV/markdown writers
├── benchmarks/       # Paper disentangling bridge (DisentanglingModel port)
├── four_region_validation.py
└── pipeline/         # DEPRECATED shim — re-exports new paths only
```

Runner phases (`collect_uncertainty_signals`, `score_uncertainty_signals`, recovery) live under [`runner/phases/`](../runner/phases/).  
Data setup (`prepare_experiment_data`) lives under [`data/setup.py`](../data/setup.py).

## Common imports

```python
from uqlab.evaluation.metrics.scoring import binary_auroc, auroc_skip_reason
from uqlab.evaluation.metrics.artifacts import EvalRunArtifacts
from uqlab.evaluation.signals.registry import METRICS, build_signal_table
from uqlab.evaluation.reporting.sweep_line_plot import build_sweep_line_plot
from uqlab.runner.phases.eval import score_uncertainty_signals
from uqlab.data.setup import prepare_experiment_data
from uqlab.evaluation.benchmarks import FastPilotDisentanglingModel
```

Backward-compat shims: `uqlab.evaluation.metrics`, `uqlab.evaluation.artifacts`, `uqlab.evaluation.result_writers`, `uqlab.evaluation.pipeline.*`.

## Adding a signal

1. Add a `MetricEntry` in [`signals/registry.py`](signals/registry.py).
2. Ensure required **sources** exist in [`signals/sources.py`](signals/sources.py).
3. Enable via Step 4 / `evaluation.signals` in run YAML — see [`docs/features/signal-registry.md`](../../../docs/features/signal-registry.md).

## Disentangling benchmark

[`benchmarks/disentangling/`](benchmarks/disentangling/) implements the vendor `DisentanglingModel` port. See [`benchmarks/README.md`](benchmarks/README.md) and [`docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md).
