# Evaluation module

**Flow:** [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)

**Artifact contract:** [`artifacts.py`](../artifacts.py) (`EvalRunArtifacts`, `signal_table` reader)

## Layout

```
evaluation/
├── README.md                 # ← you are here
├── evaluator.py              # AUROC, CSV export, 3-way classifier helpers
├── pipeline/                 # Campaign plots, paper score aggregation
├── signals/                  # METRICS registry, sources, MC + attribution primitives
│   ├── registry.py           # Single source of truth for signal names + tags
│   ├── mc_dropout.py         # MC entropy / mutual_info / expected_entropy math
│   └── README.md
└── benchmarks/               # Paper metric bridge (disentanglement_error)
    └── README.md
```

Archived: `dead_code/evaluation/` (legacy `signals.py`, `validators.py`, old benchmarks tree).

## Adding a signal

1. Add a `MetricEntry` in [`signals/registry.py`](signals/registry.py).
2. Ensure required **sources** exist in [`signals/sources.py`](signals/sources.py).
3. Enable via Step 4 / `evaluation.signals` in run YAML — see [`docs/features/signal-registry.md`](../../../docs/features/signal-registry.md).

## Disentangling benchmark

[`benchmarks/disentangling/`](benchmarks/disentangling/) implements the vendor `DisentanglingModel` port. Default pairing is **Paper mode** (`expected_entropy` + `mutual_info`); override with `predict_mode="signal"` or explicit `aleatoric_signal` / `epistemic_signal`. See [`benchmarks/README.md`](benchmarks/README.md) and [`docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md).

## Common imports

```python
from uqlab.evaluation.metrics import binary_auroc, auroc_skip_reason
from uqlab.evaluation.signals.registry import METRICS, build_signal_table
from uqlab.evaluation.benchmarks import FastPilotDisentanglingModel
```
