# Evaluation Signals

Uncertainty and attribution primitives for fast-pilot AUROC evaluation.

## Pipeline

```
sources.py  →  primitives.py  →  registry.py  →  fast-pilot AUROC
     ↑              ↑
formulas.py    attribution.py
```

1. **`sources.py`** — runs forward passes and MC Dropout, populates a `PrimitiveStore`
2. **`primitives.py`** — named tensor slots (`MC_ENTROPY`, `FWD_MEAN_PRED`, attribution keys, …)
3. **`registry.py`** — maps signal names to formulas over primitives
4. **`formulas.py`** — human-readable formula specs (including `implementation=` trace strings)

`run_sources(ctx)` in `sources.py` is the entry point called from the evaluation pipeline.

## MC Dropout metrics

[`mc_dropout.py`](mc_dropout.py) is the single source of truth for entropy, mutual information, SIRC, and aleatoric/epistemic splits from stacked softmax predictions `[T, B, C]`.

`expected_entropy` and `mutual_info` are registered in [`registry.py`](registry.py) (`aleatoric` / `epistemic` tags). Full flow → [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md).

```python
from uqlab.evaluation.signals.mc_dropout import calculate_mc_dropout_uncertainty

metrics = calculate_mc_dropout_uncertainty(mc_predictions)
# keys: entropy, mutual_info, mean_variance, mean_prediction, variance
```

Model forwards (`mc_forward_efficient`) live in [`models/mc_dropout.py`](../../models/mc_dropout.py); only `sources.py` imports both sides.

## Attribution signals

[`attribution.py`](attribution.py) computes DualXDA-based structure signals (coherence, mass, dominance) consumed by the registry.

`inverse_coherence` is registered with `aleatoric=True` in [`registry.py`](registry.py) (default aleatoric column for the disentanglement bridge). `inverse_mass` uses `epistemic=True`.

## Related

- Parent evaluation module → [`../README.md`](../README.md)
- Model factory → [`../../models/README.md`](../../models/README.md)
- Disentanglement benchmark (different metric) → [`../../../docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md)
