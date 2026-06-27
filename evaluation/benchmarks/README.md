# Evaluation benchmarks

Paper and campaign benchmarks that **read** fast-pilot run artifacts — they do not replace the runner.

**Flow:** [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md) — why `fit` vs `predict_disentangling`, call chain below.

## Where `fit` and `predict_disentangling` happen

Unlike the upstream Keras demo, **`fit` trains nothing in-process** and **`predict_disentangling` does not run MC** — it reads the last job’s artifacts.

```text
calculate_disentanglement_error (vendor)
  └ model.fit(x, y, label_noise=…)     disentangling/fast_pilot.py
       └ build_run_yaml → run_from_yaml  runner/execute.py
            └ run_experiment_core       uqlab/runner/fast_pilot_core.py
                 train_*_model + collect_uncertainty_signals → results.pt
  └ model.predict_disentangling(x)     disentangling/fast_pilot.py
       └ evaluation/artifacts.py          EvalRunArtifacts.disentangling_vectors
```

| Step | File |
|------|------|
| Vendor loops call `fit` / `predict` | [`vendor/disentanglement_error/label_noise.py`](../../vendor/disentanglement_error/label_noise.py), `decreasing_dataset.py` |
| Bridge adapter | [`disentangling/fast_pilot.py`](disentangling/fast_pilot.py) |
| Train + eval + write `results.pt` | [`runner/execute.py`](../../runner/execute.py) → `run_experiment_core` |
| `results.pt` → numpy vectors | [`artifacts.py`](../artifacts.py) (`EvalRunArtifacts`) |

## Disentangling bridge (`disentangling/`)

| File | Role |
|------|------|
| [`fast_pilot.py`](disentangling/fast_pilot.py) | `FastPilotDisentanglingModel` / `UQLabDisentanglingBridge` — vendor port |
| [`uncertainty_pairs.py`](disentangling/uncertainty_pairs.py) | `results.pt` → `(pred, aleatoric, epistemic)` vectors |
| [`disentanglement_launcher.py`](disentangling/disentanglement_launcher.py) | API grid launcher |

### Upstream Keras vs UQLab

```python
# Upstream (in-process CNN)
classifications = stochastic_preds.mean(axis=0).argmax(axis=1)
aleatorics = expected_entropy(stochastic_preds)
epistemics = mutual_information(stochastic_preds)

# UQLab Paper mode (default — `predict_mode="paper"`)
aleatorics = results.pt["signal_table"]["expected_entropy"]
epistemics = results.pt["signal_table"]["mutual_info"]
pred = results.pt["predictions"]

# UQLab Signal mode (`predict_mode="signal"` → DualXDA)
aleatorics = results.pt["signal_table"]["inverse_coherence_dualxda"]
epistemics = results.pt["signal_table"]["inverse_mass_dualxda"]

# UQLab EK-FAC signal mode (`predict_mode="signal_ek_fak"`)
aleatorics = results.pt["signal_table"]["inverse_coherence_ek_fak"]
epistemics = results.pt["signal_table"]["inverse_mass_ek_fak"]
```

Each `fit()` ignores raw `X, y` and runs [`run_from_yaml`](../../runner/execute.py) for one sweep point (label noise or dataset size). `predict_disentangling()` loads `{run_dir}/results.pt` (read-only; MC or the matching attribution backend must already be in `signal_table` — see bridge table in [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)).

**Vendor:** [`vendor/disentanglement_error/`](../../vendor/disentanglement_error/) — copied metric loops; bridge implements [`DisentanglingModel`](../../vendor/disentanglement_error/disentangling_model.py).

Feature doc: [`docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md)
