# Experiment runner (MLgym job entry)

Single execution path for every surface (CLI, Flask, FastAPI backend, facade).

**Flow:** [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)

```
run YAML  →  load ExperimentConfig  →  validate  →  fast_pilot_core.run_experiment_core
                      ↑                                      (uqlab/runner/)
              experiment.log (tee, whole run)
                      ↓
         summary.json · per_sample_signals.csv · results.pt
```

## Modules

| Module | Role |
|--------|------|
| [`pipeline.py`](pipeline.py) | `run` / `run_config` — load YAML, validate, tee log |
| [`fast_pilot_core.py`](fast_pilot_core.py) | `run_experiment_core` — train, signals, write artifacts |

## Entry points

| Caller | Function |
|--------|----------|
| `scripts/run_fast_uncertainty_classification.py` | CLI → `uqlab.runner.pipeline.run` |
| `uqlab-flask/executor.py` | `uqlab.runner.pipeline.run` |
| `backend/.../direct_executor.py` | Injected/default `uqlab.runner.pipeline.run` |
| Facade / in-memory config | `uqlab.runner.pipeline.run_config` |

## Config source

Nested YAML is built only via `uqlab_orchestrator.run_spec.build_run_yaml(workflow)`.
UI layers (Streamlit wizard, Flask wizard) edit a `workflow` dict — never flat training dicts on the hot path.

## Patterns

See `patterns.py`: **Pipeline** (load → validate → execute), **Factory** (`build_model`), **Strategy** (signal families).
