# Experiment runner (MLgym job entry)

Single execution path for every surface (CLI, Flask, FastAPI backend, facade).

**Flow:** [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)

```
run YAML  →  load ExperimentConfig  →  validate  →  experiment_core.run_experiment_core
                      ↑                                      (uqlab/runner/)
              experiment.log (tee, whole run)
                      ↓
         summary.json · per_sample_signals.csv · results.pt
```

## Modules

| Module | Role |
|--------|------|
| [`execute.py`](execute.py) | `run_from_yaml` / `run_from_python_config` — load YAML, validate, tee log |
| [`experiment_core.py`](experiment_core.py) | `run_experiment_core` — thin orchestrator; domain logic in `data/`, `models/`, `run_artifacts/` |
| [`console_log.py`](console_log.py) | Stdout-only AUROC tables and artifact listing |

Paper API mapping: [`docs/features/PAPER_FLOW.md`](../../../docs/features/PAPER_FLOW.md)

## Entry points

| Caller | Function |
|--------|----------|
| `scripts/runners/run_fast_uncertainty_classification.py` | CLI → `uqlab.runner.execute.run_from_yaml` (default: `four_region.yaml`) |
| `scripts/runners/run_validation_experiments.py` | Sweep orchestrator → temp YAML → `run_from_yaml` |
| `uqlab-flask/executor.py` | `uqlab.runner.execute.run_from_yaml` |
| `backend/.../direct_executor.py` | Injected/default `uqlab.runner.execute.run_from_yaml` |
| Facade / in-memory config | `uqlab.runner.execute.run_from_python_config` |

Deprecated aliases: `run` (= `run_from_yaml`), `run_config` (= `run_from_python_config`).

Post-hoc analysis (no training) lives under `scripts/analysis/` — see [`docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md).

## Config source

Nested YAML is built only via `uqlab_orchestrator.run_spec.build_run_yaml(workflow)`.
UI layers (Streamlit wizard, Flask wizard) edit a `workflow` dict — never flat training dicts on the hot path.

Example region-based config: `configs/experiment/four_region.yaml` (`partition_mode: four_region` + `class_regions`).

## Patterns

See `patterns.py`: **Pipeline** (load → validate → execute), **Factory** (`build_model`), **Strategy** (signal families).
