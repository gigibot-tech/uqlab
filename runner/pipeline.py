"""Single entry point: load run YAML and execute the fast-pilot experiment.

MLgym mapping
-------------
- **Config**: ``ExperimentConfig`` from nested run YAML (``run_spec.build_run_yaml``)
- **Factory**: ``build_model`` inside ``run_experiment_core``
- **Job**: this module — load → validate → execute

All callers (CLI, Flask executor, FastAPI DirectExecutor) must use :func:`run` or
:func:`run_config`; do not invoke ``run_fast_uncertainty_classification.main()`` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from uqlab.shared.config.classification import ExperimentConfig
from uqlab.models.architecture import normalize_architecture, scope_to_training_mode
from uqlab.models.training_scope import validate_training_scope
from uqlab.runner.experiment_core import run_experiment_core
from uqlab.runner.patterns import ExperimentPipeline, RunContext
from uqlab.shared.config.signals import normalize_evaluation_signals, validate_evaluation_signals
from uqlab.runtime_paths import repository_root


def _stage_load_config(ctx: RunContext) -> RunContext:
    if ctx.get("config") is not None:
        return ctx
    path = Path(ctx.get("config_path"))
    config = ExperimentConfig.from_yaml(path)
    ctx.set("config", config)
    ctx.set("config_path", path)
    return ctx


def _stage_validate_config(ctx: RunContext) -> RunContext:
    config: ExperimentConfig = ctx.get("config")
    model = config.model
    evaluation = config.evaluation

    arch = normalize_architecture(model.architecture)
    scope = getattr(model, "training_scope", None) or _legacy_scope(model)
    scope_to_training_mode(arch, scope)

    signals = normalize_evaluation_signals(getattr(evaluation, "signals", None))
    validate_evaluation_signals(
        signals=signals,
        mc_passes=int(evaluation.mc_passes),
        dropout=float(model.dropout),
    )
    ctx.set("signals", signals)
    return ctx


def _legacy_scope(model) -> str:
    if model.training_mode == "end_to_end":
        return "full"
    return "head_only"


def _stage_execute(ctx: RunContext) -> RunContext:
    config: ExperimentConfig = ctx.get("config")
    results_dir = Path(ctx.get("output_dir"))
    seed_val = ctx.get("seed")
    if seed_val is None:
        seed_val = config.seed if config.seed is not None else 42
    seed = int(seed_val)
    device_val = ctx.get("device_str")
    if device_val is None:
        device_val = config.device or "auto"
    device_str = str(device_val)
    config_path = ctx.get("config_path")
    project_root = ctx.get("project_root")
    if project_root is None:
        project_root = repository_root()

    summary = run_experiment_core(
        config,
        results_dir,
        seed=seed,
        device_str=device_str,
        config_path=config_path,
        project_root=Path(project_root),
    )
    ctx.set("summary", summary)
    return ctx


_DEFAULT_PIPELINE = ExperimentPipeline([
    _stage_load_config,
    _stage_validate_config,
    _stage_execute,
])


def _execute_pipeline(
    ctx: RunContext,
    pipeline: Optional[ExperimentPipeline] = None,
) -> RunContext:
    """Run load → validate → execute under a tee'd ``experiment.log``."""
    from uqlab.runner.experiment_log import capture_experiment_log

    results_dir = Path(ctx.get("output_dir"))
    config_path = ctx.get("config_path")
    cp = Path(config_path) if config_path is not None else None
    pipe = pipeline or _DEFAULT_PIPELINE
    with capture_experiment_log(results_dir, config_path=cp):
        return pipe.run(ctx)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    seed: Optional[int] = None,
    device_str: Optional[str] = None,
    progress_callback: Optional[Callable[..., Any]] = None,
    pipeline: Optional[ExperimentPipeline] = None,
) -> dict[str, Any]:
    """
    Execute one experiment from a run YAML file.

    Uses the Pipeline pattern (see ``runner.patterns``) for load → validate → execute.
    """
    ctx = RunContext(
        data={
            "config_path": config_path,
            "output_dir": output_dir,
            "seed": seed,
            "device_str": device_str,
            "progress_callback": progress_callback,
        }
    )
    pipe = pipeline or _DEFAULT_PIPELINE
    ctx = _execute_pipeline(ctx, pipe)
    return ctx.get("summary") or {}


def run_config(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    seed: Optional[int] = None,
    device_str: Optional[str] = None,
    config_path: Optional[Path] = None,
    progress_callback: Optional[Callable[..., Any]] = None,
    pipeline: Optional[ExperimentPipeline] = None,
) -> dict[str, Any]:
    """Execute from an in-memory ``ExperimentConfig`` (facade / tests)."""
    ctx = RunContext(
        data={
            "config": config,
            "config_path": config_path,
            "output_dir": output_dir,
            "seed": seed,
            "device_str": device_str,
            "progress_callback": progress_callback,
        }
    )
    pipe = pipeline or _DEFAULT_PIPELINE
    ctx = _execute_pipeline(ctx, pipe)
    return ctx.get("summary") or {}


def validate_model_scope_after_build(model, *, architecture: str, training_scope: str):
    """Call after ``build_model`` to assert trainable params match config."""
    from uqlab.models.training_scope import resolve_training_scope

    resolved = resolve_training_scope(
        model, architecture=architecture, training_scope=training_scope  # type: ignore[arg-type]
    )
    validate_training_scope(model, resolved)
    return resolved
