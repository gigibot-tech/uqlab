"""Configuration bundle for fast-pilot uncertainty signal collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uqlab.shared.config.classification import ExperimentConfig
    from uqlab.evaluation.pipeline.experiment_setup import RunConfigView


@dataclass(frozen=True)
class EvalSignalConfig:
    """Runtime settings for ``collect_uncertainty_signals`` (paths + signal knobs)."""

    train_batch_size: int
    mc_passes: int
    top_k: int
    run_cache_dir: Path
    results_dir: Path
    dropout: float = 0.0
    attribution_method: str = "dualxda"
    attribution_backends: tuple[str, ...] = ("dualxda",)
    enabled_signals: frozenset[str] | None = None

    @classmethod
    def from_run_config(
        cls,
        view: RunConfigView,
        *,
        results_dir: Path,
        run_cache_dir: Path,
    ) -> EvalSignalConfig:
        from uqlab.shared.config.signals import derive_attribution_backends_from_signals

        backends = derive_attribution_backends_from_signals(view.enabled_signals)
        return cls(
            train_batch_size=view.train_batch_size,
            mc_passes=view.mc_passes,
            top_k=view.top_k,
            dropout=float(view.dropout),
            attribution_method=view.attribution_method,
            attribution_backends=backends,
            enabled_signals=frozenset(view.enabled_signals),
            run_cache_dir=run_cache_dir,
            results_dir=results_dir,
        )

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        *,
        results_dir: Path,
        run_cache_dir: Path,
    ) -> EvalSignalConfig:
        from uqlab.evaluation.pipeline.experiment_setup import extract_run_config

        return cls.from_run_config(
            extract_run_config(config),
            results_dir=results_dir,
            run_cache_dir=run_cache_dir,
        )

    def resolved_enabled_signals(self) -> set[str] | None:
        if self.enabled_signals is None:
            return None
        return set(self.enabled_signals)
