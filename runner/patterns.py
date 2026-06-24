"""
Design patterns used by the experiment runner (intentionally small surface).

Factory (models)
    ``uqlab.models.factory.build_model`` — picks ResNet18 / SmallCNN / EmbeddingMLP
    from ``ModelConfig.architecture`` (canonical or legacy alias).

Pipeline (runner)
    ``ExperimentPipeline`` — ordered stages; each stage mutates shared ``RunContext``.

Strategy (signals)
    ``SIGNAL_FAMILIES`` in ``shared.config.signals`` — predictive / logit / attribution
    groups; eval picks which families to compute (MC only when predictive family needs it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, TypeVar

T = TypeVar("T")


@dataclass
class RunContext:
    """Mutable bag passed through pipeline stages."""

    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value


Stage = Callable[[RunContext], RunContext]


@dataclass
class ExperimentPipeline:
    """
    Pipeline pattern: run config through ordered stages.

    Example::

        pipe = ExperimentPipeline([
            load_config,
            validate_config,
            execute_experiment,
        ])
        pipe.run(RunContext())
    """

    stages: List[Stage]

    def run(self, ctx: RunContext) -> RunContext:
        for stage in self.stages:
            ctx = stage(ctx)
        return ctx
