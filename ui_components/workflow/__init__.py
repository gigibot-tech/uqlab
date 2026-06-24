"""Progressive workflow step renderers (lazy exports — no torch at package import)."""

from __future__ import annotations

from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    "ensure_workflow_initialized": "session",
    "render_step1_dataset": "step1_dataset",
    "render_step2_training": "step2_training",
    "render_step3_collapsed": "step3_uncertainty",
    "render_step3_uncertainty": "step3_uncertainty",
    "render_per_class_table": "step3_per_class_table",
    "get_per_class_config_summary": "step3_per_class_table",
    "render_step4_evaluation": "step4_evaluation",
    "render_step5_review": "step5_review",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    mod_name = _LAZY_EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(f"{__name__}.{mod_name}")
    return getattr(mod, name)
