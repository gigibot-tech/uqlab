"""Shim: ``uq_classification.watsonx_streamlit`` → ``5_api/integrations/watsonx`` (no sqlmodel deps)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_watsonx_path = Path(__file__).resolve().parents[1] / "5_api" / "integrations" / "watsonx.py"
_spec = importlib.util.spec_from_file_location("uqlab_watsonx_streamlit", _watsonx_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load watsonx UI module from {_watsonx_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

render_cloud_mode_toggle = _module.render_cloud_mode_toggle

__all__ = ["render_cloud_mode_toggle"]
