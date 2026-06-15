"""Shim: ``uq_classification.model_factory`` → ``uqlab.2_models.factory``."""

import importlib

_factory = importlib.import_module("uqlab.2_models.factory")

build_model = _factory.build_model

__all__ = ["build_model"]
