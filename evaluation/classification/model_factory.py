"""Shim: ``uq_classification.model_factory`` → ``uqlab.models.factory``."""

import importlib

_factory = importlib.import_module("uqlab.models.factory")

build_model = _factory.build_model

__all__ = ["build_model"]
