"""Shim: ``uq_classification.models`` → ``uqlab.models.classification_models``."""

import importlib

_models = importlib.import_module("uqlab.models.classification_models")

EmbeddingDataset = _models.EmbeddingDataset
EmbeddingDropoutMLP = _models.EmbeddingDropoutMLP

__all__ = ["EmbeddingDataset", "EmbeddingDropoutMLP"]
