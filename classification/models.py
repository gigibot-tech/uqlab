"""Shim: ``uq_classification.models`` → ``uqlab.2_models.classification_models``."""

import importlib

_models = importlib.import_module("uqlab.2_models.classification_models")

EmbeddingDataset = _models.EmbeddingDataset
EmbeddingDropoutMLP = _models.EmbeddingDropoutMLP

__all__ = ["EmbeddingDataset", "EmbeddingDropoutMLP"]
