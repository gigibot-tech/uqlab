"""Shim: ``uq_classification.feature_extractor`` → ``uqlab.models.feature_extractors``."""

import importlib

_fe = importlib.import_module("uqlab.models.feature_extractors")

create_feature_extractor = _fe.create_feature_extractor
DINOv2FeatureExtractor = _fe.DINOv2FeatureExtractor
FeatureExtractor = _fe.FeatureExtractor

__all__ = [
    "FeatureExtractor",
    "DINOv2FeatureExtractor",
    "create_feature_extractor",
]
