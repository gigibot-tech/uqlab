"""Shim: ``uq_classification.utils`` → ``uqlab.shared.utils.classification`` + core helpers."""

from uqlab.shared.utils.classification import auto_device, dino_transform, set_seed
from uqlab.shared.utils.core import get_device

__all__ = ["auto_device", "dino_transform", "get_device", "set_seed"]
