"""Model backbones and baseline architectures."""

from .dinov2_backbone import create_dinov2_model, DINOv2Backbone

__all__ = ["create_dinov2_model", "DINOv2Backbone"]
