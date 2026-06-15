"""
Checkpoint helpers (small "model registry").

Goal: keep the repo ergonomics simple:
  - save only 1-2 files by default (best + last)
  - let all experiment scripts resolve a default checkpoint consistently
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def dtag_root() -> Path:
    # .../dtag/src/utils/checkpoints.py -> parents[2] == .../dtag
    return Path(__file__).resolve().parents[2]


def checkpoints_dir() -> Path:
    return dtag_root() / "checkpoints"


def default_best_checkpoint_path() -> Path:
    return checkpoints_dir() / "best.pth"


def default_last_checkpoint_path() -> Path:
    return checkpoints_dir() / "last.pth"


def registry_path() -> Path:
    return checkpoints_dir() / "registry.json"


def load_registry() -> Optional[Dict[str, Any]]:
    p = registry_path()
    if not p.exists():
        return None
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def resolve_checkpoint_path(user_path: Optional[str]) -> Optional[Path]:
    """
    Resolution order:
      1) explicit CLI path
      2) env var DTAG_CHECKPOINT
      3) ./checkpoints/best.pth (if it exists)
    """
    if user_path:
        return Path(user_path)

    env = os.environ.get("DTAG_CHECKPOINT")
    if env:
        return Path(env)

    p = default_best_checkpoint_path()
    if p.exists():
        return p

    # Last resort: use registry.json if present. This helps when the best checkpoint
    # lives in a non-default path or when callers run from a different CWD.
    reg = load_registry()
    if isinstance(reg, dict):
        best = reg.get("best")
        if isinstance(best, str) and best:
            cand = Path(best)
            if not cand.is_absolute():
                cand = dtag_root() / cand
            if cand.exists():
                return cand

    return None


def resolve_imagenet_root(user_path: Optional[str]) -> Optional[Path]:
    """
    Resolve ImageNet root directory for "no-training" baselines.

    Resolution order:
      1) explicit CLI path
      2) env var DTAG_IMAGENET_ROOT
      3) checkpoints/registry.json (optional key: imagenet_root)
    """
    if user_path:
        p = Path(user_path)
        if not p.is_absolute():
            p = dtag_root() / p
        return p

    env = os.environ.get("DTAG_IMAGENET_ROOT")
    if env:
        p = Path(env)
        if not p.is_absolute():
            p = dtag_root() / p
        return p

    reg = load_registry()
    if isinstance(reg, dict):
        val = reg.get("imagenet_root")
        if isinstance(val, str) and val:
            p = Path(val)
            if not p.is_absolute():
                p = dtag_root() / p
            return p

    # Convenience default: if user keeps ImageNet under ./data/imagenet, pick it up automatically.
    p = dtag_root() / "data" / "imagenet"
    if p.exists():
        return p

    return None


def write_registry(entry: Dict[str, Any]) -> None:
    """
    Writes/overwrites a minimal JSON registry at ./checkpoints/registry.json.
    """
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(entry, f, indent=2)


def normalize_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return config
    return str(config)


def load_checkpoint_weights(model: Any, checkpoint: Dict[str, Any]) -> None:
    """
    Load weights into a model using a small set of supported checkpoint formats.

    Supported:
      - {"model_state_dict": ...}  (full model, common PyTorch pattern)
      - {"classifier_state_dict": ...} (head-only; used for frozen-backbone setups like DINOv2)
    """
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return

    if "classifier_state_dict" in checkpoint:
        if not hasattr(model, "classifier"):
            raise ValueError("Checkpoint contains classifier_state_dict but model has no .classifier")
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        return

    raise ValueError("Unrecognized checkpoint format: expected model_state_dict or classifier_state_dict")


def detect_model_architecture(checkpoint: Dict[str, Any]) -> str:
    """
    Detect model architecture from checkpoint keys.
    
    Returns:
        "dinov2" if checkpoint contains DINOv2/ViT architecture
        "resnet" if checkpoint contains ResNet architecture
    """
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "classifier_state_dict" in checkpoint:
        # Classifier-only checkpoint, assume DINOv2
        return "dinov2"
    else:
        raise ValueError("Cannot detect architecture: no model_state_dict or classifier_state_dict")
    
    # Check for DINOv2/ViT specific keys
    dinov2_keys = [
        "backbone.embeddings.cls_token",
        "backbone.embeddings.position_embeddings",
        "backbone.encoder.layer.0.attention",
    ]
    
    # Check for ResNet specific keys
    resnet_keys = [
        "backbone.conv1.weight",
        "backbone.layer1.0.conv1.weight",
        "fc.weight",
    ]
    
    # Use substring matching because checkpoints typically contain deeper parameters like
    # "...attention.query.weight" even if our patterns are higher-level prefixes.
    state_keys = list(state_dict.keys())
    has_dinov2 = any(pattern in k for pattern in dinov2_keys for k in state_keys)
    has_resnet = any(pattern in k for pattern in resnet_keys for k in state_keys)
    
    if has_dinov2:
        return "dinov2"
    elif has_resnet:
        return "resnet"
    else:
        raise ValueError("Cannot detect architecture: checkpoint keys don't match known patterns")
