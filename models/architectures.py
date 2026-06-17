"""
Model Architectures - Classifier heads, model compositions, and architecture utilities.

This module consolidates architecture-related functionality from:
- src/models/baseline_models.py (VGG, ResNet classifiers)
- Various scattered model composition code

Provides:
- Classifier heads (Linear, MLP, Dropout-based)
- Complete model compositions (backbone + head)
- Model registry for easy instantiation
- Architecture utilities and helpers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision import models

from .feature_extractors import DINOv2Backbone, ResNetBackbone, SimpleCNNBackbone


# ============================================================================
# Classifier Heads
# ============================================================================

class LinearHead(nn.Module):
    """
    Simple linear classifier head.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        bias: Whether to use bias term
    
    Example:
        >>> head = LinearHead(input_dim=768, num_classes=10)
        >>> logits = head(features)  # [batch_size, 10]
    """
    
    def __init__(self, input_dim: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=bias)
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear head."""
        return self.fc(x)


class MLPHead(nn.Module):
    """
    Multi-layer perceptron classifier head.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability (0 = no dropout)
        activation: Activation function ('relu', 'gelu', 'tanh')
    
    Example:
        >>> head = MLPHead(input_dim=768, num_classes=10, 
        ...                hidden_dims=[512, 256], dropout=0.1)
        >>> logits = head(features)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP head."""
        return self.mlp(x)


class DropoutHead(nn.Module):
    """
    Classifier head with MC Dropout for uncertainty estimation.
    
    This head keeps dropout active during inference for Monte Carlo sampling.
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability (should be > 0 for MC Dropout)
        mc_samples: Number of MC samples during inference
    
    Example:
        >>> head = DropoutHead(input_dim=768, num_classes=10, dropout=0.2)
        >>> head.train()  # Enable dropout
        >>> logits = head(features)  # Single forward pass
        >>> 
        >>> # MC Dropout inference
        >>> predictions, uncertainty = head.mc_forward(features, n_samples=20)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.2,
        mc_samples: int = 20
    ):
        super().__init__()
        
        if dropout <= 0:
            raise ValueError("DropoutHead requires dropout > 0 for MC sampling")
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.mc_samples = mc_samples
        
        # Build MLP with dropout
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (single sample)."""
        return self.mlp(x)
    
    def mc_forward(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo forward pass for uncertainty estimation.
        
        Args:
            x: Input features [batch_size, input_dim]
            n_samples: Number of MC samples (default: self.mc_samples)
        
        Returns:
            predictions: Mean softmax probabilities [batch_size, num_classes]
            uncertainty: Predictive entropy [batch_size]
        """
        if n_samples is None:
            n_samples = self.mc_samples
        
        # Enable dropout
        self.train()
        
        # Collect MC samples
        logits_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                logits_samples.append(logits)
        
        # Stack samples: [n_samples, batch_size, num_classes]
        logits_samples = torch.stack(logits_samples, dim=0)
        
        # Convert to probabilities
        probs_samples = torch.softmax(logits_samples, dim=-1)
        
        # Mean prediction
        predictions = probs_samples.mean(dim=0)
        
        # Predictive entropy (uncertainty)
        entropy = -(predictions * torch.log(predictions + 1e-10)).sum(dim=-1)
        
        return predictions, entropy


# ============================================================================
# Complete Model Compositions
# ============================================================================

class ClassificationModel(nn.Module):
    """
    Complete classification model: backbone + classifier head.
    
    Args:
        backbone: Feature extractor (DINOv2, ResNet, etc.)
        head: Classifier head (Linear, MLP, Dropout)
        freeze_backbone: Whether to freeze backbone weights
    
    Example:
        >>> from .feature_extractors import DINOv2Backbone
        >>> backbone = DINOv2Backbone('base')
        >>> head = LinearHead(input_dim=768, num_classes=10)
        >>> model = ClassificationModel(backbone, head, freeze_backbone=True)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.backbone = backbone
        self.head = head
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, C, H, W]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        features = self.backbone.extract_features(x)
        logits = self.head(features)
        return logits
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with Maximum Softmax Probability (MSP) uncertainty.
        
        Args:
            x: Input images [batch_size, C, H, W]
        
        Returns:
            predictions: Softmax probabilities [batch_size, num_classes]
            uncertainty: MSP uncertainty (1 - max_prob) [batch_size]
        """
        logits = self.forward(x)
        predictions = torch.softmax(logits, dim=1)
        
        # MSP uncertainty: 1 - max(softmax)
        max_probs, _ = predictions.max(dim=1)
        uncertainty = 1.0 - max_probs
        
        return predictions, uncertainty


class MCDropoutModel(ClassificationModel):
    """
    Classification model with MC Dropout for uncertainty estimation.
    
    Requires a DropoutHead for MC sampling.
    
    Example:
        >>> backbone = DINOv2Backbone('base')
        >>> head = DropoutHead(input_dim=768, num_classes=10, dropout=0.2)
        >>> model = MCDropoutModel(backbone, head, freeze_backbone=True)
        >>> predictions, uncertainty = model.predict_with_uncertainty(images)
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        head: DropoutHead,
        freeze_backbone: bool = False
    ):
        super().__init__(backbone, head, freeze_backbone)
        
        if not isinstance(head, DropoutHead):
            raise TypeError("MCDropoutModel requires a DropoutHead")
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with MC Dropout uncertainty.
        
        Args:
            x: Input images [batch_size, C, H, W]
            n_samples: Number of MC samples
        
        Returns:
            predictions: Mean softmax probabilities [batch_size, num_classes]
            uncertainty: Predictive entropy [batch_size]
        """
        features = self.backbone.extract_features(x)
        predictions, uncertainty = self.head.mc_forward(features, n_samples)
        return predictions, uncertainty


# ============================================================================
# Model Registry & Factory
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model creation."""
    backbone_type: str  # 'dinov2', 'resnet', 'simple_cnn'
    backbone_variant: str  # 'base', 'resnet50', etc.
    head_type: str  # 'linear', 'mlp', 'dropout'
    num_classes: int = 10
    freeze_backbone: bool = True
    
    # Head-specific configs
    hidden_dims: Optional[List[int]] = None
    dropout: float = 0.0
    mc_samples: int = 20
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512]


class ModelRegistry:
    """
    Registry for creating models from configurations.
    
    Example:
        >>> config = ModelConfig(
        ...     backbone_type='dinov2',
        ...     backbone_variant='base',
        ...     head_type='dropout',
        ...     num_classes=10,
        ...     dropout=0.2
        ... )
        >>> model = ModelRegistry.create(config)
    """
    
    @staticmethod
    def create_backbone(backbone_type: str, variant: str) -> nn.Module:
        """Create backbone from type and variant."""
        if backbone_type == 'dinov2':
            return DINOv2Backbone(model_name=variant)
        elif backbone_type == 'resnet':
            return ResNetBackbone(variant=variant)
        elif backbone_type == 'simple_cnn':
            return SimpleCNNBackbone()
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    @staticmethod
    def create_head(
        head_type: str,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]],
        dropout: float,
        mc_samples: int
    ) -> nn.Module:
        """Create classifier head from configuration."""
        # Default hidden dims if None
        if hidden_dims is None:
            hidden_dims = [512]
        
        if head_type == 'linear':
            return LinearHead(input_dim, num_classes)
        elif head_type == 'mlp':
            return MLPHead(
                input_dim, num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif head_type == 'dropout':
            return DropoutHead(
                input_dim, num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
                mc_samples=mc_samples
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")
    
    @staticmethod
    def create(config: ModelConfig, device: str = 'cuda') -> nn.Module:
        """
        Create complete model from configuration.
        
        Args:
            config: Model configuration
            device: Device to load model on
        
        Returns:
            Complete classification model
        """
        # Create backbone
        backbone = ModelRegistry.create_backbone(
            config.backbone_type,
            config.backbone_variant
        )
        
        # Get feature dimension (it's an attribute, not a method)
        if hasattr(backbone, 'feature_dim'):
            input_dim: int = backbone.feature_dim  # type: ignore
        else:
            raise AttributeError(f"Backbone {type(backbone)} missing feature_dim attribute")
        
        # Create head
        head = ModelRegistry.create_head(
            config.head_type,
            input_dim,
            config.num_classes,
            config.hidden_dims,
            config.dropout,
            config.mc_samples
        )
        
        # Create complete model
        if config.head_type == 'dropout':
            # Type assertion for mypy/pyright
            if not isinstance(head, DropoutHead):
                raise TypeError(f"Expected DropoutHead, got {type(head)}")
            model = MCDropoutModel(backbone, head, config.freeze_backbone)
        else:
            model = ClassificationModel(backbone, head, config.freeze_backbone)
        
        # Move to device
        model = model.to(device)
        
        return model


# ============================================================================
# Convenience Functions
# ============================================================================

def create_model(
    backbone: str = 'dinov2-base',
    head: str = 'linear',
    num_classes: int = 10,
    freeze_backbone: bool = True,
    dropout: float = 0.0,
    device: str = 'cuda'
) -> nn.Module:
    """
    Convenience function to create a model.
    
    Args:
        backbone: Backbone specification (e.g., 'dinov2-base', 'resnet50')
        head: Head type ('linear', 'mlp', 'dropout')
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone
        dropout: Dropout probability (for 'mlp' and 'dropout' heads)
        device: Device to load model on
    
    Returns:
        Complete classification model
    
    Example:
        >>> model = create_model('dinov2-base', 'dropout', num_classes=10, dropout=0.2)
    """
    # Parse backbone specification
    if backbone.startswith('dinov2'):
        backbone_type = 'dinov2'
        variant = backbone.replace('dinov2-', '')
    elif backbone.startswith('resnet'):
        backbone_type = 'resnet'
        variant = backbone
    elif backbone == 'simple_cnn':
        backbone_type = 'simple_cnn'
        variant = 'default'
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    # Create config
    config = ModelConfig(
        backbone_type=backbone_type,
        backbone_variant=variant,
        head_type=head,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        dropout=dropout
    )
    
    return ModelRegistry.create(config, device)


# Made with Bob