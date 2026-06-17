"""
DINOv2 Backbone Integration for dtag

Provides pre-trained DINOv2 models as feature extractors for uncertainty quantification.
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Optional, Tuple, Dict
from pathlib import Path


class DINOv2Backbone(nn.Module):
    """
    DINOv2 pre-trained backbone for feature extraction.
    
    Features:
    - Automatic caching via Hugging Face
    - Multiple model sizes (small, base, large, giant)
    - Optional register tokens for better performance
    - Compatible with MC Dropout and uncertainty quantification
    
    Example:
        >>> backbone = DINOv2Backbone(model_name='large', num_classes=10)
        >>> features = backbone(images)  # Extract features
        >>> logits = backbone.classifier(features)  # Classification
    """
    
    AVAILABLE_MODELS = {
        'small': 'facebook/dinov2-small',
        'base': 'facebook/dinov2-base',
        'large': 'facebook/dinov2-large',
        'giant': 'facebook/dinov2-giant',
        'small-reg': 'facebook/dinov2-small-reg',
        'base-reg': 'facebook/dinov2-base-reg',
        'large-reg': 'facebook/dinov2-large-reg',
        'giant-reg': 'facebook/dinov2-giant-reg',
    }
    
    FEATURE_DIMS = {
        'small': 384,
        'base': 768,
        'large': 1024,
        'giant': 1536,
    }
    
    def __init__(
        self,
        model_name: str = 'large',
        num_classes: Optional[int] = None,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
        use_cls_token: bool = True,
    ):
        """
        Initialize DINOv2 backbone.
        
        Args:
            model_name: Model variant ('small', 'base', 'large', 'giant', with optional '-reg')
            num_classes: Number of output classes (None = feature extraction only)
            dropout_rate: Dropout rate for classifier head
            freeze_backbone: Whether to freeze backbone weights
            use_cls_token: Use [CLS] token (True) or mean pooling (False)
        """
        super().__init__()
        
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} not found. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_cls_token = use_cls_token
        
        # Get feature dimension
        base_name = model_name.replace('-reg', '')
        self.feature_dim = self.FEATURE_DIMS[base_name]
        
        # Load pre-trained model
        model_id = self.AVAILABLE_MODELS[model_name]
        print(f"Loading DINOv2 model: {model_id}")
        
        self.backbone = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen (no gradient updates)")
        
        # Add classifier head if num_classes specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, num_classes)
            )
        else:
            self.classifier = None
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            features: [B, feature_dim]
        """
        outputs = self.backbone(pixel_values=x)
        hidden_states = outputs.last_hidden_state  # [B, num_patches, feature_dim]
        
        if self.use_cls_token:
            # Use [CLS] token (first token)
            features = hidden_states[:, 0]
        else:
            # Mean pooling over all patches
            features = hidden_states.mean(dim=1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            If classifier exists: logits [B, num_classes]
            Otherwise: features [B, feature_dim]
        """
        features = self.extract_features(x)
        
        if self.classifier is not None:
            return self.classifier(features)
        else:
            return features
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'model_id': self.AVAILABLE_MODELS[self.model_name],
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'use_cls_token': self.use_cls_token,
        }


class DINOv2WithMCDropout(DINOv2Backbone):
    """
    DINOv2 backbone with MC Dropout for uncertainty quantification.
    
    Adds dropout layers that remain active during inference for epistemic
    uncertainty estimation via Monte Carlo sampling.
    """
    
    def __init__(
        self,
        model_name: str = 'large',
        num_classes: int = 10,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True,  # Usually freeze for MC Dropout
        use_cls_token: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
            use_cls_token=use_cls_token,
        )
        
        # Replace classifier with MC Dropout version
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_forward(
        self,
        x: torch.Tensor,
        n_passes: int = 10
    ) -> torch.Tensor:
        """
        Monte Carlo forward pass for uncertainty estimation.
        
        Args:
            x: Input images [B, C, H, W]
            n_passes: Number of MC forward passes
        
        Returns:
            predictions: Stacked predictions [n_passes, B, num_classes]
        """
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_passes, B, num_classes]
        
        return predictions


def create_dinov2_model(
    model_name: str = 'large',
    num_classes: int = 10,
    dropout_rate: float = 0.3,
    mc_dropout: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Factory function to create DINOv2 models.
    
    Args:
        model_name: Model variant
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        mc_dropout: Use MC Dropout version
        freeze_backbone: Freeze backbone weights
    
    Returns:
        DINOv2 model
    """
    if mc_dropout:
        return DINOv2WithMCDropout(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )
    else:
        return DINOv2Backbone(
            model_name=model_name,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            freeze_backbone=freeze_backbone,
        )


if __name__ == '__main__':
    # Test the model
    print("Testing DINOv2 backbone...")
    
    model = create_dinov2_model(
        model_name='base',  # Use smaller model for testing
        num_classes=10,
        mc_dropout=True,
    )
    
    print(f"\nModel info:")
    for k, v in model.get_model_info().items():
        print(f"  {k}: {v}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    print(f"\nTesting forward pass...")
    logits = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    
    print(f"\nTesting MC Dropout...")
    predictions = model.mc_forward(x, n_passes=5)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Mean prediction: {predictions.mean(dim=0).shape}")
    print(f"  Variance: {predictions.var(dim=0).mean().item():.6f}")
    
    print("\n✅ All tests passed!")

# Made with Bob
