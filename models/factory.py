"""
Model Factory for Uncertainty Classification.

Provides a unified interface for creating models with MC Dropout support:
- DINOv2 + MLP (feature space)
- CNN with MC Dropout (end-to-end)
- ResNet18 with MC Dropout (end-to-end)

All models implement:
- forward(x) -> logits [B, C]
- mc_forward(x, n_passes) -> probabilities [T, B, C]
- extract_features(x) -> features [B, D] (for attribution)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from uqlab.shared.config.classification import ModelConfig

from .classification_models import EmbeddingDropoutMLP


class CNNMCDropout(nn.Module):
    """
    CNN with MC Dropout for end-to-end training.
    
    Based on Gaussian Logits architecture:
    Conv32 -> Conv64 -> Conv64 -> FC128 -> Dropout -> FC(num_classes)
    
    Features:
    - MC Dropout for uncertainty estimation
    - Feature extraction from FC128 layer for attribution
    - Compatible with DualXDA (has identifiable classifier layer)
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Feature layer (for attribution)
        # After 3 pooling operations: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc_features = nn.Linear(64 * 4 * 4, 128)
        
        # Classifier with dropout
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)
        
        self.dropout_rate = dropout
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from FC128 layer for attribution.
        
        Args:
            x: Input images [B, 3, 32, 32]
            
        Returns:
            features: Feature vectors [B, 128]
        """
        # Conv blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and extract features
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc_features(x))
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        
        Args:
            x: Input images [B, 3, 32, 32]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        features = self.extract_features(x)
        
        # Apply dropout if in training mode
        if self.training:
            features = self.dropout(features)
        
        logits = self.classifier(features)
        return logits
    
    def enable_dropout(self) -> None:
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def mc_forward(
        self,
        x: torch.Tensor,
        n_passes: int = 20,
        *,
        sample_batch_size: int = 256,
    ) -> torch.Tensor:
        """MC Dropout; backbone features computed once per sample chunk."""
        from uqlab.mc_dropout_uq import mc_forward_efficient

        return mc_forward_efficient(
            self, x, n_passes, sample_batch_size=sample_batch_size
        )


class ResNet18MCDropout(nn.Module):
    """
    ResNet18 with MC Dropout for both end-to-end and feature-space training.
    
    Uses torchvision ResNet18 as backbone with:
    - Dropout before final FC layer
    - MC Dropout for uncertainty estimation
    - Feature extraction from avgpool layer for attribution
    - Optional backbone freezing for feature-space mode (like DINOv2)
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: If True, freeze all backbone parameters (feature-space mode).
                        If False, train entire network (end-to-end mode).
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        pretrained: bool = False,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Load ResNet18 backbone
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Get feature dimension
        num_features = self.backbone.fc.in_features
        
        # Replace final FC layer with dropout + classifier
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Freeze backbone if requested (for feature-space mode)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Only classifier parameters will be trained
        
        self.dropout_rate = dropout
        self.num_features = num_features
        self.freeze_backbone = freeze_backbone
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from avgpool layer for attribution.
        
        Args:
            x: Input images [B, 3, 32, 32]
            
        Returns:
            features: Feature vectors [B, 512]
        """
        # Forward through ResNet backbone (without final FC)
        features = self.backbone(x)
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        
        Args:
            x: Input images [B, 3, 32, 32]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        features = self.extract_features(x)
        
        # Apply dropout if in training mode
        if self.training:
            features = self.dropout(features)
        
        logits = self.classifier(features)
        return logits
    
    def enable_dropout(self) -> None:
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    @torch.no_grad()
    def mc_forward(
        self,
        x: torch.Tensor,
        n_passes: int = 20,
        *,
        sample_batch_size: int = 256,
    ) -> torch.Tensor:
        """MC Dropout; backbone features computed once per sample chunk."""
        from uqlab.mc_dropout_uq import mc_forward_efficient

        return mc_forward_efficient(
            self, x, n_passes, sample_batch_size=sample_batch_size
        )


def build_model(
    config: ModelConfig,
    num_classes: int,
    feature_dim: Optional[int] = None
) -> nn.Module:
    """
    Build model based on architecture configuration.
    
    Factory function that instantiates the correct model architecture
    based on the configuration. All models implement the standard interface:
    - forward(x) -> logits [B, C]
    - mc_forward(x, n_passes) -> probabilities [T, B, C]
    - extract_features(x) -> features [B, D] (for CNN/ResNet)
    
    Args:
        config: Model configuration with architecture selection
        num_classes: Number of output classes
        feature_dim: Feature dimensionality (required for dinov2_mlp)
        
    Returns:
        Model implementing the standard uncertainty interface
        
    Raises:
        ValueError: If configuration is invalid or required parameters are missing
        
    Examples:
        >>> # DINOv2 + MLP (feature space)
        >>> config = ModelConfig(architecture="dinov2_mlp", hidden_dim=256, dropout=0.2)
        >>> model = build_model(config, num_classes=10, feature_dim=768)
        >>> 
        >>> # CNN MC Dropout (end-to-end)
        >>> config = ModelConfig(architecture="cnn_mcdropout", dropout=0.3)
        >>> model = build_model(config, num_classes=10)
        >>> 
        >>> # ResNet18 MC Dropout (end-to-end)
        >>> config = ModelConfig(architecture="resnet18_mcdropout", dropout=0.3)
        >>> model = build_model(config, num_classes=10)
    """
    if config.architecture == "dinov2_mlp":
        # Feature-space model: DINOv2 embeddings + MLP
        if feature_dim is None:
            raise ValueError(
                "feature_dim is required for dinov2_mlp architecture. "
                "This should be the dimensionality of DINOv2 embeddings (e.g., 768)."
            )
        
        if config.training_mode != "feature_space":
            raise ValueError(
                f"dinov2_mlp only supports feature_space training mode, "
                f"got: {config.training_mode}"
            )
        
        return EmbeddingDropoutMLP(
            input_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )
    
    elif config.architecture == "cnn_mcdropout":
        # End-to-end CNN with MC Dropout
        if config.training_mode != "end_to_end":
            raise ValueError(
                f"cnn_mcdropout requires end_to_end training mode, "
                f"got: {config.training_mode}"
            )
        
        return CNNMCDropout(
            num_classes=num_classes,
            dropout=config.dropout,
        )
    
    elif config.architecture == "resnet18_mcdropout":
        # ResNet18 with MC Dropout - supports both training modes
        # - feature_space: Frozen backbone (like DINOv2), train only classifier
        # - end_to_end: Train entire network including backbone
        
        # Use pretrained weights if not explicitly disabled
        pretrained = not config.use_untrained_resnet
        
        # Determine if backbone should be frozen
        freeze_backbone = (config.training_mode == "feature_space")
        
        return ResNet18MCDropout(
            num_classes=num_classes,
            dropout=config.dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    
    else:
        raise ValueError(
            f"Unknown architecture: {config.architecture}. "
            f"Supported architectures: dinov2_mlp, cnn_mcdropout, resnet18_mcdropout"
        )


# Convenience functions for direct instantiation
def build_dinov2_mlp(
    feature_dim: int,
    num_classes: int = 10,
    hidden_dim: int = 256,
    dropout: float = 0.2,
) -> EmbeddingDropoutMLP:
    """
    Build DINOv2 + MLP model for feature-space training.
    
    Args:
        feature_dim: DINOv2 embedding dimension (768 for base, 384 for small, etc.)
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
        
    Returns:
        EmbeddingDropoutMLP model
    """
    return EmbeddingDropoutMLP(
        input_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def build_cnn_mcdropout(
    num_classes: int = 10,
    dropout: float = 0.3,
) -> CNNMCDropout:
    """
    Build CNN with MC Dropout for end-to-end training.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        CNNMCDropout model
    """
    return CNNMCDropout(
        num_classes=num_classes,
        dropout=dropout,
    )


def build_resnet18_mcdropout(
    num_classes: int = 10,
    dropout: float = 0.3,
    pretrained: bool = False,
) -> ResNet18MCDropout:
    """
    Build ResNet18 with MC Dropout for end-to-end training.
    
    Args:
        num_classes: Number of output classes
        dropout: Dropout probability
        pretrained: Use ImageNet pretrained weights
        
    Returns:
        ResNet18MCDropout model
    """
    return ResNet18MCDropout(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    """Test model factory with all architectures."""
    
    print("Testing Model Factory\n" + "=" * 50)
    
    # Test 1: DINOv2 + MLP
    print("\n1. DINOv2 + MLP (feature space)")
    config_dinov2 = ModelConfig(
        architecture="dinov2_mlp",
        training_mode="feature_space",
        hidden_dim=256,
        dropout=0.2,
    )
    model_dinov2 = build_model(config_dinov2, num_classes=10, feature_dim=768)
    print(f"   Model: {model_dinov2.__class__.__name__}")
    
    # Test with embeddings
    embeddings = torch.randn(4, 768)
    logits = model_dinov2(embeddings)
    mc_preds = model_dinov2.mc_forward(embeddings, n_passes=5)
    print(f"   Forward: {embeddings.shape} -> {logits.shape}")
    print(f"   MC Forward: {embeddings.shape} -> {mc_preds.shape}")
    
    # Test 2: CNN MC Dropout
    print("\n2. CNN MC Dropout (end-to-end)")
    config_cnn = ModelConfig(
        architecture="cnn_mcdropout",
        training_mode="end_to_end",
        dropout=0.3,
    )
    model_cnn = build_model(config_cnn, num_classes=10)
    print(f"   Model: {model_cnn.__class__.__name__}")
    
    # Test with images
    images = torch.randn(4, 3, 32, 32)
    logits = model_cnn(images)
    features = model_cnn.extract_features(images)
    mc_preds = model_cnn.mc_forward(images, n_passes=5)
    print(f"   Forward: {images.shape} -> {logits.shape}")
    print(f"   Features: {images.shape} -> {features.shape}")
    print(f"   MC Forward: {images.shape} -> {mc_preds.shape}")
    
    # Test 3: ResNet18 MC Dropout
    print("\n3. ResNet18 MC Dropout (end-to-end)")
    config_resnet = ModelConfig(
        architecture="resnet18_mcdropout",
        training_mode="end_to_end",
        dropout=0.3,
        use_untrained_resnet=True,  # Don't download pretrained weights for test
    )
    model_resnet = build_model(config_resnet, num_classes=10)
    print(f"   Model: {model_resnet.__class__.__name__}")
    
    # Test with images
    logits = model_resnet(images)
    features = model_resnet.extract_features(images)
    mc_preds = model_resnet.mc_forward(images, n_passes=5)
    print(f"   Forward: {images.shape} -> {logits.shape}")
    print(f"   Features: {images.shape} -> {features.shape}")
    print(f"   MC Forward: {images.shape} -> {mc_preds.shape}")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")

# Made with Bob