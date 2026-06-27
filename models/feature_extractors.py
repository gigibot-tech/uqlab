"""
Model-agnostic feature extraction system for uncertainty classification.

This module provides a unified interface for extracting features from any model architecture:
- DINOv2: Extract features from pretrained vision transformer
- CNN: Extract features from custom CNN backbone (before classifier)
- ResNet: Extract features from ResNet avgpool layer

All extractors implement a common interface for consistent feature extraction and caching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from uqlab.shared.config.classification import ModelConfig
from uqlab.data.loaders.cifar10n_loader import CIFAR10NDataset
from uqlab.data.experiment_loader import EmbeddingOrganizer, SplitSpec


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extraction.
    
    All feature extractors must implement:
    1. extract_features() - Extract features for entire dataset
    2. get_feature_dim() - Return feature dimensionality
    
    This enables:
    - Unified interface across architectures
    - Consistent caching mechanism
    - Easy addition of new architectures
    """
    
    @abstractmethod
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract features for entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            features: Extracted features [N, feature_dim]
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Return feature dimensionality.
        
        Returns:
            feature_dim: Dimension of extracted features
        """
        pass


class DINOv2FeatureExtractor(FeatureExtractor):
    """
    Feature extractor for DINOv2 models.
    
    Wraps the existing EmbeddingOrganizer to maintain backward compatibility
    while providing the unified FeatureExtractor interface.
    
    Features:
    - Uses pretrained DINOv2 vision transformer
    - Extracts CLS token embeddings (768-dim for small, 1024 for base, etc.)
    - Leverages existing caching mechanism
    - Works in feature_space training mode
    
    Args:
        dataset: CIFAR-10N dataset
        split_spec: Train/eval split specification
        feature_cache_dir: Directory for embedding caching
        noise_type: CIFAR-10N noise type
        dinov2_model: DINOv2 model variant (e.g., "dinov2_vits14")
        batch_size: Batch size for feature extraction
        device: Device to run on
    """
    
    def __init__(
        self,
        dataset: CIFAR10NDataset,
        split_spec: SplitSpec,
        feature_cache_dir: Path,
        noise_type: str,
        dinov2_model: str,
        batch_size: int,
        device: torch.device,
    ):
        self.dataset = dataset
        self.split_spec = split_spec
        self.feature_cache_dir = feature_cache_dir
        self.noise_type = noise_type
        self.dinov2_model = dinov2_model
        self.batch_size = batch_size
        self.device = device
        
        # Create EmbeddingOrganizer (existing implementation)
        self.organizer = EmbeddingOrganizer(
            dataset=dataset,
            split_spec=split_spec,
            feature_cache_dir=feature_cache_dir,
            noise_type=noise_type,
            dinov2_model=dinov2_model,
            batch_size=batch_size,
            device=device,
        )
        
        # Feature dimension depends on DINOv2 variant
        self._feature_dim = self._get_dinov2_feature_dim(dinov2_model)
    
    def _get_dinov2_feature_dim(self, model_name: str) -> int:
        """Get feature dimension for DINOv2 model variant."""
        if "vits14" in model_name or "small" in model_name:
            return 384
        elif "vitb14" in model_name or "base" in model_name:
            return 768
        elif "vitl14" in model_name or "large" in model_name:
            return 1024
        elif "vitg14" in model_name or "giant" in model_name:
            return 1536
        else:
            # Default to base model dimension
            return 768
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract DINOv2 features using EmbeddingOrganizer.
        
        Note: This method signature matches the base class, but DINOv2
        extraction is handled by EmbeddingOrganizer which works with
        the full dataset and split specification.
        
        Args:
            dataloader: DataLoader (not used, kept for interface compatibility)
            
        Returns:
            features: DINOv2 embeddings [N, 768] (or other dim based on model)
        """
        # Load or compute features using existing mechanism
        self.organizer.load_or_compute_features()
        
        # Return train pack features as example
        # In practice, you'd call get_train_pack(), get_clean_eval_pack(), etc.
        train_pack = self.organizer.get_train_pack()
        return train_pack["features"]
    
    def get_feature_dim(self) -> int:
        """Return DINOv2 feature dimensionality."""
        return self._feature_dim
    
    def get_train_pack(self) -> Dict[str, torch.Tensor]:
        """Get training data pack with features and labels."""
        return self.organizer.get_train_pack()
    
    def get_clean_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Get clean evaluation data pack."""
        return self.organizer.get_clean_eval_pack()
    
    def get_aleatoric_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Get aleatoric uncertainty evaluation pack."""
        return self.organizer.get_aleatoric_eval_pack()
    
    def get_epistemic_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Get epistemic uncertainty evaluation pack."""
        return self.organizer.get_epistemic_eval_pack()

    def get_ood_eval_pack(self) -> Dict[str, torch.Tensor]:
        """Get OOD evaluation pack (four-region mode)."""
        return self.organizer.get_ood_eval_pack()


class CNNFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for custom CNN models.
    
    Extracts features from the CNN backbone before the final classifier layer.
    This enables:
    - End-to-end training on raw images
    - Feature extraction for attribution signals
    - MC Dropout uncertainty estimation
    
    Architecture:
        Conv layers -> Global Average Pooling -> [Features] -> Classifier
                                                      ↑
                                              Extract here
    
    Args:
        model: CNN model with feature extraction capability
        device: Device to run on
        batch_size: Batch size for feature extraction
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Determine feature dimension by running a dummy forward pass
        self._feature_dim = self._infer_feature_dim()
    
    def _infer_feature_dim(self) -> int:
        """Infer feature dimension by running a dummy forward pass."""
        self.model.eval()
        with torch.no_grad():
            # Create dummy input (CIFAR-10 size: 3x32x32)
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            
            # Extract features using the model's feature extraction method
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(dummy_input)
            else:
                # Fallback: assume model has a 'features' attribute (backbone)
                # and extract before final classifier
                if hasattr(self.model, 'features'):
                    features = self.model.features(dummy_input)
                    # Apply global average pooling if needed
                    if len(features.shape) == 4:  # [B, C, H, W]
                        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = features.view(features.size(0), -1)
                else:
                    raise ValueError(
                        "CNN model must have either 'extract_features' method "
                        "or 'features' attribute for feature extraction"
                    )
            
            return features.shape[1]
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract features from CNN backbone for entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            features: Extracted features [N, feature_dim]
        """
        self.model.eval()
        all_features: List[torch.Tensor] = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]  # First element is always images
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Extract features
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(images)
                else:
                    # Fallback to features attribute
                    features = self.model.features(images)
                    if len(features.shape) == 4:  # [B, C, H, W]
                        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                        features = features.view(features.size(0), -1)
                
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def get_feature_dim(self) -> int:
        """Return CNN feature dimensionality."""
        return self._feature_dim


class ResNetFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for ResNet models.
    
    Extracts features from the avgpool layer (before final FC layer).
    This provides:
    - 512-dim features for ResNet18/34
    - 2048-dim features for ResNet50/101/152
    - End-to-end training capability
    - MC Dropout uncertainty estimation
    
    Architecture:
        ResNet blocks -> AvgPool -> [Features] -> FC
                                         ↑
                                   Extract here
    
    Args:
        model: ResNet model (can be pretrained or untrained)
        device: Device to run on
        batch_size: Batch size for feature extraction
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int = 64,
    ):
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Determine feature dimension
        self._feature_dim = self._infer_feature_dim()
    
    def _infer_feature_dim(self) -> int:
        """Infer feature dimension from ResNet architecture."""
        # Try to get feature dimension from model structure
        if hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features'):
            return self.model.fc.in_features
        
        # Fallback: run dummy forward pass
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(dummy_input)
            else:
                # Extract features manually by removing final FC layer
                # Standard ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
                x = self.model.conv1(dummy_input)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                
                x = self.model.avgpool(x)
                features = torch.flatten(x, 1)
            
            return features.shape[1]
    
    def extract_features(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract features from ResNet avgpool layer for entire dataset.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            features: Extracted features [N, 512 or 2048]
        """
        self.model.eval()
        all_features: List[torch.Tensor] = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]  # First element is always images
                else:
                    images = batch
                
                images = images.to(self.device)
                
                # Extract features
                if hasattr(self.model, 'extract_features'):
                    features = self.model.extract_features(images)
                else:
                    # Extract features manually
                    x = self.model.conv1(images)
                    x = self.model.bn1(x)
                    x = self.model.relu(x)
                    x = self.model.maxpool(x)
                    
                    x = self.model.layer1(x)
                    x = self.model.layer2(x)
                    x = self.model.layer3(x)
                    x = self.model.layer4(x)
                    
                    x = self.model.avgpool(x)
                    features = torch.flatten(x, 1)
                
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def get_feature_dim(self) -> int:
        """Return ResNet feature dimensionality."""
        return self._feature_dim


def create_feature_extractor(
    config: ModelConfig,
    device: torch.device,
    dataset: Optional[CIFAR10NDataset] = None,
    split_spec: Optional[SplitSpec] = None,
    feature_cache_dir: Optional[Path] = None,
    noise_type: Optional[str] = None,
    batch_size: int = 64,
    model: Optional[nn.Module] = None,
) -> FeatureExtractor:
    """
    Factory function to create the appropriate feature extractor based on configuration.
    
    This function selects the correct feature extractor implementation based on
    the model architecture specified in the configuration.
    
    Args:
        config: Model configuration specifying architecture
        device: Device to run on
        dataset: CIFAR-10N dataset (required for DINOv2)
        split_spec: Train/eval split specification (required for DINOv2)
        feature_cache_dir: Directory for caching (required for DINOv2)
        noise_type: CIFAR-10N noise type (required for DINOv2)
        batch_size: Batch size for feature extraction
        model: Pre-initialized model (required for CNN/ResNet)
        
    Returns:
        FeatureExtractor: Appropriate feature extractor for the architecture
        
    Raises:
        ValueError: If required arguments are missing for the selected architecture
        
    Example:
        >>> # DINOv2 extractor
        >>> extractor = create_feature_extractor(
        ...     config=config,
        ...     device=device,
        ...     dataset=dataset,
        ...     split_spec=split_spec,
        ...     feature_cache_dir=Path("./cache"),
        ...     noise_type="worse_label",
        ... )
        >>> 
        >>> # CNN extractor
        >>> extractor = create_feature_extractor(
        ...     config=config,
        ...     device=device,
        ...     model=cnn_model,
        ... )
    """
    from uqlab.models.architecture import normalize_architecture, normalize_dinov2_model

    arch = normalize_architecture(config.architecture)
    if arch == "dinov2_mlp":
        # DINOv2 requires dataset and split information
        if dataset is None or split_spec is None or feature_cache_dir is None or noise_type is None:
            raise ValueError(
                "DINOv2 feature extractor requires: dataset, split_spec, "
                "feature_cache_dir, and noise_type"
            )

        dinov2_model = normalize_dinov2_model(config.dinov2_model)
        
        return DINOv2FeatureExtractor(
            dataset=dataset,
            split_spec=split_spec,
            feature_cache_dir=feature_cache_dir,
            noise_type=noise_type,
            dinov2_model=dinov2_model,
            batch_size=batch_size,
            device=device,
        )
    
    elif arch in ("cnn_small", "pixel_mlp"):
        # End-to-end models with extract_features() (CNN backbone or pixel MLP).
        if model is None:
            raise ValueError(f"{arch} feature extractor requires a pre-initialized model")

        return CNNFeatureExtractor(
            model=model,
            device=device,
            batch_size=batch_size,
        )

    elif arch == "resnet18":
        # ResNet requires pre-initialized model
        if model is None:
            raise ValueError("ResNet feature extractor requires a pre-initialized model")
        
        return ResNetFeatureExtractor(
            model=model,
            device=device,
            batch_size=batch_size,
        )
    
    else:
        raise ValueError(
            f"Unknown architecture: {config.architecture}. "
            f"Supported: dinov2_mlp, cnn_mcdropout, pixel_mlp, resnet18_mcdropout"
        )


# Made with Bob