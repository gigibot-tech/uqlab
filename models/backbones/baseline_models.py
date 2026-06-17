"""
Classic Baseline Models for AURC Benchmarking
Implements VGG-16 and ResNet-50 with Maximum Softmax Probability (MSP) uncertainty.

These are the standard baselines from the selective classification literature:
- Geifman & El-Yaniv "Selective Classification" (JMLR 2017)
- Hendrycks & Gimpel "A Baseline for Detecting Misclassified Examples" (ICLR 2017)
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple, Optional


class BaselineVGG16(nn.Module):
    """
    VGG-16 with Maximum Softmax Probability (MSP) for uncertainty.
    
    This is the classic baseline from selective classification papers.
    Uncertainty = 1 - max(softmax(logits))
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(BaselineVGG16, self).__init__()
        
        # Load pretrained VGG-16
        self.backbone = models.vgg16(pretrained=pretrained)
        
        # Replace classifier for target dataset
        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x):
        """
        Predict with MSP uncertainty.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            predictions: Softmax probabilities [batch_size, num_classes]
            uncertainty: MSP uncertainty scores [batch_size]
        """
        logits = self.forward(x)
        predictions = torch.softmax(logits, dim=1)
        
        # MSP uncertainty: 1 - max(softmax)
        max_probs, _ = predictions.max(dim=1)
        uncertainty = 1.0 - max_probs
        
        return predictions, uncertainty


class BaselineResNet50(nn.Module):
    """
    ResNet-50 with Maximum Softmax Probability (MSP) for uncertainty.
    
    This is the modern baseline - deeper than ResNet-18 but still standard.
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(BaselineResNet50, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x):
        """
        Predict with MSP uncertainty.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            predictions: Softmax probabilities [batch_size, num_classes]
            uncertainty: MSP uncertainty scores [batch_size]
        """
        logits = self.forward(x)
        predictions = torch.softmax(logits, dim=1)
        
        # MSP uncertainty: 1 - max(softmax)
        max_probs, _ = predictions.max(dim=1)
        uncertainty = 1.0 - max_probs
        
        return predictions, uncertainty


class BaselineResNet18(nn.Module):
    """
    ResNet-18 with Maximum Softmax Probability (MSP) for uncertainty.
    
    Lighter baseline for faster experiments.
    """
    
    def __init__(self, num_classes=10, pretrained=True):
        super(BaselineResNet18, self).__init__()
        
        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x):
        """
        Predict with MSP uncertainty.
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            predictions: Softmax probabilities [batch_size, num_classes]
            uncertainty: MSP uncertainty scores [batch_size]
        """
        logits = self.forward(x)
        predictions = torch.softmax(logits, dim=1)
        
        # MSP uncertainty: 1 - max(softmax)
        max_probs, _ = predictions.max(dim=1)
        uncertainty = 1.0 - max_probs
        
        return predictions, uncertainty


def create_baseline_model(
    model_name: str,
    num_classes: int = 10,
    pretrained: bool = True,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create baseline models.
    
    Args:
        model_name: One of ['vgg16', 'resnet50', 'resnet18']
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        device: Device to load model on
        
    Returns:
        Baseline model with MSP uncertainty
    """
    model_map = {
        'vgg16': BaselineVGG16,
        'resnet50': BaselineResNet50,
        'resnet18': BaselineResNet18,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    
    model = model_map[model_name](num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    
    return model


def batch_predict_with_msp(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batch prediction with MSP uncertainty for baseline models.
    
    Args:
        model: Baseline model (VGG-16, ResNet-50, etc.)
        dataloader: Data loader
        device: Device to run on
        
    Returns:
        all_predictions: Softmax probabilities [N, num_classes]
        all_uncertainties: MSP uncertainty scores [N]
        all_labels: Ground truth labels [N]
    """
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            # Get predictions and MSP uncertainty
            predictions, uncertainty = model.predict_with_uncertainty(batch_x)
            
            all_predictions.append(predictions.cpu())
            all_uncertainties.append(uncertainty.cpu())
            all_labels.append(batch_y)
    
    return (
        torch.cat(all_predictions, dim=0),
        torch.cat(all_uncertainties, dim=0),
        torch.cat(all_labels, dim=0)
    )


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")
    
    # Create models
    vgg = create_baseline_model('vgg16', num_classes=10, pretrained=False, device='cpu')
    resnet50 = create_baseline_model('resnet50', num_classes=10, pretrained=False, device='cpu')
    resnet18 = create_baseline_model('resnet18', num_classes=10, pretrained=False, device='cpu')
    
    # Test forward pass
    x = torch.randn(4, 3, 224, 224)
    
    for name, model in [('VGG-16', vgg), ('ResNet-50', resnet50), ('ResNet-18', resnet18)]:
        predictions, uncertainty = model.predict_with_uncertainty(x)
        print(f"\n{name}:")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Uncertainty shape: {uncertainty.shape}")
        print(f"  Mean uncertainty: {uncertainty.mean():.4f}")
        print(f"  Uncertainty range: [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    
    print("\n✅ All baseline models working correctly!")

# Made with Bob
