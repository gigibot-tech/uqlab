"""
MC Dropout Model Implementation
Implements Monte Carlo Dropout for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MCDropoutResNet(nn.Module):
    """
    ResNet with MC Dropout enabled during inference.
    
    This model keeps dropout active during test time to enable
    stochastic forward passes for uncertainty estimation.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3, pretrained=False):
        super(MCDropoutResNet, self).__init__()
        
        # Load base ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final FC layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add dropout and classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)
        
        self.dropout_rate = dropout_rate
        
    def forward(self, x, enable_dropout=True):
        """
        Forward pass with optional dropout.
        
        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            enable_dropout: If True, apply dropout even in eval mode
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        features = self.backbone(x)
        
        # Apply dropout (even in eval mode if enable_dropout=True)
        if enable_dropout:
            # CRITICAL: Set dropout to training mode to enable it during inference
            self.dropout.train()
            features = self.dropout(features)
        else:
            if self.training:
                features = self.dropout(features)
        
        logits = self.fc(features)
        return logits
    
    def mc_forward(self, x, n_passes=50):
        """
        Perform multiple stochastic forward passes for MC Dropout.
        
        Args:
            x: Input tensor [batch_size, 3, 32, 32]
            n_passes: Number of forward passes
            
        Returns:
            predictions: Stacked predictions [n_passes, batch_size, num_classes]
        """
        self.eval()  # Set to eval mode but keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x, enable_dropout=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions: [n_passes, batch_size, num_classes]
        predictions = torch.stack(predictions, dim=0)
        return predictions


class MCDropoutCNN(nn.Module):
    """
    Simple CNN with MC Dropout for faster experimentation.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(MCDropoutCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout_rate = dropout_rate
        
    def forward(self, x, enable_dropout=True):
        # Conv blocks
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers with dropout
        if enable_dropout:
            # CRITICAL: Set dropout to training mode to enable it during inference
            self.dropout.train()
            x = self.dropout(F.relu(self.fc1(x)))
        else:
            if self.training:
                x = self.dropout(F.relu(self.fc1(x)))
            else:
                x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x
    
    def mc_forward(self, x, n_passes=50):
        """MC Dropout forward passes."""
        self.eval()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward(x, enable_dropout=True)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions, dim=0)
        return predictions


def create_mc_dropout_model(architecture="resnet18", num_classes=10, 
                            dropout_rate=0.3, pretrained=False):
    """
    Factory function to create MC Dropout models.
    
    Args:
        architecture: Model architecture (resnet18, cnn)
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Use pretrained weights
        
    Returns:
        model: MC Dropout model
    """
    if architecture == "resnet18":
        return MCDropoutResNet(num_classes, dropout_rate, pretrained)
    elif architecture == "cnn":
        return MCDropoutCNN(num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test the model
    model = create_mc_dropout_model("resnet18", num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    
    # Standard forward
    logits = model(x)
    print(f"Standard forward: {logits.shape}")
    
    # MC Dropout forward
    mc_predictions = model.mc_forward(x, n_passes=10)
    print(f"MC Dropout forward: {mc_predictions.shape}")
    
    # Calculate variance
    variance = mc_predictions.var(dim=0)
    print(f"Prediction variance: {variance.shape}")

# Made with Bob
