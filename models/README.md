# UQLab Models Module

This module provides model architectures, feature extractors, and uncertainty quantification utilities for the UQLab uncertainty classification framework.

## 📁 Module Structure

```
2_models/
├── README.md                    # This file
├── __init__.py                  # Module exports
├── factory.py                   # Model factory and configuration
├── architectures.py             # Classifier heads and model compositions
├── classification_models.py     # Specialized classification models
├── feature_extractors.py        # Feature extraction interfaces
└── uncertainty.py               # MC Dropout and uncertainty utilities
```

## 🎯 Quick Start

```python
from uqlab.shared.config.classification import ModelConfig
from uqlab.2_models.factory import build_model

# Create a model configuration
config = ModelConfig(
    architecture="resnet18_mcdropout",
    training_mode="feature_space",  # or "end_to_end"
    dropout=0.3,
    use_untrained_resnet=False
)

# Build the model
model = build_model(config, num_classes=10)

# Use MC Dropout for uncertainty estimation
model.eval()
predictions = model.mc_forward(images, n_passes=20)
```

## 📄 File Descriptions

### [`factory.py`](./factory.py) - Model Factory & Core Architectures

**Purpose**: Central factory for creating models with consistent configuration.

**Key Components**:
- `ModelConfig`: Configuration dataclass for model parameters
- `DINOv2MLP`: DINOv2 features + MLP classifier (feature-space only)
- `CNNMCDropout`: Custom CNN with MC Dropout (end-to-end only)
- `ResNet18MCDropout`: ResNet18 with MC Dropout (both modes) ⭐ **NEW**
- `build_model()`: Factory function to instantiate models

**Supported Architectures**:

| Architecture | Training Modes | MC Dropout | Use Case |
|-------------|----------------|------------|----------|
| `dinov2_mlp` | `feature_space` | ✅ | Pre-extracted DINOv2 features |
| `cnn_mcdropout` | `end_to_end` | ✅ | Custom CNN from scratch |
| `resnet18_mcdropout` | `feature_space`, `end_to_end` | ✅ | ResNet18 with flexible training |

**Training Modes**:
- **`feature_space`**: Train only classifier head on frozen backbone features
  - Faster training, lower memory
  - Required for Data Attribution (DualXDA)
  - Backbone parameters frozen (`requires_grad=False`)
  
- **`end_to_end`**: Train entire network including backbone
  - Full fine-tuning on your dataset
  - Higher capacity, more parameters
  - Backbone parameters trainable

**Recent Updates** (2026-06-15):
- ✅ Added `freeze_backbone` parameter to `ResNet18MCDropout`
- ✅ ResNet now supports both `feature_space` and `end_to_end` modes
- ✅ Removed restriction forcing ResNet to `end_to_end` only

**Dependencies**:
```python
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0  # For DINOv2
```

**Example Usage**:
```python
# Feature-space mode (frozen ResNet backbone)
config = ModelConfig(
    architecture="resnet18_mcdropout",
    training_mode="feature_space",
    dropout=0.3,
    use_untrained_resnet=False  # Use ImageNet pretrained
)
model = build_model(config, num_classes=10)
# Only classifier head will be trained

# End-to-end mode (full fine-tuning)
config = ModelConfig(
    architecture="resnet18_mcdropout",
    training_mode="end_to_end",
    dropout=0.3,
    use_untrained_resnet=False
)
model = build_model(config, num_classes=10)
# Entire network will be trained
```

---

### [`architectures.py`](./architectures.py) - Classifier Heads & Compositions

**Purpose**: Reusable classifier heads and model composition utilities.

**Key Components**:
- `LinearHead`: Simple linear classifier
- `MLPHead`: Multi-layer perceptron classifier
- `DropoutMLPHead`: MLP with dropout for uncertainty
- Model composition utilities

**Example Usage**:
```python
from uqlab.2_models.architectures import DropoutMLPHead

# Create a classifier head
head = DropoutMLPHead(
    input_dim=768,      # DINOv2 feature dimension
    hidden_dim=256,
    num_classes=10,
    dropout=0.3
)

# Use with pre-extracted features
logits = head(dinov2_features)
```

**Dependencies**:
```python
torch>=2.0.0
torchvision>=0.15.0
```

---

### [`classification_models.py`](./classification_models.py) - Specialized Models

**Purpose**: Specialized models for specific use cases.

**Key Components**:
- `EmbeddingDataset`: Dataset for pre-extracted embeddings
  - Compatible with DualXDA data attribution
  - Stores features, labels, and metadata
- `EmbeddingDropoutMLP`: Small MLP for embedding classification

**Example Usage**:
```python
from uqlab.2_models.classification_models import EmbeddingDataset

# Create dataset from pre-extracted features
dataset = EmbeddingDataset(
    features=dinov2_embeddings,      # [N, 768]
    labels=training_labels,          # [N] (may be noisy)
    clean_labels=ground_truth,       # [N]
    is_noisy=noise_mask,            # [N]
    original_indices=indices         # [N]
)

# Use with DataLoader
loader = DataLoader(dataset, batch_size=256)
```

**Dependencies**:
```python
torch>=2.0.0
```

---

### [`feature_extractors.py`](./feature_extractors.py) - Feature Extraction

**Purpose**: Unified interface for extracting features from any model architecture.

**Key Components**:
- `FeatureExtractor`: Abstract base class
- `DINOv2Backbone`: Extract DINOv2 features
- `ResNetBackbone`: Extract ResNet features
- `SimpleCNNBackbone`: Extract CNN features
- Feature caching and organization utilities

**Example Usage**:
```python
from uqlab.2_models.feature_extractors import DINOv2Backbone

# Create feature extractor
extractor = DINOv2Backbone(
    model_name="dinov2_vits14",
    device="cuda"
)

# Extract features for entire dataset
features = extractor.extract_features(dataloader)  # [N, 384]
feature_dim = extractor.get_feature_dim()  # 384
```

**Supported Backbones**:
- **DINOv2**: `dinov2_vits14` (384-dim), `dinov2_vitb14` (768-dim), `dinov2_vitl14` (1024-dim)
- **ResNet**: ResNet18 (512-dim), ResNet34, ResNet50
- **Custom CNN**: Configurable dimensions

**Dependencies**:
```python
torch>=2.0.0
transformers>=4.30.0  # For DINOv2
torchvision>=0.15.0   # For ResNet
```

---

### [`uncertainty.py`](./uncertainty.py) - Uncertainty Quantification

**Purpose**: MC Dropout, Deep Ensembles, and uncertainty metric calculations.

**Key Components**:
- `enable_mc_dropout()`: Enable dropout during inference
- `disable_mc_dropout()`: Disable dropout (standard inference)
- `mc_forward_efficient()`: Efficient MC Dropout implementation
- Uncertainty metric calculations (entropy, mutual information)

**Example Usage**:
```python
from uqlab.2_models.uncertainty import enable_mc_dropout

# Standard inference (dropout disabled)
model.eval()
predictions = model(images)

# MC Dropout inference (dropout enabled)
model.eval()
enable_mc_dropout(model)
mc_predictions = []
for _ in range(20):
    mc_predictions.append(model(images))
mc_predictions = torch.stack(mc_predictions)

# Calculate uncertainty metrics
mean_pred = mc_predictions.mean(dim=0)
uncertainty = mc_predictions.std(dim=0)
```

**Dependencies**:
```python
torch>=2.0.0
```

---

## 🔧 Configuration Requirements

### Model Configuration

All models are configured via `ModelConfig` dataclass:

```python
@dataclass
class ModelConfig:
    architecture: str              # "dinov2_mlp", "cnn_mcdropout", "resnet18_mcdropout"
    training_mode: str            # "feature_space" or "end_to_end"
    dropout: float = 0.3          # Dropout probability for MC Dropout
    hidden_dim: int = 256         # Hidden dimension for MLP heads
    use_untrained_resnet: bool = False  # Train ResNet from scratch (no ImageNet)
```

### Hardware Requirements

**Minimum**:
- GPU: 8GB VRAM (for feature-space mode)
- RAM: 16GB
- Storage: 10GB for model weights and features

**Recommended**:
- GPU: 16GB+ VRAM (for end-to-end mode)
- RAM: 32GB
- Storage: 50GB for experiments and caching

### Software Requirements

```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Optional (for specific features)
timm>=0.9.0              # Additional vision models
scikit-learn>=1.3.0      # Metrics and utilities
```

---

## 🚀 Common Workflows

### 1. Feature-Space Training (Recommended for Data Attribution)

```python
# Step 1: Extract features
from uqlab.2_models.feature_extractors import DINOv2Backbone

extractor = DINOv2Backbone("dinov2_vits14", device="cuda")
features = extractor.extract_features(train_loader)

# Step 2: Create embedding dataset
from uqlab.2_models.classification_models import EmbeddingDataset

dataset = EmbeddingDataset(features, labels, clean_labels, is_noisy, indices)

# Step 3: Train classifier
from uqlab.2_models.factory import build_model

config = ModelConfig(
    architecture="dinov2_mlp",
    training_mode="feature_space",
    dropout=0.3
)
model = build_model(config, num_classes=10, feature_dim=384)

# Step 4: Train with MC Dropout
for epoch in range(epochs):
    for batch_features, batch_labels in dataloader:
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        # ... training loop
```

### 2. End-to-End Training

```python
# Step 1: Create model
config = ModelConfig(
    architecture="resnet18_mcdropout",
    training_mode="end_to_end",
    dropout=0.3,
    use_untrained_resnet=False  # Use ImageNet pretrained
)
model = build_model(config, num_classes=10)

# Step 2: Train on images directly
for epoch in range(epochs):
    for images, labels in dataloader:
        logits = model(images)
        loss = criterion(logits, labels)
        # ... training loop
```

### 3. MC Dropout Inference

```python
# After training, use MC Dropout for uncertainty
model.eval()
mc_predictions = model.mc_forward(
    images,
    n_passes=20,              # Number of MC samples
    sample_batch_size=256     # Batch size for inference
)

# mc_predictions shape: [n_passes, batch_size, num_classes]
mean_pred = mc_predictions.mean(dim=0)
uncertainty = mc_predictions.std(dim=0)
```

---

## 🔍 Architecture Comparison

| Feature | DINOv2 MLP | CNN MC Dropout | ResNet18 MC Dropout |
|---------|-----------|----------------|---------------------|
| **Training Mode** | Feature-space only | End-to-end only | Both modes ⭐ |
| **Parameters** | ~260K (MLP only) | ~11M | ~11M |
| **Training Speed** | ⚡ Fast | 🐢 Slow | 🏃 Medium |
| **Memory Usage** | 💚 Low | 🔴 High | 🟡 Medium |
| **Pretrained** | ✅ DINOv2 | ❌ Random init | ✅ ImageNet (optional) |
| **Data Attribution** | ✅ Compatible | ❌ Not compatible | ✅ Compatible (feature-space) |
| **Best For** | Quick experiments, attribution | Custom architectures | Balanced performance |

---

## 📚 Additional Resources

- **DINOv2 Paper**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **MC Dropout Paper**: [Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## 🐛 Troubleshooting

### Issue: "ResNet feature extractor requires a pre-initialized model"

**Solution**: This was a UI configuration bug (fixed 2026-06-15). Ensure you're using the latest version where ResNet properly supports both training modes.

### Issue: Out of memory during end-to-end training

**Solution**: 
1. Reduce batch size
2. Use feature-space mode instead
3. Use gradient checkpointing
4. Use mixed precision training (fp16)

### Issue: MC Dropout not working

**Solution**: Ensure dropout is enabled during inference:
```python
model.eval()  # Set to eval mode
model.enable_dropout()  # Re-enable dropout layers
```

---

## 📝 Version History

- **2026-06-15**: Added ResNet feature-space mode support, updated documentation
- **2026-06-01**: Initial module organization and factory pattern
- **2026-05-15**: Added MC Dropout utilities and uncertainty quantification

---

**Maintained by**: UQLab Team  
**Last Updated**: 2026-06-15