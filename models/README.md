# UQLab Models

Trainable PyTorch architectures and the model factory for fast-pilot uncertainty classification.

## Directory map

```
models/
├── factory.py              # build_model(), SmallCNN, ResNetMC, mc_forward helpers
├── architecture.py         # Canonical architecture names + training scope
├── architectures.py        # Legacy composition helpers (ClassificationModel, etc.)
├── classification_models.py # EmbeddingMLP / EmbeddingDropoutMLP for feature-space runs
├── feature_extractors.py   # DINOv2 embedding extraction
├── training_scope.py       # End-to-end vs feature-space training flags
├── mc_dropout.py           # MC Dropout forward passes (inference only)
├── training.py             # train_feature_model / train_image_model (paper fit)
└── backbones/              # DINOv2 weight loading
```

**Not here:** uncertainty metrics and `signal_table` live in [`evaluation/signals/`](../evaluation/signals/README.md) and [`evaluation/artifacts.py`](../evaluation/artifacts.py); the paper bridge lives in [`evaluation/benchmarks/disentangling/`](../evaluation/benchmarks/disentangling/). Flow → [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md).

## Quick start — `build_model()`

```python
from uqlab.models.factory import build_model
from uqlab.shared.config.classification import ModelConfig

# Feature space: frozen DINOv2 embeddings + MLP head
config = ModelConfig(architecture="dinov2_mlp", hidden_dim=256, dropout=0.2)
model = build_model(config, num_classes=10, feature_dim=768)

logits = model(embeddings)  # [B, C]
mc_probs = model.mc_forward(embeddings, n_passes=20)  # [T, B, C]
```

```python
# End-to-end: ResNet18 with MC Dropout
config = ModelConfig(architecture="resnet18_mcdropout", dropout=0.3)
model = build_model(config, num_classes=10)

logits = model(images)  # [B, 3, 32, 32] → [B, C]
features = model.extract_features(images)  # for attribution
mc_probs = model.mc_forward(images, n_passes=20)
```

Supported architectures (via `normalize_architecture`):

| Name | Mode | Notes |
|------|------|-------|
| `dinov2_mlp` | Feature space | Requires precomputed embeddings + `feature_dim` |
| `cnn_mcdropout` | End-to-end | Small CNN, CIFAR-scale |
| `resnet18_mcdropout` | End-to-end | torchvision ResNet18 backbone |

## MC Dropout inference

[`mc_dropout.py`](mc_dropout.py) provides efficient batched forwards:

- `mc_forward_efficient(model, x, n_passes)` — chunks large eval sets, reuses CNN/ResNet backbone features
- `batch_mc_dropout_uncertainty(model, dataloader, ...)` — full-dataset helper

Entropy, mutual information, and SIRC scores are computed in [`evaluation/signals/mc_dropout.py`](../evaluation/signals/mc_dropout.py) from stacked softmax tensors.

## Feature extractors

[`feature_extractors.py`](feature_extractors.py) wraps DINOv2 for embedding extraction used by the fast-pilot loader and feature-space training path. See [`backbones/`](backbones/) for weight loading.

## Training scope

[`training_scope.py`](training_scope.py) and [`architecture.py`](architecture.py) map user-facing architecture strings to end-to-end vs feature-space training. The runner and UI pass these through `ModelConfig`.

## Related docs

- **Uncertainty signals & AUROC** → [`evaluation/signals/README.md`](../evaluation/signals/README.md)
- **Paper disentanglement benchmark** → [`docs/features/disentanglement-benchmark.md`](../../../docs/features/disentanglement-benchmark.md)
- **UQ flow** → [`docs/UQLAB_FLOW.md`](../../../docs/UQLAB_FLOW.md)
- **Vendored metric loops** → [`vendor/disentanglement_error/UPSTREAM.md`](../vendor/disentanglement_error/UPSTREAM.md)
- **Evaluation pipeline** → [`evaluation/README.md`](../evaluation/README.md)

## Archived

Duplicate MC Dropout utilities previously in `models/uncertainty.py` are archived at `dead_code/models/uncertainty.py` (zero production callers).
