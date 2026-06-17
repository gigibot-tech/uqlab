# Decision Boundary Visualization Module

## Overview

The `decision_boundary_viz.py` module provides comprehensive tools for visualizing decision boundaries of trained classification models. It supports both 2D data and high-dimensional data with automatic dimensionality reduction, and integrates seamlessly with PyTorch models and the ExperimentTracker system.

## Features

- **Flexible Model Support**: Works with PyTorch models and scikit-learn style models
- **Automatic Dimensionality Reduction**: Uses t-SNE or UMAP for high-dimensional data
- **Checkpoint Loading**: Load and visualize models from PyTorch checkpoint files
- **Batch Processing**: Generate visualizations for multiple checkpoints at once
- **Customizable Visualizations**: Control colors, resolution, figure size, and more
- **Train/Test Comparison**: Side-by-side visualization of training and test data

## Dependencies

### Required Dependencies

```bash
pip install numpy matplotlib
```

### Optional Dependencies

For full functionality, install these additional packages:

```bash
# For PyTorch model support
pip install torch

# For t-SNE dimensionality reduction
pip install scikit-learn

# For UMAP dimensionality reduction (recommended for large datasets)
pip install umap-learn
```

### Complete Installation

```bash
# Install all dependencies at once
pip install numpy matplotlib torch scikit-learn umap-learn
```

## Quick Start

### Basic Usage with 2D Data

```python
from uq_classification import plot_decision_boundary
import numpy as np

# Your trained model
model = train_your_model(X_train, y_train)

# Visualize decision boundary
fig = plot_decision_boundary(
    model, 
    X_train, y_train,
    X_test=X_test, y_test=y_test
)
plt.show()
```

### High-Dimensional Data with Dimensionality Reduction

```python
from uq_classification import plot_decision_boundary

# High-dimensional data (e.g., 100 features)
X_train = np.random.randn(500, 100)
y_train = np.random.randint(0, 3, 500)

# Automatically reduces to 2D using t-SNE
fig = plot_decision_boundary(
    model, 
    X_train, y_train,
    reduce_dims=True,
    reduction_method='tsne'  # or 'umap'
)
```

### Visualizing from Checkpoint

```python
from uq_classification import visualize_checkpoint

# Load and visualize a saved checkpoint
fig, save_path = visualize_checkpoint(
    checkpoint_path='checkpoints/model_epoch_10.pt',
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    save_dir='visualizations'
)
```

### Batch Checkpoint Visualization

```python
from uq_classification import visualize_checkpoints_batch

# Visualize all checkpoints in a directory
image_paths = visualize_checkpoints_batch(
    checkpoint_paths='checkpoints/',
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    save_dir='visualizations',
    pattern='*.pt'
)

print(f"Generated {len(image_paths)} visualizations")
```

## API Reference

### Core Functions

#### `plot_decision_boundary()`

Main visualization function for decision boundaries.

**Parameters:**
- `model`: Trained model with `predict()` or `forward()` method
- `X`: Training data (n_samples, n_features)
- `y`: Training labels (n_samples,)
- `X_test`: Optional test data
- `y_test`: Optional test labels
- `reduce_dims`: Whether to apply dimensionality reduction (default: True)
- `reduction_method`: 'tsne' or 'umap' (default: 'tsne')
- `resolution`: Grid resolution for boundary (default: 200)
- `figsize`: Figure size tuple (default: (12, 5))
- `cmap`: Colormap name (default: 'RdYlBu')
- `alpha`: Boundary transparency (default: 0.3)
- `title`: Optional plot title
- `save_path`: Optional path to save figure

**Returns:** matplotlib Figure object

#### `visualize_checkpoint()`

Load and visualize a model from checkpoint.

**Parameters:**
- `checkpoint_path`: Path to checkpoint file
- `X`: Training data
- `y`: Training labels
- `model`: Optional model instance to load weights into
- `X_test`: Optional test data
- `y_test`: Optional test labels
- `save_dir`: Optional directory to save visualization
- `**plot_kwargs`: Additional arguments for `plot_decision_boundary()`

**Returns:** Tuple of (figure, save_path)

#### `visualize_checkpoints_batch()`

Generate visualizations for multiple checkpoints.

**Parameters:**
- `checkpoint_paths`: List of paths or directory containing checkpoints
- `X`: Training data
- `y`: Training labels
- `model`: Optional model instance
- `X_test`: Optional test data
- `y_test`: Optional test labels
- `save_dir`: Directory to save visualizations (default: 'visualizations')
- `pattern`: File pattern for finding checkpoints (default: '*.pt')
- `**plot_kwargs`: Additional arguments for `plot_decision_boundary()`

**Returns:** List of paths to generated images

### Helper Functions

#### `reduce_dimensions()`

Reduce high-dimensional data to 2D.

**Parameters:**
- `X`: Input data (n_samples, n_features)
- `method`: 'tsne' or 'umap' (default: 'tsne')
- `n_components`: Number of dimensions (default: 2)
- `random_state`: Random seed (default: 42)
- `**kwargs`: Additional arguments for the reduction method

**Returns:** Reduced data (n_samples, n_components)

#### `create_meshgrid()`

Create evaluation grid for decision boundaries.

**Parameters:**
- `X`: 2D input data (n_samples, 2)
- `resolution`: Grid points per dimension (default: 100)
- `padding`: Padding around data bounds (default: 0.5)

**Returns:** Tuple of (xx, yy, grid_points)

#### `load_checkpoint()`

Load a PyTorch model from checkpoint.

**Parameters:**
- `checkpoint_path`: Path to checkpoint file
- `model`: Optional model instance to load weights into
- `device`: Device to load on (default: 'cpu')

**Returns:** Loaded model

## Advanced Usage

### Custom Colormap and Styling

```python
fig = plot_decision_boundary(
    model, X, y,
    cmap='viridis',
    alpha=0.5,
    figsize=(15, 6),
    title='Custom Decision Boundary'
)
```

### Using UMAP for Large Datasets

```python
# UMAP is faster and often better for large datasets
fig = plot_decision_boundary(
    model, X, y,
    reduction_method='umap',
    n_neighbors=15,  # UMAP parameter
    min_dist=0.1     # UMAP parameter
)
```

### Integration with ExperimentTracker

```python
from uq_classification import ExperimentTracker, visualize_checkpoint

# During training
tracker = ExperimentTracker(experiment_name='my_experiment')
tracker.start_run()

# Train model and save checkpoints
for epoch in range(10):
    train_model(model, epoch)
    checkpoint_path = f'checkpoints/model_epoch_{epoch}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    
    # Log checkpoint
    tracker.log_artifact(checkpoint_path)

tracker.end_run()

# After training, visualize all checkpoints
image_paths = visualize_checkpoints_batch(
    'checkpoints/',
    X_train, y_train,
    save_dir=f'visualizations/{tracker.run_id}'
)
```

### Handling Different Model Types

#### PyTorch Model

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)
    
    def forward(self, x):
        return self.fc(x)

model = MyModel()
# Train model...

fig = plot_decision_boundary(model, X, y)
```

#### Scikit-learn Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

fig = plot_decision_boundary(model, X_train, y_train)
```

## Error Handling

The module includes comprehensive error handling:

```python
try:
    fig = visualize_checkpoint('model.pt', X, y)
except FileNotFoundError:
    print("Checkpoint file not found")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Model loading failed: {e}")
```

## Performance Tips

1. **Use UMAP for large datasets**: UMAP is generally faster than t-SNE for datasets with >1000 samples
2. **Reduce resolution for quick previews**: Use `resolution=100` for faster rendering
3. **Close figures in batch processing**: The module automatically closes figures to save memory
4. **Use GPU for PyTorch models**: Models on GPU will be faster for prediction

## Troubleshooting

### "PyTorch not available" Warning

Install PyTorch:
```bash
pip install torch
```

### "t-SNE requires scikit-learn" Error

Install scikit-learn:
```bash
pip install scikit-learn
```

### "UMAP requires umap-learn" Error

Install umap-learn:
```bash
pip install umap-learn
```

### Checkpoint Loading Fails

Ensure your checkpoint format is supported:
- Full model: `torch.save(model, 'model.pt')`
- State dict: `torch.save(model.state_dict(), 'model.pt')`
- With metadata: `torch.save({'state_dict': model.state_dict()}, 'model.pt')`

## Examples

See `example_usage.py` for more examples of using the visualization module with the ExperimentTracker.

## License

This module is part of the UQ Classification package.

---

Made with Bob