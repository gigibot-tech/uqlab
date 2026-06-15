# UQ Benchmarks Package

A new package for uncertainty quantification benchmarking that integrates formal disentanglement methods from research with production infrastructure.

## Overview

This package bridges the gap between research (`uq_disentanglement_comparison`) and production (`uq_classification`) by providing:

- **Multiple UQ Methods**: Gaussian Logits, Information-Theoretic, DualXDA
- **Formal Benchmarks**: Label noise, dataset size, OOD detection
- **Clean Abstractions**: Dataset format compatible with both Keras and PyTorch
- **Production Ready**: Integrates with FastAPI backend and Streamlit UI

## Package Structure

```
uq_benchmarks/
├── __init__.py              # Package exports
├── datatypes.py             # Core data structures (Dataset, UncertaintyResults)
├── data/                    # Dataset loaders
│   ├── __init__.py
│   ├── cifar10.py          # CIFAR-10 with noise injection & epistemic manipulation
│   ├── fashion_mnist.py    # TODO: Fashion-MNIST loader
│   └── blobs.py            # TODO: Toy dataset for testing
├── models/                  # UQ method implementations
│   ├── __init__.py
│   ├── base.py             # TODO: Abstract UQMethod interface
│   ├── gaussian_logits.py  # TODO: Two-head Gaussian model
│   ├── information_theoretic.py  # TODO: MI/EE/PE decomposition
│   └── dualxda_adapter.py  # TODO: Wrap existing DualXDA
├── benchmarks/              # Formal benchmark experiments
│   ├── __init__.py
│   ├── label_noise.py      # TODO: Label noise sweep
│   ├── dataset_size.py     # TODO: Dataset size sweep
│   └── ood_detection.py    # TODO: OOD class detection
└── utils/                   # Helper functions
    ├── __init__.py
    └── adapters.py         # TODO: Format translation utilities
```

## Core Data Types

### Dataset
```python
from uq_benchmarks import Dataset

dataset = Dataset(
    X_train=x_train,        # np.ndarray: Training features
    y_train=y_train,        # np.ndarray: Training labels
    X_test=x_test,          # np.ndarray: Test features
    y_test=y_test,          # np.ndarray: Test labels
    is_regression=False,    # bool: Classification or regression
    noise_mask=None,        # Optional[np.ndarray]: Boolean mask for noisy labels
    clean_labels=None,      # Optional[np.ndarray]: Original clean labels
    noise_rate=0.0          # float: Percentage of noisy labels
)
```

### UncertaintyResults
```python
from uq_benchmarks import UncertaintyResults

results = UncertaintyResults()
results.append_values(
    accuracy=0.85,
    aleatoric_uncertainty=0.42,
    epistemic_uncertainty=0.31,
    parameter=0.2  # e.g., noise_rate=0.2
)
```

## Usage Examples

### Load CIFAR-10 with Label Noise
```python
from uq_benchmarks.data.cifar10 import get_cifar10_dataset

# Standard CIFAR-10
dataset = get_cifar10_dataset()

# With 40% label noise
noisy_dataset = get_cifar10_dataset(noise_rate=0.4, seed=42)

# Test mode (smaller dataset for quick testing)
test_dataset = get_cifar10_dataset(test_mode=True)
```

### Create Epistemic Uncertainty Manipulation
```python
from uq_benchmarks.data.cifar10 import get_cifar10_with_epistemic_manipulation

# Under-support classes 3 and 5 (cats and dogs)
dataset = get_cifar10_with_epistemic_manipulation(
    under_supported_classes=[3, 5],
    under_train_per_class=50,      # Few samples for cats/dogs
    regular_train_per_class=300,   # Normal samples for other classes
    eval_per_class=100,            # Balanced test set
    noise_rate=0.2,                # Add 20% label noise (aleatoric)
    seed=42
)
```

## Integration with Existing System

### Backend Integration
The package is designed to integrate with the existing FastAPI backend:

```python
# backend/app/api/routes/benchmarks.py
from uq_benchmarks.data.cifar10 import get_cifar10_dataset
from uq_benchmarks.benchmarks.label_noise import run_label_noise_benchmark

@router.post("/label-noise")
async def run_label_noise(config: BenchmarkConfig):
    dataset = get_cifar10_dataset()
    results = run_label_noise_benchmark(dataset, config)
    # Store results in PostgreSQL
    return results
```

### Streamlit Integration
```python
# streamlit_app.py
from uq_benchmarks import Dataset, UncertaintyResults
from uq_benchmarks.data.cifar10 import get_cifar10_dataset

dataset = get_cifar10_dataset(noise_rate=st.slider("Noise Rate", 0.0, 1.0, 0.2))
st.write(f"Training samples: {len(dataset.X_train)}")
st.write(f"Noise rate: {dataset.noise_rate:.1%}")
```

## Design Principles

### 1. Clean Separation
- **New package** (`uq_benchmarks`) separate from existing code
- No modifications to `uq_classification` or `backend/app`
- Can be developed and tested independently

### 2. Framework Agnostic
- Dataset format works with both Keras and PyTorch
- NumPy arrays as common ground
- Easy conversion to framework-specific formats

### 3. Backward Compatible
- Existing experiments continue to work
- New features added via new endpoints
- Database changes are additive only

### 4. Research to Production
- Formal methods from `uq_disentanglement_comparison`
- Production infrastructure from `uq_classification`
- Best of both worlds

## Development Status

### ✅ Completed (Phase 1)
- [x] Package structure created
- [x] Core datatypes defined
- [x] CIFAR-10 loader with noise injection
- [x] CIFAR-10 loader with epistemic manipulation
- [x] Package documentation

### 🚧 In Progress
- [ ] UQ method implementations (Gaussian Logits, IT)
- [ ] Benchmark framework (label noise, dataset size, OOD)
- [ ] Backend API integration
- [ ] Streamlit UI components

### 📋 Planned
- [ ] Additional datasets (Fashion-MNIST, blobs)
- [ ] DualXDA adapter
- [ ] Comprehensive testing
- [ ] Performance optimization

## Dependencies

### Required
- `numpy`: Core array operations
- `keras`: Dataset loading (CIFAR-10, Fashion-MNIST)

### Optional
- `tensorflow`: For Gaussian Logits and IT methods
- `torch`: For DualXDA method
- `scikit-learn`: For metrics and utilities

## Installation

```bash
# From walaris-cen directory
pip install -e uq_benchmarks/

# Or with dependencies
pip install -e uq_benchmarks/[keras]  # For Keras-based methods
pip install -e uq_benchmarks/[torch]  # For PyTorch-based methods
pip install -e uq_benchmarks/[all]    # All dependencies
```

## Testing

```bash
# Run tests
pytest uq_benchmarks/tests/

# Test data loading
python -c "from uq_benchmarks.data.cifar10 import get_cifar10_dataset; print(get_cifar10_dataset(test_mode=True))"
```

## Visualization

The package includes visualization tools for comparing research methods with production signals.

### Creating Comparison Plots

```python
from uq_benchmarks.visualization import plot_benchmark_comparison_grid

# Research method results (from Gaussian Logits, IT, etc.)
research_results = {
    'gaussian_logits': {
        'epistemic': [0.05, 0.06, 0.07, 0.09, 0.12],
        'aleatoric': [0.02, 0.05, 0.10, 0.15, 0.20],
        'accuracy': [0.95, 0.92, 0.88, 0.82, 0.75]
    }
}

# Production signals (your existing 7 signals)
production_signals = {
    'msp_uncertainty': [...],
    'predictive_entropy': [...],
    'mutual_info': [...],
    'inverse_coherence': [...],  # Aleatoric indicator
    'dominance': [...],  # Epistemic indicator
    'inverse_mass': [...],  # Best epistemic
    'inverse_logit_magnitude': [...],
    'accuracy': [...]
}

# Create subplot grid comparing all methods
fig = plot_benchmark_comparison_grid(
    research_results=research_results,
    production_signals=production_signals,
    parameter_values=[0.0, 0.1, 0.2, 0.3, 0.4],
    parameter_name='noise_rate',
    title='Label Noise Sweep: Research vs Production'
)

fig.savefig('comparison.png')
```

### Signal Correlation Analysis

```python
from uq_benchmarks.visualization import plot_signal_correlation_heatmap

# Analyze which signals are redundant vs unique
fig = plot_signal_correlation_heatmap(
    signals_data=production_signals,
    title='Production Signals Correlation Matrix'
)
```

### Example Script

See `examples/visualization_example.py` for a complete example showing:
- Label noise sweep comparison
- Dataset size sweep comparison
- Signal correlation analysis

Run it with:
```bash
cd uq_benchmarks
python examples/visualization_example.py
```

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### With Research Package Support

```bash
# Install research dependencies
pip install -r requirements-research.txt

# Install research package in editable mode
cd ../uq_disentanglement_comparison-72CC
pip install -e .
```


## Contributing

This package follows the architecture plan in `ARCHITECTURE_REWORK_PLAN.md`. Key principles:

1. **Additive changes only**: Never break existing functionality
2. **Clean abstractions**: Use adapters for format translation
3. **Type safety**: Full type hints for all public APIs
4. **Documentation**: Docstrings for all public functions
5. **Testing**: Unit tests for all components

## License

Same as parent project.

## References

- Research paper: "Measuring Uncertainty Disentanglement Error in Classification" (arXiv:2408.12175)
- Original package: `uq_disentanglement_comparison-72CC/`
- Production system: `uq_classification/`