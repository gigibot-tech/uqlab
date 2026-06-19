# Experiment Facade Pattern

This package implements the **Facade Pattern** (Gang of Four) to provide a simplified, high-level interface for running complete ML experiments.

## Architecture Overview

The facade coordinates five specialized **Coordinator** classes, each responsible for a specific domain:

```
┌─────────────────────────────────────────────────────────────┐
│                    ExperimentFacade                         │
│                  (Main Orchestrator)                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──► DataCoordinator
             │    └─ Dataset loading & preprocessing
             │
             ├──► ModelCoordinator
             │    └─ Model creation & management
             │
             ├──► TrainingCoordinator
             │    └─ Training loop execution
             │
             ├──► EvaluationCoordinator
             │    └─ Evaluation & uncertainty quantification
             │
             └──► ResultCoordinator
                  └─ Result collection & storage
```

## Design Principles

### 1. Separation of Concerns (SoC)
Each coordinator handles a single domain:
- **DataCoordinator**: Only data operations
- **ModelCoordinator**: Only model operations
- **TrainingCoordinator**: Only training operations
- **EvaluationCoordinator**: Only evaluation operations
- **ResultCoordinator**: Only result operations

### 2. Single Responsibility Principle (SRP)
Each class has one reason to change:
- Data format changes → DataCoordinator
- Model architecture changes → ModelCoordinator
- Training algorithm changes → TrainingCoordinator
- Evaluation metrics changes → EvaluationCoordinator
- Result storage changes → ResultCoordinator

### 3. Dependency Inversion Principle (DIP)
- High-level `ExperimentFacade` depends on abstractions (`BaseCoordinator`)
- Low-level coordinators implement the abstraction
- Easy to swap implementations without changing the facade

### 4. Open/Closed Principle (OCP)
- Open for extension: Add new coordinators by extending `BaseCoordinator`
- Closed for modification: Existing coordinators don't need changes

## Usage

### Basic Usage

```python
from uqlab.facade import ExperimentFacade

# Define experiment configuration
config = {
    "experiment_name": "cifar10n_dinov2_experiment",
    "dataset_name": "cifar10n",
    "noise_type": "worse_label",
    "model_type": "dinov2",
    "dinov2_model": "small",
    "epochs": 10,
    "learning_rate": 0.001,
    "mc_passes": 20,
}

# Run experiment
facade = ExperimentFacade(config)
results = facade.run_experiment()

print(f"Test Accuracy: {results['evaluation']['accuracy']:.2f}%")
```

### Advanced Usage with Manual Control

```python
from uqlab.facade import ExperimentFacade

config = {...}
facade = ExperimentFacade(config)

# Manual setup
facade.setup()

try:
    # Access individual coordinators
    data_stats = facade.data_coordinator.get_dataset_stats()
    model_info = facade.model_coordinator.get_model_info()
    
    # Run custom training logic
    # ... your custom code ...
    
    # Run evaluation
    results = facade._run_evaluation()
    
finally:
    # Always teardown
    facade.teardown()
```

### Using Individual Coordinators

```python
from uqlab.facade.coordinators import DataCoordinator, ModelCoordinator

# Use coordinators independently
data_config = {
    "dataset_name": "cifar10n",
    "batch_size": 256,
}

data_coordinator = DataCoordinator(data_config)
data_coordinator.setup()

train_loader = data_coordinator.get_train_loader()
# ... use train_loader ...

data_coordinator.teardown()
```

## Configuration

### Complete Configuration Example

```python
config = {
    # Experiment metadata
    "experiment_name": "my_experiment",
    "results_dir": "results",
    
    # Data configuration
    "dataset_name": "cifar10n",
    "noise_type": "worse_label",
    "under_supported": "random:2",  # or "0,1,2" for specific classes
    "under_train_per_class": 50,
    "regular_train_per_class": 300,
    "eval_per_group": 100,
    "train_batch_size": 256,
    "num_workers": 4,
    
    # Model configuration
    "model_type": "dinov2",
    "dinov2_model": "small",  # "small", "base", "large"
    "hidden_dim": 256,
    "dropout": 0.2,
    "num_classes": 10,
    "freeze_backbone": False,
    "use_untrained_resnet": False,
    "device": "cuda",  # or "cpu"
    
    # Training configuration
    "epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",  # or "sgd"
    "scheduler": None,  # or "step", "cosine"
    "early_stopping_patience": None,  # or integer
    "gradient_clip": None,  # or float
    "criterion": "cross_entropy",
    
    # Evaluation configuration
    "mc_passes": 20,
    "uncertainty_signals": ["predictive_entropy", "mutual_information"],
    "eval_batch_size": 256,
    "compute_auroc": True,
    
    # Result configuration
    "save_format": "json",
    "save_checkpoints": True,
}
```

## Coordinators

### DataCoordinator

**Responsibilities:**
- Load datasets (CIFAR-10N, etc.)
- Apply data augmentation
- Create train/val/test splits
- Manage data loaders
- Handle epistemic uncertainty (under-supported classes)
- Handle aleatoric uncertainty (label noise)

**Key Methods:**
- `setup()` - Load datasets and create data loaders
- `get_train_loader()` - Get training data loader
- `get_val_loader()` - Get validation data loader
- `get_test_loader()` - Get test data loader
- `get_dataset_stats()` - Get dataset statistics
- `teardown()` - Release resources

### ModelCoordinator

**Responsibilities:**
- Create and initialize models
- Manage model architecture
- Handle checkpointing and loading
- Configure MC Dropout
- Manage device placement (CPU/GPU)

**Key Methods:**
- `setup()` - Create and initialize model
- `get_model()` - Get the model instance
- `get_device()` - Get the device
- `load_checkpoint(path)` - Load model weights
- `save_checkpoint(path)` - Save model weights
- `set_train_mode()` - Set model to training mode
- `set_eval_mode()` - Set model to evaluation mode
- `freeze_backbone()` - Freeze feature extractor
- `teardown()` - Release resources

### TrainingCoordinator

**Responsibilities:**
- Manage training loop
- Configure optimizer and scheduler
- Handle loss computation
- Track training metrics
- Implement early stopping
- Support callbacks

**Key Methods:**
- `setup()` - Initialize training state
- `initialize_optimizer(model)` - Create optimizer
- `initialize_criterion()` - Create loss function
- `train_epoch(model, loader, device)` - Train one epoch
- `validate_epoch(model, loader, device)` - Validate one epoch
- `should_stop_early()` - Check early stopping
- `get_training_state()` - Get current training state
- `teardown()` - Release resources

### EvaluationCoordinator

**Responsibilities:**
- Perform model evaluation
- Calculate uncertainty metrics
- Compute MC Dropout predictions
- Calculate AUROC and other metrics
- Generate uncertainty signals
- Perform triage analysis

**Key Methods:**
- `setup()` - Initialize evaluation config
- `evaluate_model(model, loader, device)` - Evaluate model
- `compute_auroc_metrics(uncertainties, is_correct)` - Calculate AUROC
- `compute_dualxda_signals(...)` - Compute DualXDA signals
- `get_evaluation_summary()` - Get evaluation summary
- `teardown()` - Release resources

### ResultCoordinator

**Responsibilities:**
- Collect experiment results
- Save results to disk
- Load previous results
- Generate summaries and reports
- Manage result versioning

**Key Methods:**
- `setup()` - Create result directories
- `add_training_epoch_result(epoch, metrics)` - Record training result
- `add_evaluation_result(split, metrics)` - Record evaluation result
- `add_metadata(key, value)` - Add metadata
- `save_results(filename)` - Save results to disk
- `load_results(path)` - Load results from disk
- `get_best_epoch(metric, mode)` - Get best epoch
- `generate_summary()` - Generate result summary
- `export_to_csv(filename)` - Export to CSV
- `teardown()` - Save final results and cleanup

## Extending the Facade

### Adding a New Coordinator

1. Create a new coordinator class extending `BaseCoordinator`:

```python
from uqlab.facade.coordinators import BaseCoordinator

class CustomCoordinator(BaseCoordinator):
    def setup(self) -> None:
        # Initialize resources
        pass
    
    def teardown(self) -> None:
        # Cleanup resources
        pass
    
    def custom_method(self) -> Any:
        # Your custom logic
        pass
```

2. Add it to the facade:

```python
class ExtendedExperimentFacade(ExperimentFacade):
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.custom_coordinator = CustomCoordinator(
            self._extract_custom_config(),
            logger=self.logger.getChild("Custom")
        )
    
    def setup(self):
        super().setup()
        self.custom_coordinator.setup()
    
    def teardown(self):
        self.custom_coordinator.teardown()
        super().teardown()
```

### Customizing Experiment Workflow

Override `run_experiment()` to customize the workflow:

```python
class CustomExperimentFacade(ExperimentFacade):
    def run_experiment(self):
        if not self._is_setup:
            self.setup()
        
        try:
            # Custom workflow
            self._run_custom_phase_1()
            training_results = self._run_training()
            self._run_custom_phase_2()
            evaluation_results = self._run_evaluation()
            return self._collect_results(training_results, evaluation_results)
        finally:
            self.teardown()
    
    def _run_custom_phase_1(self):
        # Your custom logic
        pass
```

## Benefits of This Architecture

### 1. Reduced Complexity
- **Before**: 1,460-line monolithic script
- **After**: ~1,500 lines split across 7 focused classes
- Each class < 300 lines, focused on single responsibility

### 2. Improved Testability
- Test each coordinator independently
- Mock coordinators for integration tests
- Clear interfaces make testing easier

### 3. Better Maintainability
- Changes isolated to specific coordinators
- Easy to understand and modify
- Clear separation of concerns

### 4. Enhanced Reusability
- Use coordinators independently
- Compose different workflows
- Share coordinators across projects

### 5. Easier Debugging
- Clear responsibility boundaries
- Isolated error handling
- Better logging per coordinator

## Migration from Monolithic Script

### Before (Monolithic)

```python
# run_fast_uncertainty_classification.py (1,460 lines)
def main():
    # Load data (100 lines)
    # Create model (150 lines)
    # Train model (300 lines)
    # Evaluate model (400 lines)
    # Save results (100 lines)
    # ... everything mixed together
```

### After (Facade)

```python
from uqlab.facade import ExperimentFacade

config = {...}
facade = ExperimentFacade(config)
results = facade.run_experiment()
```

## Related Documentation

- [Dual Facade Architecture](../../../DUAL_FACADE_ARCHITECTURE.md) - Overall architecture plan
- [Component Reuse Analysis](../../../COMPONENT_REUSE_ANALYSIS.md) - Analysis that led to this design
- [Dead Code Archive](../../../dead_code/README.md) - Archived unused components

## Future Enhancements

### PHASE 3: Backend Facade Extension
- Create `BackendExperimentFacade` extending `ExperimentFacade`
- Add database integration
- Add API endpoint support
- Add async execution support

### PHASE 4: Integration
- Refactor `run_fast_uncertainty_classification.py` to use facade
- Update CLI scripts
- Update FastAPI routes
- Add comprehensive tests

### PHASE 5: Validation
- Verify all metrics improved
- Benchmark performance
- Validate code quality metrics
- User acceptance testing

## Version History

- **v0.1.0** (2026-06-19) - Initial implementation
  - Created 5 coordinator classes
  - Implemented ExperimentFacade
  - Added comprehensive documentation