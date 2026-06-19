"""
Facade Pattern Implementation for ML Experiments.

This package provides a simplified, high-level interface for running complete
ML experiments by coordinating specialized domain coordinators.

Main Classes:
    - ExperimentFacade: Main orchestration class
    - DataCoordinator: Dataset loading and preprocessing
    - ModelCoordinator: Model creation and management
    - TrainingCoordinator: Training loop execution
    - EvaluationCoordinator: Evaluation and uncertainty quantification
    - ResultCoordinator: Result collection and storage

Usage:
    ```python
    from uqlab.facade import ExperimentFacade
    
    config = {
        "experiment_name": "my_experiment",
        "dataset_name": "cifar10n",
        "model_type": "dinov2",
        "epochs": 10,
        # ... other config
    }
    
    facade = ExperimentFacade(config)
    results = facade.run_experiment()
    ```
"""

from .experiment_facade import ExperimentFacade
from .backend_experiment_facade import BackendExperimentFacade
from .coordinators import (
    BaseCoordinator,
    DataCoordinator,
    ModelCoordinator,
    TrainingCoordinator,
    EvaluationCoordinator,
    ResultCoordinator,
)

__all__ = [
    "ExperimentFacade",
    "BackendExperimentFacade",
    "BaseCoordinator",
    "DataCoordinator",
    "ModelCoordinator",
    "TrainingCoordinator",
    "EvaluationCoordinator",
    "ResultCoordinator",
]

__version__ = "0.1.0"

# Made with Bob
