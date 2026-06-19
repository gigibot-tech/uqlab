"""
Coordinator classes for the Experiment Facade pattern.

These coordinators encapsulate domain-specific logic and provide clean interfaces
for the ExperimentFacade to orchestrate complex ML workflows.
"""

from .base import BaseCoordinator
from .data_coordinator import DataCoordinator
from .model_coordinator import ModelCoordinator
from .training_coordinator import TrainingCoordinator
from .evaluation_coordinator import EvaluationCoordinator
from .result_coordinator import ResultCoordinator

__all__ = [
    "BaseCoordinator",
    "DataCoordinator",
    "ModelCoordinator",
    "TrainingCoordinator",
    "EvaluationCoordinator",
    "ResultCoordinator",
]

# Made with Bob
