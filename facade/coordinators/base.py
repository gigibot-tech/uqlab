"""
Base coordinator class for the Experiment Facade pattern.

Provides common functionality and interface for all coordinators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


class BaseCoordinator(ABC):
    """
    Abstract base class for all coordinators.
    
    Coordinators encapsulate domain-specific logic and provide clean interfaces
    for the ExperimentFacade to orchestrate complex ML workflows.
    
    Each coordinator is responsible for a specific domain:
    - DataCoordinator: Dataset loading and preprocessing
    - ModelCoordinator: Model creation and management
    - TrainingCoordinator: Training loop orchestration
    - EvaluationCoordinator: Evaluation and metrics calculation
    - ResultCoordinator: Result collection and storage
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the coordinator.
        
        Args:
            config: Configuration dictionary for this coordinator's domain
            logger: Optional logger instance. If None, creates a new logger.
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._state: Dict[str, Any] = {}
        
    @abstractmethod
    def setup(self) -> None:
        """
        Setup the coordinator's resources.
        
        This method is called once during initialization to prepare
        any resources needed by the coordinator.
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """
        Cleanup the coordinator's resources.
        
        This method is called when the coordinator is no longer needed
        to release any resources it holds.
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the coordinator.
        
        Returns:
            Dictionary containing the coordinator's current state
        """
        return self._state.copy()
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the coordinator's state.
        
        Args:
            state: Dictionary containing the new state
        """
        self._state = state.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the coordinator's configuration.
        
        Args:
            config: Dictionary containing configuration updates
        """
        self.config.update(config)
        self.logger.info(f"Configuration updated: {list(config.keys())}")

# Made with Bob
