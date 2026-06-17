"""
Abstract base class for UQ methods.
All uncertainty quantification methods must implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np

from ..datatypes import Dataset, UncertaintyResults


class UQMethod(ABC):
    """
    Abstract base class for uncertainty quantification methods.
    
    All UQ methods (Gaussian Logits, Information-Theoretic, DualXDA) 
    must implement this interface to ensure consistency.
    """
    
    def __init__(self, name: str, framework: str = "keras"):
        """
        Initialize UQ method.
        
        Args:
            name: Method name (e.g., 'gaussian_logits', 'information_theoretic', 'dualxda')
            framework: Framework used ('keras' or 'pytorch')
        """
        self.name = name
        self.framework = framework
    
    @abstractmethod
    def train(self, dataset: Dataset, config: Dict[str, Any]) -> Any:
        """
        Train the model on the given dataset.
        
        Args:
            dataset: Training and test data
            config: Training configuration (epochs, learning_rate, etc.)
            
        Returns:
            Trained model (framework-specific)
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, dataset: Dataset, config: Dict[str, Any]) -> Tuple[float, float, float]:
        """
        Evaluate the model and compute uncertainties.
        
        Args:
            model: Trained model
            dataset: Test data
            config: Evaluation configuration (num_samples, batch_size, etc.)
            
        Returns:
            Tuple of (accuracy, aleatoric_uncertainty, epistemic_uncertainty)
        """
        pass
    
    def train_and_evaluate(
        self, 
        dataset: Dataset, 
        train_config: Dict[str, Any],
        eval_config: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Convenience method to train and evaluate in one call.
        
        Args:
            dataset: Training and test data
            train_config: Training configuration
            eval_config: Evaluation configuration
            
        Returns:
            Tuple of (accuracy, aleatoric_uncertainty, epistemic_uncertainty)
        """
        model = self.train(dataset, train_config)
        return self.evaluate(model, dataset, eval_config)
    
    def run_benchmark(
        self,
        dataset_generator,
        parameter_values: list,
        train_config: Dict[str, Any],
        eval_config: Dict[str, Any]
    ) -> UncertaintyResults:
        """
        Run a benchmark experiment by varying a parameter.
        
        Args:
            dataset_generator: Function that takes a parameter value and returns a Dataset
            parameter_values: List of parameter values to sweep (e.g., noise rates)
            train_config: Training configuration
            eval_config: Evaluation configuration
            
        Returns:
            UncertaintyResults with results for each parameter value
        """
        results = UncertaintyResults()
        
        for param_value in parameter_values:
            # Generate dataset for this parameter value
            dataset = dataset_generator(param_value)
            
            # Train and evaluate
            accuracy, aleatoric, epistemic = self.train_and_evaluate(
                dataset, train_config, eval_config
            )
            
            # Store results
            results.append_values(accuracy, aleatoric, epistemic, param_value)
        
        return results
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', framework='{self.framework}')"


class KerasUQMethod(UQMethod):
    """Base class for Keras-based UQ methods."""
    
    def __init__(self, name: str):
        super().__init__(name, framework="keras")
    
    def _check_keras_available(self) -> bool:
        """Check if Keras is available."""
        try:
            import keras
            import tensorflow as tf
            return True
        except ImportError:
            raise ImportError(
                f"{self.name} requires Keras and TensorFlow. "
                "Install with: pip install keras tensorflow"
            )


class PyTorchUQMethod(UQMethod):
    """Base class for PyTorch-based UQ methods."""
    
    def __init__(self, name: str):
        super().__init__(name, framework="pytorch")
    
    def _check_pytorch_available(self) -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            raise ImportError(
                f"{self.name} requires PyTorch. "
                "Install with: pip install torch torchvision"
            )

# Made with Bob
