"""
Data Coordinator for the Experiment Facade pattern.

Handles dataset loading, preprocessing, and data pipeline management.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
import logging

from .base import BaseCoordinator


class DataCoordinator(BaseCoordinator):
    """
    Coordinates all data-related operations for experiments.
    
    Responsibilities:
    - Load and prepare datasets (CIFAR-10N, etc.)
    - Apply data augmentation and preprocessing
    - Create train/val/test splits
    - Manage data loaders with proper batching
    - Handle epistemic uncertainty (under-supported classes)
    - Handle aleatoric uncertainty (label noise)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Data Coordinator.
        
        Args:
            config: Configuration dictionary containing:
                - dataset_name: Name of dataset (e.g., "cifar10n")
                - noise_type: Type of label noise (e.g., "worse_label")
                - under_supported: Under-supported class configuration
                - under_train_per_class: Samples per under-supported class
                - regular_train_per_class: Samples per regular class
                - eval_per_group: Evaluation samples per group
                - batch_size: Training batch size
                - num_workers: DataLoader workers
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None
        
    def setup(self) -> None:
        """
        Setup data resources.
        
        Loads datasets and creates data loaders based on configuration.
        """
        self.logger.info("Setting up Data Coordinator...")
        
        # Extract configuration
        dataset_name = self.config.get("dataset_name", "cifar10n")
        noise_type = self.config.get("noise_type", "worse_label")
        
        self.logger.info(f"Loading dataset: {dataset_name} with noise type: {noise_type}")
        
        # Load datasets (implementation will use existing CIFAR10NDataset)
        self._load_datasets()
        
        # Create data loaders
        self._create_data_loaders()
        
        # Store dataset statistics in state
        self._state["dataset_name"] = dataset_name
        self._state["noise_type"] = noise_type
        self._state["train_size"] = len(self.train_dataset) if self.train_dataset else 0
        self._state["val_size"] = len(self.val_dataset) if self.val_dataset else 0
        self._state["test_size"] = len(self.test_dataset) if self.test_dataset else 0
        
        self.logger.info(f"Data Coordinator setup complete. Train: {self._state['train_size']}, "
                        f"Val: {self._state['val_size']}, Test: {self._state['test_size']}")
    
    def teardown(self) -> None:
        """
        Cleanup data resources.
        
        Releases dataset and data loader references.
        """
        self.logger.info("Tearing down Data Coordinator...")
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._state.clear()
        self.logger.info("Data Coordinator teardown complete")
    
    def _load_datasets(self) -> None:
        """
        Load train, validation, and test datasets.
        
        This method will integrate with the existing CIFAR10NDataset class
        and handle epistemic/aleatoric uncertainty configuration.
        """
        # TODO: Integrate with existing uqlab.data.loaders.CIFAR10NDataset
        # For now, create placeholder datasets
        self.logger.warning("Dataset loading not yet implemented - using placeholders")
        
        # Placeholder implementation
        # In the full implementation, this will:
        # 1. Load CIFAR10NDataset with noise_type
        # 2. Apply under-supported class filtering
        # 3. Create train/val/test splits
        # 4. Apply data augmentation
        
        self.train_dataset = None  # Will be CIFAR10NDataset instance
        self.val_dataset = None
        self.test_dataset = None
    
    def _create_data_loaders(self) -> None:
        """
        Create PyTorch DataLoader instances for train, val, and test sets.
        """
        batch_size = self.config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)
        
        if self.train_dataset:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
    
    def get_train_loader(self) -> Optional[DataLoader]:
        """Get the training data loader."""
        return self.train_loader
    
    def get_val_loader(self) -> Optional[DataLoader]:
        """Get the validation data loader."""
        return self.val_loader
    
    def get_test_loader(self) -> Optional[DataLoader]:
        """Get the test data loader."""
        return self.test_loader
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics like sizes, class distribution, etc.
        """
        return {
            "train_size": self._state.get("train_size", 0),
            "val_size": self._state.get("val_size", 0),
            "test_size": self._state.get("test_size", 0),
            "dataset_name": self._state.get("dataset_name", "unknown"),
            "noise_type": self._state.get("noise_type", "unknown"),
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the class distribution in the training set.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        # TODO: Implement class distribution calculation
        return {}
    
    def apply_data_augmentation(self, augmentation_config: Dict[str, Any]) -> None:
        """
        Apply data augmentation to the training dataset.
        
        Args:
            augmentation_config: Configuration for data augmentation
        """
        self.logger.info(f"Applying data augmentation: {augmentation_config}")
        # TODO: Implement data augmentation
        pass

# Made with Bob
