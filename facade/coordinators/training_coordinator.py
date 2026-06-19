"""
Training Coordinator for the Experiment Facade pattern.

Handles model training, optimization, and training loop management.
"""

from typing import Any, Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

from .base import BaseCoordinator


class TrainingCoordinator(BaseCoordinator):
    """
    Coordinates all training-related operations for experiments.
    
    Responsibilities:
    - Manage training loop execution
    - Configure optimizer and learning rate scheduler
    - Handle loss computation and backpropagation
    - Track training metrics (loss, accuracy, etc.)
    - Implement early stopping and checkpointing
    - Support callbacks for custom training logic
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Training Coordinator.
        
        Args:
            config: Configuration dictionary containing:
                - epochs: Number of training epochs
                - learning_rate: Initial learning rate
                - weight_decay: L2 regularization weight
                - optimizer: Optimizer type ("adam", "sgd", etc.)
                - scheduler: LR scheduler type (optional)
                - early_stopping_patience: Patience for early stopping (optional)
                - gradient_clip: Gradient clipping value (optional)
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.current_epoch: int = 0
        self.best_val_loss: float = float('inf')
        self.epochs_without_improvement: int = 0
        
    def setup(self) -> None:
        """
        Setup training resources.
        
        Initializes optimizer, scheduler, and loss criterion.
        """
        self.logger.info("Setting up Training Coordinator...")
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Store config in state
        self._state["epochs"] = self.config.get("epochs", 10)
        self._state["learning_rate"] = self.config.get("learning_rate", 0.001)
        self._state["current_epoch"] = 0
        self._state["best_val_loss"] = float('inf')
        
        self.logger.info(f"Training Coordinator setup complete. "
                        f"Epochs: {self._state['epochs']}, "
                        f"LR: {self._state['learning_rate']}")
    
    def teardown(self) -> None:
        """
        Cleanup training resources.
        
        Releases optimizer and scheduler references.
        """
        self.logger.info("Tearing down Training Coordinator...")
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self._state.clear()
        self.logger.info("Training Coordinator teardown complete")
    
    def initialize_optimizer(self, model: nn.Module) -> None:
        """
        Initialize the optimizer for the given model.
        
        Args:
            model: The model to optimize
        """
        optimizer_type = self.config.get("optimizer", "adam").lower()
        learning_rate = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0001)
        
        self.logger.info(f"Initializing {optimizer_type} optimizer with LR={learning_rate}")
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Initialize scheduler if configured
        scheduler_type = self.config.get("scheduler")
        if scheduler_type:
            self._initialize_scheduler()
    
    def _initialize_scheduler(self) -> None:
        """Initialize the learning rate scheduler."""
        if self.optimizer is None:
            raise ValueError("Optimizer must be initialized before scheduler")
        
        scheduler_type = self.config.get("scheduler", "").lower()
        
        if scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("scheduler_step_size", 10),
                gamma=self.config.get("scheduler_gamma", 0.1)
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get("epochs", 10)
            )
        
        if self.scheduler:
            self.logger.info(f"Initialized {scheduler_type} scheduler")
    
    def initialize_criterion(self) -> None:
        """Initialize the loss criterion."""
        criterion_type = self.config.get("criterion", "cross_entropy").lower()
        
        if criterion_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion: {criterion_type}")
        
        self.logger.info(f"Initialized {criterion_type} criterion")
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            device: Device to train on
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary containing training metrics (loss, accuracy, etc.)
        """
        if self.optimizer is None or self.criterion is None:
            raise ValueError("Optimizer and criterion must be initialized")
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["gradient_clip"]
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Progress callback
            if progress_callback:
                progress_callback(batch_idx, len(train_loader), loss.item())
        
        # Update scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        self.current_epoch += 1
        self._state["current_epoch"] = self.current_epoch
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
    
    def validate_epoch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Validate the model for one epoch.
        
        Args:
            model: The model to validate
            val_loader: Validation data loader
            device: Device to validate on
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.criterion is None:
            raise ValueError("Criterion must be initialized")
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        # Update best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.epochs_without_improvement = 0
            self._state["best_val_loss"] = self.best_val_loss
        else:
            self.epochs_without_improvement += 1
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def should_stop_early(self) -> bool:
        """
        Check if training should stop early based on validation performance.
        
        Returns:
            True if training should stop, False otherwise
        """
        patience = self.config.get("early_stopping_patience")
        if patience is None:
            return False
        
        return self.epochs_without_improvement >= patience
    
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get the current training state.
        
        Returns:
            Dictionary containing training state information
        """
        return {
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "learning_rate": self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0,
        }

# Made with Bob
