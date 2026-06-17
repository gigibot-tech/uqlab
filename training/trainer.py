"""
Uncertainty Trainer - Main training loop for uncertainty quantification models.

This module provides:
- Training loop with validation
- Optimizer and scheduler management
- Mixed precision training
- Gradient accumulation
- Callback integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader

from shared.types import MetricsDict
from shared.utils import get_device, get_logger, set_seed

from .callbacks import CallbackList, Callback
from .config import TrainingConfig


class UncertaintyTrainer:
    """
    Trainer for uncertainty quantification models.
    
    Handles:
    - Training loop with validation
    - Optimizer and scheduler setup
    - Mixed precision training
    - Gradient accumulation
    - Callback management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        callbacks: Optional[list[Callback]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            callbacks: List of callbacks
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Setup callbacks
        self.callbacks = CallbackList(callbacks or [])
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        
        # Logger
        self.logger = get_logger()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        opt_config = self.config.optimizer
        
        # Get model parameters
        params = self.model.parameters()
        
        # Create optimizer
        if opt_config.optimizer_type == "adam":
            optimizer = Adam(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
                amsgrad=opt_config.amsgrad,
            )
        elif opt_config.optimizer_type == "adamw":
            optimizer = AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay,
                amsgrad=opt_config.amsgrad,
            )
        elif opt_config.optimizer_type == "sgd":
            optimizer = SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
                nesterov=opt_config.nesterov,
            )
        elif opt_config.optimizer_type == "rmsprop":
            optimizer = RMSprop(
                params,
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler from configuration."""
        sched_config = self.config.scheduler
        
        if sched_config.scheduler_type == "none":
            return None
        
        if sched_config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.t_max or self.config.epochs,
                eta_min=sched_config.eta_min,
            )
        elif sched_config.scheduler_type == "step":
            if sched_config.milestones:
                from torch.optim.lr_scheduler import MultiStepLR
                scheduler = MultiStepLR(
                    self.optimizer,
                    milestones=sched_config.milestones,
                    gamma=sched_config.gamma,
                )
            else:
                scheduler = StepLR(
                    self.optimizer,
                    step_size=sched_config.step_size,
                    gamma=sched_config.gamma,
                )
        elif sched_config.scheduler_type == "exponential":
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=sched_config.decay_rate,
            )
        elif sched_config.scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.mode,
                factor=sched_config.factor,
                patience=sched_config.patience,
                threshold=sched_config.threshold,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_config.scheduler_type}")
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Callback: batch begin
            self.callbacks.on_batch_begin(batch_idx, self)
            
            # Move batch to device
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast(dtype=torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = nn.functional.cross_entropy(outputs, targets)
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # Gradient clipping
                if self.config.optimizer.max_grad_norm is not None:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.optimizer.max_grad_norm,
                    )
                
                # Optimizer step
                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Calculate metrics
            with torch.no_grad():
                predictions = outputs.argmax(dim=1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.size(0)
                total_loss += loss.item() * self.config.accumulation_steps * targets.size(0)
            
            # Callback: batch end
            self.callbacks.on_batch_end(batch_idx, loss.item() * self.config.accumulation_steps, self)
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.callbacks.on_validation_begin(self)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.val_loader:
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            # Calculate metrics
            predictions = outputs.argmax(dim=1)
            correct = (predictions == targets).sum().item()
            
            total_loss += loss.item() * targets.size(0)
            total_correct += correct
            total_samples += targets.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        metrics = {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }
        
        self.callbacks.on_validation_end(metrics, self)
        
        return metrics
    
    def train(self) -> Dict[str, list]:
        """
        Run full training loop.
        
        Returns:
            Training history
        """
        self.logger.info("Starting training...")
        self.callbacks.on_train_begin(self)
        
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        
        # Validation before training
        if self.config.val_before_training and self.val_loader is not None:
            val_metrics = self.validate()
            self.logger.info(f"Initial validation: {val_metrics}")
        
        # Training loop
        for epoch in range(self.config.epochs):
            if self.should_stop:
                self.logger.info(f"Training stopped early at epoch {epoch}")
                break
            
            self.current_epoch = epoch
            self.callbacks.on_epoch_begin(epoch, self)
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if (epoch + 1) % self.config.val_every_n_epochs == 0 and self.val_loader is not None:
                val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_metrics['learning_rate'] = current_lr
            
            # Update history
            for key, value in epoch_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Plateau scheduler needs a metric
                    metric_name = self.config.scheduler.mode
                    if "val_loss" in epoch_metrics:
                        self.scheduler.step(epoch_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Callbacks
            self.callbacks.on_epoch_end(epoch, epoch_metrics, self)
        
        self.callbacks.on_train_end(self)
        self.logger.info("Training completed!")
        
        return history
    
    def save_checkpoint(self, path: Path, **extra_data) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            **extra_data: Additional data to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            **extra_data,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path, load_optimizer: bool = True, load_scheduler: bool = True) -> Dict:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if load_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Update state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        self.logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint


# Made with Bob