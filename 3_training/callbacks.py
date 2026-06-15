"""
Training Callbacks - Modular callbacks for training loop.

This module provides callbacks for:
- Checkpointing
- Logging
- Early stopping
- Progress tracking
- Custom metrics
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from shared.types import MetricsDict
from shared.utils import format_number, format_time, get_logger, save_json


class Callback:
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, trainer: Any) -> None:
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, metrics: MetricsDict, trainer: Any) -> None:
        """Called at the end of validation."""
        pass


class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        save_dir: Path,
        save_every_n_epochs: int = 5,
        save_last: bool = True,
        save_best: bool = True,
        monitor_metric: str = "val_accuracy",
        monitor_mode: str = "max",
        keep_last_n: int = 3,
        keep_best_n: int = 1,
    ):
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_last = save_last
        self.save_best = save_best
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Track best metric
        self.best_metric = float('-inf') if monitor_mode == 'max' else float('inf')
        self.best_epoch = -1
        
        # Track saved checkpoints
        self.saved_checkpoints: List[Path] = []
        self.best_checkpoints: List[tuple[float, Path]] = []
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.monitor_mode == 'max':
            return current > best
        return current < best
    
    def _save_checkpoint(
        self,
        trainer: Any,
        epoch: int,
        metrics: MetricsDict,
        checkpoint_type: str = "regular",
    ) -> Path:
        """Save a checkpoint."""
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
        
        # Determine filename
        if checkpoint_type == "best":
            filename = f"best_epoch{epoch:03d}_{self.monitor_metric}{metrics.get(self.monitor_metric, 0):.4f}.pt"
        elif checkpoint_type == "last":
            filename = "last.pt"
        else:
            filename = f"epoch{epoch:03d}.pt"
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to keep only the most recent."""
        if len(self.saved_checkpoints) > self.keep_last_n:
            # Remove oldest checkpoints
            to_remove = self.saved_checkpoints[:-self.keep_last_n]
            for path in to_remove:
                if path.exists() and path.name != "last.pt":
                    path.unlink()
                    self.logger.debug(f"Removed old checkpoint: {path}")
            self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n:]
        
        # Cleanup best checkpoints
        if len(self.best_checkpoints) > self.keep_best_n:
            self.best_checkpoints.sort(key=lambda x: x[0], reverse=(self.monitor_mode == 'max'))
            to_remove = self.best_checkpoints[self.keep_best_n:]
            for _, path in to_remove:
                if path.exists():
                    path.unlink()
                    self.logger.debug(f"Removed old best checkpoint: {path}")
            self.best_checkpoints = self.best_checkpoints[:self.keep_best_n]
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Save checkpoints at the end of each epoch."""
        # Save regular checkpoint
        if (epoch + 1) % self.save_every_n_epochs == 0:
            checkpoint_path = self._save_checkpoint(trainer, epoch, metrics, "regular")
            self.saved_checkpoints.append(checkpoint_path)
            self._cleanup_checkpoints()
        
        # Save last checkpoint
        if self.save_last:
            self._save_checkpoint(trainer, epoch, metrics, "last")
        
        # Save best checkpoint
        if self.save_best and self.monitor_metric in metrics:
            current_metric = metrics[self.monitor_metric]
            if self._is_better(current_metric, self.best_metric):
                self.best_metric = current_metric
                self.best_epoch = epoch
                checkpoint_path = self._save_checkpoint(trainer, epoch, metrics, "best")
                self.best_checkpoints.append((current_metric, checkpoint_path))
                self._cleanup_checkpoints()
                self.logger.info(
                    f"New best {self.monitor_metric}: {current_metric:.4f} at epoch {epoch}"
                )


class LoggingCallback(Callback):
    """Callback for logging training progress."""
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_every_n_steps: int = 10,
        log_to_console: bool = True,
        log_to_file: bool = True,
    ):
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_every_n_steps = log_every_n_steps
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger()
        self.history: Dict[str, List[float]] = {}
    
    def on_train_begin(self, trainer: Any) -> None:
        """Log training start."""
        self.logger.info("=" * 80)
        self.logger.info("Training started")
        self.logger.info(f"Total epochs: {trainer.config.epochs}")
        self.logger.info(f"Train batch size: {trainer.config.train_batch_size}")
        self.logger.info("=" * 80)
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Log epoch start."""
        self.logger.info(f"\nEpoch {epoch + 1}/{trainer.config.epochs}")
        self.logger.info("-" * 80)
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Log batch progress."""
        if (batch_idx + 1) % self.log_every_n_steps == 0:
            self.logger.info(
                f"Batch {batch_idx + 1}/{len(trainer.train_loader)}: "
                f"loss = {format_number(loss)}"
            )
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Log epoch metrics."""
        # Update history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Log metrics
        self.logger.info("\nEpoch metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {format_number(value)}")
        
        # Save history to file
        if self.log_to_file and self.log_dir:
            history_path = self.log_dir / "training_history.json"
            save_json(self.history, history_path)
    
    def on_train_end(self, trainer: Any) -> None:
        """Log training end."""
        self.logger.info("=" * 80)
        self.logger.info("Training completed")
        self.logger.info("=" * 80)


class EarlyStoppingCallback(Callback):
    """Callback for early stopping."""
    
    def __init__(
        self,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min",
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
    ):
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.logger = get_logger()
        self.best_metric = float('-inf') if monitor_mode == 'max' else float('inf')
        self.best_epoch = -1
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = -1
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.monitor_mode == 'max':
            return current > best + self.min_delta
        return current < best - self.min_delta
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Check for early stopping."""
        if self.monitor_metric not in metrics:
            return
        
        current_metric = metrics[self.monitor_metric]
        
        if self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.wait = 0
            
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                self.logger.info(
                    f"\nEarly stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor_metric}: {self.best_metric:.4f} at epoch {self.best_epoch}"
                )
                
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                    self.logger.info("Restored best weights")


class ProgressCallback(Callback):
    """Callback for progress bar using tqdm."""
    
    def __init__(self, show_batch_progress: bool = True):
        self.show_batch_progress = show_batch_progress
        self.epoch_pbar = None
        self.batch_pbar = None
    
    def on_train_begin(self, trainer: Any) -> None:
        """Initialize epoch progress bar."""
        self.epoch_pbar = tqdm(
            total=trainer.config.epochs,
            desc="Training",
            unit="epoch",
            position=0,
        )
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Initialize batch progress bar."""
        if self.show_batch_progress:
            self.batch_pbar = tqdm(
                total=len(trainer.train_loader),
                desc=f"Epoch {epoch + 1}",
                unit="batch",
                position=1,
                leave=False,
            )
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Update batch progress bar."""
        if self.batch_pbar:
            self.batch_pbar.update(1)
            self.batch_pbar.set_postfix({"loss": f"{loss:.4f}"})
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Update epoch progress bar."""
        if self.batch_pbar:
            self.batch_pbar.close()
            self.batch_pbar = None
        
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            # Show key metrics in progress bar
            postfix = {k: f"{v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))}
            self.epoch_pbar.set_postfix(postfix)
    
    def on_train_end(self, trainer: Any) -> None:
        """Close progress bars."""
        if self.batch_pbar:
            self.batch_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()


class CallbackList:
    """Container for managing multiple callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer: Any) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer: Any) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, metrics: MetricsDict, trainer: Any) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)
    
    def on_batch_begin(self, batch_idx: int, trainer: Any) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)
    
    def on_batch_end(self, batch_idx: int, loss: float, trainer: Any) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, trainer)
    
    def on_validation_begin(self, trainer: Any) -> None:
        """Call on_validation_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_begin(trainer)
    
    def on_validation_end(self, metrics: MetricsDict, trainer: Any) -> None:
        """Call on_validation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(metrics, trainer)


# Made with Bob