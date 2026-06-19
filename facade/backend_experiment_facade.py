"""
Backend Experiment Facade - Extension for FastAPI backend integration.

This facade extends ExperimentFacade to add backend-specific functionality
like database integration, async execution, and API endpoint support.
"""

from typing import Any, Dict, Optional, List, Callable
import logging
import asyncio
from datetime import datetime
from pathlib import Path

from .experiment_facade import ExperimentFacade


class BackendExperimentFacade(ExperimentFacade):
    """
    Backend-specific extension of ExperimentFacade.
    
    Adds functionality for:
    - Database integration (experiment tracking)
    - Async execution support
    - Progress callbacks for API endpoints
    - Experiment status management
    - Result persistence to database
    
    Usage:
        ```python
        from uqlab.facade import BackendExperimentFacade
        
        config = {...}
        facade = BackendExperimentFacade(
            config,
            experiment_id="exp_123",
            db_session=session
        )
        
        # Async execution
        results = await facade.run_experiment_async()
        
        # Or sync with progress callbacks
        def progress_callback(phase, progress, message):
            print(f"{phase}: {progress}% - {message}")
        
        results = facade.run_experiment(progress_callback=progress_callback)
        ```
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
        db_session: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the Backend Experiment Facade.
        
        Args:
            config: Complete experiment configuration dictionary
            experiment_id: Optional experiment ID for database tracking
            db_session: Optional database session for persistence
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.db_session = db_session
        self.status = "pending"
        self.progress = 0.0
        self.current_phase = "initialization"
        self.progress_callbacks: List[Callable[[str, float, str], None]] = []
        
    def _generate_experiment_id(self) -> str:
        """Generate a unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get("experiment_name", "exp")
        return f"{exp_name}_{timestamp}"
    
    def add_progress_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """
        Add a progress callback function.
        
        Args:
            callback: Function with signature (phase: str, progress: float, message: str)
        """
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, phase: str, progress: float, message: str) -> None:
        """
        Notify all progress callbacks.
        
        Args:
            phase: Current phase name
            progress: Progress percentage (0-100)
            message: Progress message
        """
        self.current_phase = phase
        self.progress = progress
        
        for callback in self.progress_callbacks:
            try:
                callback(phase, progress, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def _update_status(self, status: str) -> None:
        """
        Update experiment status.
        
        Args:
            status: New status ("pending", "running", "completed", "failed")
        """
        self.status = status
        self.logger.info(f"Experiment status: {status}")
        
        # Update database if session available
        if self.db_session:
            self._persist_status_to_db()
    
    def _persist_status_to_db(self) -> None:
        """Persist current status to database."""
        # TODO: Implement database persistence
        # This will integrate with the existing database models
        self.logger.debug(f"Persisting status to DB: {self.status}")
        pass
    
    def _persist_results_to_db(self, results: Dict[str, Any]) -> None:
        """
        Persist experiment results to database.
        
        Args:
            results: Experiment results dictionary
        """
        # TODO: Implement database persistence
        # This will integrate with the existing database models
        self.logger.debug("Persisting results to DB")
        pass
    
    def setup(self) -> None:
        """
        Setup all coordinators with progress tracking.
        """
        self._update_status("running")
        self._notify_progress("setup", 0, "Initializing coordinators...")
        
        super().setup()
        
        self._notify_progress("setup", 100, "Setup complete")
    
    def run_experiment(self, progress_callback: Optional[Callable[[str, float, str], None]] = None) -> Dict[str, Any]:
        """
        Run the complete experiment workflow with progress tracking.
        
        Args:
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary containing experiment results
        """
        if progress_callback:
            self.add_progress_callback(progress_callback)
        
        try:
            self._update_status("running")
            
            if not self._is_setup:
                self.setup()
            
            self.logger.info("=" * 80)
            self.logger.info(f"Starting Backend Experiment: {self.experiment_id}")
            self.logger.info("=" * 80)
            
            # Phase 1: Training
            self._notify_progress("training", 0, "Starting training phase...")
            training_results = self._run_training_with_progress()
            self._notify_progress("training", 100, "Training complete")
            
            # Phase 2: Evaluation
            self._notify_progress("evaluation", 0, "Starting evaluation phase...")
            evaluation_results = self._run_evaluation()
            self._notify_progress("evaluation", 100, "Evaluation complete")
            
            # Phase 3: Result Collection
            self._notify_progress("results", 0, "Collecting results...")
            final_results = self._collect_results(training_results, evaluation_results)
            self._notify_progress("results", 100, "Results collected")
            
            # Add experiment metadata
            final_results["experiment_id"] = self.experiment_id
            final_results["status"] = "completed"
            
            # Persist to database
            if self.db_session:
                self._persist_results_to_db(final_results)
            
            self._update_status("completed")
            
            self.logger.info("=" * 80)
            self.logger.info("Backend Experiment Complete!")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Backend experiment failed: {str(e)}", exc_info=True)
            self._update_status("failed")
            self._notify_progress("error", 0, f"Experiment failed: {str(e)}")
            raise
        finally:
            self.teardown()
    
    def _run_training_with_progress(self) -> Dict[str, Any]:
        """
        Run training phase with progress callbacks.
        
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Initializing training...")
        
        # Get model and data loaders
        model = self.model_coordinator.get_model()
        device = self.model_coordinator.get_device()
        train_loader = self.data_coordinator.get_train_loader()
        val_loader = self.data_coordinator.get_val_loader()
        
        if model is None or train_loader is None:
            raise ValueError("Model or training data not initialized")
        
        # Initialize optimizer and criterion
        self.training_coordinator.initialize_optimizer(model)
        self.training_coordinator.initialize_criterion()
        
        # Training loop with progress tracking
        epochs = self.config.get("epochs", 10)
        training_history = []
        final_epoch = epochs  # Default to full epochs
        
        for epoch in range(1, epochs + 1):
            final_epoch = epoch  # Track actual final epoch
            epoch_progress = (epoch / epochs) * 100
            self._notify_progress(
                "training",
                epoch_progress,
                f"Training epoch {epoch}/{epochs}"
            )
            
            self.logger.info(f"\nEpoch {epoch}/{epochs}")
            self.logger.info("-" * 40)
            
            # Train epoch
            train_metrics = self.training_coordinator.train_epoch(
                model, train_loader, device
            )
            
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validate epoch
            if val_loader:
                val_metrics = self.training_coordinator.validate_epoch(
                    model, val_loader, device
                )
                self.logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                              f"Val Acc: {val_metrics['accuracy']:.2f}%")
                
                epoch_metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "learning_rate": train_metrics["learning_rate"],
                }
            else:
                epoch_metrics = {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "learning_rate": train_metrics["learning_rate"],
                }
            
            # Record results
            self.result_coordinator.add_training_epoch_result(epoch, epoch_metrics)
            training_history.append(epoch_metrics)
            
            # Persist intermediate results to database
            if self.db_session and epoch % 5 == 0:
                self._persist_intermediate_results(epoch, epoch_metrics)
            
            # Check early stopping
            if self.training_coordinator.should_stop_early():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        return {
            "training_history": training_history,
            "final_epoch": final_epoch,
        }
    
    def _persist_intermediate_results(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Persist intermediate training results to database.
        
        Args:
            epoch: Current epoch number
            metrics: Epoch metrics
        """
        # TODO: Implement database persistence
        self.logger.debug(f"Persisting intermediate results for epoch {epoch}")
        pass
    
    async def run_experiment_async(self) -> Dict[str, Any]:
        """
        Run the experiment asynchronously.
        
        This method allows the experiment to run in the background
        without blocking the main thread, useful for API endpoints.
        
        Returns:
            Dictionary containing experiment results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_experiment)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current experiment status.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "progress": self.progress,
            "current_phase": self.current_phase,
            "coordinator_states": self.get_coordinator_states() if self._is_setup else {},
        }
    
    def cancel_experiment(self) -> None:
        """
        Cancel the running experiment.
        
        This method attempts to gracefully stop the experiment
        and save any intermediate results.
        """
        self.logger.warning("Experiment cancellation requested")
        self._update_status("cancelled")
        
        # Save intermediate results
        try:
            self.result_coordinator.save_results("cancelled_results.json")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate results: {e}")
        
        # Teardown
        self.teardown()

# Made with Bob
