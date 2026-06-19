"""
Experiment Facade - Main orchestration class for ML experiments.

This facade provides a simplified, high-level interface for running complete
ML experiments by coordinating specialized domain coordinators.
"""

from typing import Any, Dict, Optional
import logging
from pathlib import Path

from .coordinators import (
    DataCoordinator,
    ModelCoordinator,
    TrainingCoordinator,
    EvaluationCoordinator,
    ResultCoordinator,
)


class ExperimentFacade:
    """
    Facade for orchestrating complete ML experiments.
    
    This class implements the Facade pattern (Gang of Four) to provide a
    simplified interface to the complex subsystem of ML experiment execution.
    It coordinates five specialized coordinators:
    
    1. DataCoordinator - Dataset loading and preprocessing
    2. ModelCoordinator - Model creation and management
    3. TrainingCoordinator - Training loop execution
    4. EvaluationCoordinator - Evaluation and uncertainty quantification
    5. ResultCoordinator - Result collection and storage
    
    Usage:
        ```python
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
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Experiment Facade.
        
        Args:
            config: Complete experiment configuration dictionary
            logger: Optional logger instance. If None, creates a new logger.
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Initialize coordinators
        self.data_coordinator = DataCoordinator(
            self._extract_data_config(),
            logger=self.logger.getChild("Data")
        )
        
        self.model_coordinator = ModelCoordinator(
            self._extract_model_config(),
            logger=self.logger.getChild("Model")
        )
        
        self.training_coordinator = TrainingCoordinator(
            self._extract_training_config(),
            logger=self.logger.getChild("Training")
        )
        
        self.evaluation_coordinator = EvaluationCoordinator(
            self._extract_evaluation_config(),
            logger=self.logger.getChild("Evaluation")
        )
        
        self.result_coordinator = ResultCoordinator(
            self._extract_result_config(),
            logger=self.logger.getChild("Result")
        )
        
        self._is_setup = False
    
    def _extract_data_config(self) -> Dict[str, Any]:
        """Extract data-related configuration."""
        return {
            "dataset_name": self.config.get("dataset_name", "cifar10n"),
            "noise_type": self.config.get("noise_type", "worse_label"),
            "under_supported": self.config.get("under_supported", "random:2"),
            "under_train_per_class": self.config.get("under_train_per_class", 50),
            "regular_train_per_class": self.config.get("regular_train_per_class", 300),
            "eval_per_group": self.config.get("eval_per_group", 100),
            "batch_size": self.config.get("train_batch_size", 256),
            "num_workers": self.config.get("num_workers", 4),
        }
    
    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract model-related configuration."""
        return {
            "model_type": self.config.get("model_type", "dinov2"),
            "dinov2_model": self.config.get("dinov2_model", "small"),
            "hidden_dim": self.config.get("hidden_dim", 256),
            "dropout": self.config.get("dropout", 0.2),
            "num_classes": self.config.get("num_classes", 10),
            "freeze_backbone": self.config.get("freeze_backbone", False),
            "use_untrained_resnet": self.config.get("use_untrained_resnet", False),
            "device": self.config.get("device", "cuda"),
        }
    
    def _extract_training_config(self) -> Dict[str, Any]:
        """Extract training-related configuration."""
        return {
            "epochs": self.config.get("epochs", 10),
            "learning_rate": self.config.get("learning_rate", 0.001),
            "weight_decay": self.config.get("weight_decay", 0.0001),
            "optimizer": self.config.get("optimizer", "adam"),
            "scheduler": self.config.get("scheduler"),
            "early_stopping_patience": self.config.get("early_stopping_patience"),
            "gradient_clip": self.config.get("gradient_clip"),
            "criterion": self.config.get("criterion", "cross_entropy"),
        }
    
    def _extract_evaluation_config(self) -> Dict[str, Any]:
        """Extract evaluation-related configuration."""
        return {
            "mc_passes": self.config.get("mc_passes", 20),
            "uncertainty_signals": self.config.get("uncertainty_signals", []),
            "eval_batch_size": self.config.get("eval_batch_size", 256),
            "compute_auroc": self.config.get("compute_auroc", True),
        }
    
    def _extract_result_config(self) -> Dict[str, Any]:
        """Extract result-related configuration."""
        return {
            "results_dir": self.config.get("results_dir", "results"),
            "experiment_name": self.config.get("experiment_name", "experiment"),
            "save_format": self.config.get("save_format", "json"),
            "save_checkpoints": self.config.get("save_checkpoints", True),
        }
    
    def setup(self) -> None:
        """
        Setup all coordinators.
        
        This method must be called before running the experiment.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Setting up Experiment: {self.config.get('experiment_name', 'unnamed')}")
        self.logger.info("=" * 80)
        
        # Setup coordinators in order
        self.data_coordinator.setup()
        self.model_coordinator.setup()
        self.training_coordinator.setup()
        self.evaluation_coordinator.setup()
        self.result_coordinator.setup()
        
        self._is_setup = True
        self.logger.info("All coordinators setup complete")
    
    def teardown(self) -> None:
        """
        Teardown all coordinators.
        
        This method should be called after the experiment is complete
        to release resources.
        """
        self.logger.info("Tearing down all coordinators...")
        
        # Teardown in reverse order
        self.result_coordinator.teardown()
        self.evaluation_coordinator.teardown()
        self.training_coordinator.teardown()
        self.model_coordinator.teardown()
        self.data_coordinator.teardown()
        
        self._is_setup = False
        self.logger.info("All coordinators torn down")
    
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the complete experiment workflow.
        
        This is the main entry point for running an experiment. It orchestrates
        all phases: setup, training, evaluation, and result collection.
        
        Returns:
            Dictionary containing experiment results
        """
        if not self._is_setup:
            self.setup()
        
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting Experiment Execution")
            self.logger.info("=" * 80)
            
            # Phase 1: Training
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PHASE 1: Training")
            self.logger.info("=" * 80)
            training_results = self._run_training()
            
            # Phase 2: Evaluation
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PHASE 2: Evaluation")
            self.logger.info("=" * 80)
            evaluation_results = self._run_evaluation()
            
            # Phase 3: Result Collection
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PHASE 3: Result Collection")
            self.logger.info("=" * 80)
            final_results = self._collect_results(training_results, evaluation_results)
            
            self.logger.info("=" * 80)
            self.logger.info("Experiment Complete!")
            self.logger.info("=" * 80)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise
        finally:
            self.teardown()
    
    def _run_training(self) -> Dict[str, Any]:
        """
        Run the training phase.
        
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
        
        # Training loop
        epochs = self.config.get("epochs", 10)
        training_history = []
        
        for epoch in range(1, epochs + 1):
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
                
                # Combine metrics
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
            
            # Check early stopping
            if self.training_coordinator.should_stop_early():
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        return {
            "training_history": training_history,
            "final_epoch": epoch,
        }
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """
        Run the evaluation phase.
        
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Running evaluation...")
        
        model = self.model_coordinator.get_model()
        device = self.model_coordinator.get_device()
        test_loader = self.data_coordinator.get_test_loader()
        
        if model is None or test_loader is None:
            raise ValueError("Model or test data not initialized")
        
        # Evaluate on test set
        eval_results = self.evaluation_coordinator.evaluate_model(
            model, test_loader, device, compute_uncertainty=True
        )
        
        self.logger.info(f"Test Accuracy: {eval_results['accuracy']:.2f}%")
        
        # Record results
        self.result_coordinator.add_evaluation_result("test", eval_results)
        
        return eval_results
    
    def _collect_results(
        self,
        training_results: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Collect and organize all experiment results.
        
        Args:
            training_results: Results from training phase
            evaluation_results: Results from evaluation phase
        
        Returns:
            Dictionary containing complete experiment results
        """
        self.logger.info("Collecting results...")
        
        # Generate summary
        summary = self.result_coordinator.generate_summary()
        
        # Save results
        results_path = self.result_coordinator.save_results()
        
        final_results = {
            "summary": summary,
            "training": training_results,
            "evaluation": evaluation_results,
            "results_path": str(results_path),
        }
        
        self.logger.info(f"Results saved to: {results_path}")
        
        return final_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get the experiment configuration."""
        return self.config.copy()
    
    def get_coordinator_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current state of all coordinators.
        
        Returns:
            Dictionary mapping coordinator names to their states
        """
        return {
            "data": self.data_coordinator.get_state(),
            "model": self.model_coordinator.get_state(),
            "training": self.training_coordinator.get_state(),
            "evaluation": self.evaluation_coordinator.get_state(),
            "result": self.result_coordinator.get_state(),
        }

# Made with Bob
