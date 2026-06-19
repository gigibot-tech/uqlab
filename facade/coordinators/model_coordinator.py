"""
Model Coordinator for the Experiment Facade pattern.

Handles model creation, initialization, and management.
"""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import logging

from .base import BaseCoordinator


class ModelCoordinator(BaseCoordinator):
    """
    Coordinates all model-related operations for experiments.
    
    Responsibilities:
    - Create and initialize models (DINOv2, ResNet, etc.)
    - Manage model architecture configuration
    - Handle model checkpointing and loading
    - Configure MC Dropout for uncertainty estimation
    - Manage model device placement (CPU/GPU)
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the Model Coordinator.
        
        Args:
            config: Configuration dictionary containing:
                - model_type: Type of model ("dinov2", "resnet", etc.)
                - dinov2_model: DINOv2 variant ("small", "base", "large")
                - hidden_dim: Hidden layer dimension
                - dropout: Dropout rate for MC Dropout
                - num_classes: Number of output classes
                - freeze_backbone: Whether to freeze feature extractor
                - device: Device to use ("cuda" or "cpu")
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        
    def setup(self) -> None:
        """
        Setup model resources.
        
        Creates and initializes the model based on configuration.
        """
        self.logger.info("Setting up Model Coordinator...")
        
        # Determine device
        device_str = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        self.logger.info(f"Using device: {self.device}")
        
        # Create model
        self._create_model()
        
        # Move model to device
        if self.model:
            self.model = self.model.to(self.device)
            
        # Store model info in state
        self._state["model_type"] = self.config.get("model_type", "unknown")
        self._state["device"] = str(self.device)
        self._state["num_parameters"] = self._count_parameters()
        
        self.logger.info(f"Model Coordinator setup complete. "
                        f"Model: {self._state['model_type']}, "
                        f"Parameters: {self._state['num_parameters']:,}")
    
    def teardown(self) -> None:
        """
        Cleanup model resources.
        
        Releases model and moves it off GPU if necessary.
        """
        self.logger.info("Tearing down Model Coordinator...")
        if self.model and self.device.type == "cuda":
            self.model = self.model.cpu()
        self.model = None
        self._state.clear()
        torch.cuda.empty_cache()
        self.logger.info("Model Coordinator teardown complete")
    
    def _create_model(self) -> None:
        """
        Create the model based on configuration.
        
        This method will integrate with existing model classes like
        EmbeddingDropoutMLP and DINOv2Backbone.
        """
        model_type = self.config.get("model_type", "dinov2")
        
        self.logger.info(f"Creating model: {model_type}")
        
        # TODO: Integrate with existing model classes
        # For now, create a placeholder
        self.logger.warning("Model creation not yet implemented - using placeholder")
        
        # Placeholder implementation
        # In the full implementation, this will:
        # 1. Load DINOv2 backbone or ResNet
        # 2. Create EmbeddingDropoutMLP head
        # 3. Configure MC Dropout
        # 4. Optionally freeze backbone
        
        self.model = None  # Will be actual model instance
    
    def _count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            Number of trainable parameters
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model(self) -> Optional[nn.Module]:
        """
        Get the model instance.
        
        Returns:
            The PyTorch model
        """
        return self.model
    
    def get_device(self) -> torch.device:
        """
        Get the device the model is on.
        
        Returns:
            The torch device
        """
        return self.device
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Checkpoint loaded successfully")
    
    def save_checkpoint(self, checkpoint_path: str, additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights to a checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint
            additional_info: Optional additional information to save
        """
        self.logger.info(f"Saving checkpoint to: {checkpoint_path}")
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info("Checkpoint saved successfully")
    
    def set_train_mode(self) -> None:
        """Set the model to training mode."""
        if self.model:
            self.model.train()
            self.logger.debug("Model set to training mode")
    
    def set_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        if self.model:
            self.model.eval()
            self.logger.debug("Model set to evaluation mode")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_type": self._state.get("model_type", "unknown"),
            "num_parameters": self._state.get("num_parameters", 0),
            "device": self._state.get("device", "unknown"),
            "is_training": self.model.training if self.model else False,
        }
    
    def freeze_backbone(self) -> None:
        """
        Freeze the backbone (feature extractor) parameters.
        
        Useful for transfer learning scenarios.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        
        self.logger.info("Freezing backbone parameters")
        # TODO: Implement backbone freezing logic
        # This will depend on the model architecture
        pass
    
    def unfreeze_backbone(self) -> None:
        """
        Unfreeze the backbone (feature extractor) parameters.
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        
        self.logger.info("Unfreezing backbone parameters")
        # TODO: Implement backbone unfreezing logic
        pass

# Made with Bob
