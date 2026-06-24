"""
Workflow Validation Logic (ML Core Layer)

This module provides validation rules and logic for ML workflows:
- Pydantic validation models
- Configuration consistency checks
- Pure Python validation functions
- No UI code

Architecture:
- Layer: ML Core (validation logic)
- Used by: All ML modules, orchestration layer
- Visualized by: uqlab.ui_components.visualization.validation (UI layer)

Related Modules:
- Visualization: uqlab.ui_components.visualization.validation (validation dashboards)
- Orchestration: uqlab_orchestrator.config (experiment configurations)

Note: This is the ML core layer - visualization happens in the UI layer.
"""

"""
Pydantic validation models for Streamlit workflow configuration.

Ensures configuration consistency and catches errors early.
"""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError


class WorkflowDatasetConfig(BaseModel):
    """Validated dataset configuration."""
    
    dataset_name: Literal["cifar10", "cifar10n"] = "cifar10"
    noise_type: str = "clean_label"
    stats: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("noise_type")
    @classmethod
    def validate_noise_type(cls, v: str) -> str:
        """Validate noise type is recognized."""
        valid_types = [
            "clean_label",
            "worse_label",
            "aggre_label",
            "random_label1",
            "random_label2",
            "random_label3",
        ]
        if v not in valid_types:
            raise ValueError(f"noise_type must be one of {valid_types}, got {v}")
        return v
    
    @field_validator("stats")
    @classmethod
    def validate_stats(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure stats has required fields."""
        if "noise_rate" in v:
            noise_rate = v["noise_rate"]
            if not (0.0 <= noise_rate <= 1.0):
                raise ValueError(f"noise_rate must be in [0, 1], got {noise_rate}")
        return v


class WorkflowTrainingConfig(BaseModel):
    """Validated training configuration."""
    
    use_checkpoint: bool = False
    checkpoint_id: Optional[str] = None
    model_architecture: str = "dinov2-small"
    hidden_dim: int = Field(256, ge=64, le=2048)
    dropout: float = Field(0.2, ge=0.0, le=0.9)
    epochs: int = Field(12, ge=1, le=200)
    learning_rate: float = Field(0.001, gt=0.0, le=1.0)
    batch_size: int = Field(256, ge=1, le=2048)
    
    @field_validator("model_architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        """Validate model architecture."""
        valid_archs = [
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
            "resnet18",
            "resnet50",
        ]
        if v not in valid_archs:
            raise ValueError(f"model_architecture must be one of {valid_archs}, got {v}")
        return v
    
    @model_validator(mode="after")
    def validate_checkpoint_consistency(self):
        """Validate checkpoint configuration."""
        if self.use_checkpoint and not self.checkpoint_id:
            raise ValueError("use_checkpoint=True but checkpoint_id is None")
        
        if not self.use_checkpoint and self.checkpoint_id:
            # Warning: checkpoint_id set but not used
            self.checkpoint_id = None
        
        return self


class WorkflowUncertaintyConfig(BaseModel):
    """Validated uncertainty configuration."""
    
    # Epistemic uncertainty
    epistemic_enabled: bool = True
    under_supported: str = "random:2"
    under_train_per_class: int = Field(50, ge=1, le=5000)
    regular_train_per_class: int = Field(300, ge=1, le=5000)
    
    # Aleatoric uncertainty
    aleatoric_enabled: bool = True
    custom_noise_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Sweep configuration
    sweep_enabled: bool = False
    sweep_kind: Optional[Literal["dataset_size", "label_noise", "2d_grid"]] = None
    sweep_mode: Optional[Literal["quick", "standard", "thorough"]] = None
    
    epistemic_sweep_enabled: bool = False
    epistemic_sweep_values: List[int] = Field(default_factory=list)
    
    aleatoric_sweep_enabled: bool = False
    aleatoric_sweep_values: List[float] = Field(default_factory=list)
    
    @field_validator("under_supported")
    @classmethod
    def validate_under_supported(cls, v: str) -> str:
        """Validate under_supported format."""
        if v.startswith("random:"):
            try:
                num = int(v.split(":")[1])
                if not (1 <= num <= 9):
                    raise ValueError(f"random:N must have N in [1, 9], got {num}")
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid under_supported format: {v}") from e
        else:
            # Should be comma-separated class indices
            try:
                classes = [int(x.strip()) for x in v.split(",") if x.strip()]
                if not all(0 <= c <= 9 for c in classes):
                    raise ValueError("Class indices must be in [0, 9]")
                if len(classes) < 1 or len(classes) > 9:
                    raise ValueError("Must have 1-9 under-supported classes")
            except ValueError as e:
                raise ValueError(f"Invalid under_supported format: {v}") from e
        return v
    
    @model_validator(mode="after")
    def validate_sweep_consistency(self):
        """Validate sweep configuration consistency."""
        if self.sweep_enabled:
            if not self.sweep_kind:
                raise ValueError("sweep_enabled=True but sweep_kind is None")
            
            if self.sweep_kind == "dataset_size" and not self.epistemic_sweep_enabled:
                raise ValueError("sweep_kind=dataset_size but epistemic_sweep_enabled=False")
            
            if self.sweep_kind == "label_noise" and not self.aleatoric_sweep_enabled:
                raise ValueError("sweep_kind=label_noise but aleatoric_sweep_enabled=False")
            
            if self.sweep_kind == "2d_grid" and not (self.epistemic_sweep_enabled and self.aleatoric_sweep_enabled):
                raise ValueError("sweep_kind=2d_grid but both sweeps not enabled")
        
        return self


class WorkflowEvaluationConfig(BaseModel):
    """Validated evaluation configuration."""
    
    eval_per_group: int = Field(100, ge=10, le=10000)
    mc_passes: int = Field(0, ge=0, le=100)
    selected_signals: List[str] = Field(default_factory=list)
    
    @field_validator("selected_signals")
    @classmethod
    def validate_signals(cls, v: List[str]) -> List[str]:
        """Validate signal names."""
        from uqlab.evaluation.signals.catalog import SIGNAL_ID_ALIASES, METRIC_META, normalize_signal_id

        valid_signals = set(METRIC_META.keys()) | set(SIGNAL_ID_ALIASES.keys()) | {
            "epistemic_entropy",
            "aleatoric_entropy",
        }
        for signal in v:
            if signal not in valid_signals:
                raise ValueError(
                    f"Unknown signal: {signal}. Valid: {sorted(valid_signals)}"
                )
        return [normalize_signal_id(s) for s in v]


class WorkflowConfig(BaseModel):
    """Complete validated workflow configuration."""
    
    step1_complete: bool = False
    step2_complete: bool = False
    step3_complete: bool = False
    step4_complete: bool = False
    
    dataset_config: WorkflowDatasetConfig
    training_config: WorkflowTrainingConfig
    uncertainty_config: WorkflowUncertaintyConfig
    evaluation_config: WorkflowEvaluationConfig
    
    @model_validator(mode="after")
    def validate_aleatoric_consistency(self):
        """Validate aleatoric config against dataset config."""
        if self.uncertainty_config.aleatoric_enabled:
            noise_type = self.dataset_config.noise_type
            custom_noise = self.uncertainty_config.custom_noise_rate
            
            # Check if we have a noise source
            is_clean = noise_type == "clean_label"
            has_custom = custom_noise is not None
            
            if is_clean and not has_custom:
                raise ValueError(
                    "aleatoric_enabled=True but no noise source: "
                    "either set custom_noise_rate or use noisy dataset (noise_type != 'clean_label')"
                )
        
        return self
    
    @model_validator(mode="after")
    def validate_step_progression(self):
        """Validate that steps are completed in order."""
        if self.step2_complete and not self.step1_complete:
            raise ValueError("Cannot complete step 2 before step 1")
        if self.step3_complete and not self.step2_complete:
            raise ValueError("Cannot complete step 3 before step 2")
        if self.step4_complete and not self.step3_complete:
            raise ValueError("Cannot complete step 4 before step 3")
        
        return self
    
    class Config:
        """Pydantic config."""
        validate_assignment = True  # Validate on attribute assignment
        extra = "forbid"  # Forbid extra fields


def validate_workflow(workflow: Dict[str, Any]) -> WorkflowConfig:
    """
    Validate workflow configuration.
    
    Args:
        workflow: Raw workflow dictionary from Streamlit session state
        
    Returns:
        Validated WorkflowConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    return WorkflowConfig(**workflow)


def get_validation_errors(workflow: Dict[str, Any]) -> List[str]:
    """
    Get list of validation errors without raising exception.
    
    Args:
        workflow: Raw workflow dictionary
        
    Returns:
        List of error messages (empty if valid)
    """
    try:
        validate_workflow(workflow)
        return []
    except ValidationError as e:
        # Parse Pydantic validation errors
        return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
    except Exception as e:
        return [str(e)]

# Made with Bob
