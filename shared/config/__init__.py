"""
Configuration Management (ML Core Layer)

This module provides global ML settings and configuration management:
- Global ML settings (DataConfig, ModelConfig, TrainingConfig, etc.)
- Workflow validation (workflow_validation.py)
- Shared constants
- Configuration utilities (get_config, update_config, reset_config)

Architecture:
- Layer: ML Core (global settings)
- Used by: All ML modules
- Related to: uqlab_orchestrator.config (orchestration settings)

Related Modules:
- Orchestration Config: uqlab_orchestrator.config (sweep presets, experiment templates)
- UI Config: uqlab.ui_components.config (UI form builders, Pydantic models)
- Validation: workflow_validation.py (validation rules and logic)

Note: This is the ML core layer - orchestration settings are in uqlab_orchestrator.config,
UI-specific configurations are in uqlab.ui_components.config.
"""

# Import from global_config.py (was ../config.py)
from .global_config import (
    DataConfig,
    EvaluationConfig,
    GlobalConfig,
    ModelConfig,
    PathConfig,
    SystemConfig,
    TrainingConfig,
    get_config,
    reset_config,
    update_config,
)

# Import from classification.py and schemas.py
from .classification import *
from .schemas import *

__all__ = [
    # From global_config.py
    "DataConfig",
    "EvaluationConfig",
    "GlobalConfig",
    "ModelConfig",
    "PathConfig",
    "SystemConfig",
    "TrainingConfig",
    "get_config",
    "reset_config",
    "update_config",
]
