"""
Optional Hydra integration for advanced configuration management.

This module provides Hydra integration while maintaining backward compatibility.
Hydra is OPTIONAL - the code works fine without it.

Features with Hydra:
- Config composition (combine multiple YAML files)
- Command-line overrides (python script.py data.epochs=20)
- Config versioning and experiment tracking
- Multi-run sweeps (parameter grid search)

Usage:
    # With Hydra (optional)
    import hydra
    from omegaconf import DictConfig
    from uqlab.evaluation.classification.hydra_wrapper import hydra_to_dataclass
    
    @hydra.main(config_path="../configs", config_name="experiment/default")
    def main(cfg: DictConfig):
        # Convert to validated dataclass
        config = hydra_to_dataclass(cfg)
        config.validate()
        
        # Use config
        run_experiment(config)
    
    # Without Hydra (still works)
    from uqlab.evaluation.classification.config_schema import ExperimentConfig
    config = ExperimentConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    try:
        from omegaconf import DictConfig
    except ImportError:
        DictConfig = Dict[str, Any]  # type: ignore

from uqlab.evaluation.classification.config_schema import ExperimentConfig


def hydra_to_dataclass(cfg: "DictConfig") -> ExperimentConfig:
    """
    Convert Hydra OmegaConf to validated dataclass config.
    
    This function bridges Hydra's DictConfig with our type-safe dataclass config.
    
    Args:
        cfg: Hydra DictConfig from @hydra.main decorator
        
    Returns:
        Validated ExperimentConfig instance
        
    Example:
        >>> @hydra.main(config_path="../configs", config_name="experiment/default")
        >>> def main(cfg: DictConfig):
        >>>     config = hydra_to_dataclass(cfg)
        >>>     config.validate()
        >>>     run_experiment(config)
    """
    try:
        from omegaconf import OmegaConf
        
        # Convert OmegaConf to plain dict
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        # Hydra not installed - treat as regular dict
        config_dict = dict(cfg)
    
    # Convert to dataclass with validation
    return ExperimentConfig.from_dict(config_dict)


def dataclass_to_hydra(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Convert dataclass config to Hydra-compatible dict.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        Dictionary compatible with Hydra/OmegaConf
        
    Example:
        >>> config = ExperimentConfig()
        >>> hydra_dict = dataclass_to_hydra(config)
        >>> # Save as Hydra config
        >>> OmegaConf.save(hydra_dict, "config.yaml")
    """
    return config.to_dict()


def is_hydra_available() -> bool:
    """
    Check if Hydra is installed.
    
    Returns:
        True if Hydra is available, False otherwise
    """
    try:
        import hydra  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
        return True
    except ImportError:
        return False


# Example usage function (for documentation)
def example_hydra_script():
    """
    Example of using Hydra with our config system.
    
    Save this as a separate script (e.g., train_with_hydra.py):
    
    ```python
    import hydra
    from omegaconf import DictConfig
    from uqlab.evaluation.classification.hydra_wrapper import hydra_to_dataclass
    
    @hydra.main(version_base=None, config_path="../configs", config_name="experiment/default")
    def main(cfg: DictConfig):
        # Convert to validated dataclass
        config = hydra_to_dataclass(cfg)
        config.validate()
        
        print(f"Running experiment with config:")
        print(f"  Noise type: {config.data.noise_type}")
        print(f"  Model: {config.model.dinov2_model}")
        print(f"  Epochs: {config.training.epochs}")
        
        # Run your experiment
        # run_experiment(config)
    
    if __name__ == "__main__":
        main()
    ```
    
    Then run with:
    ```bash
    # Use default config
    python train_with_hydra.py
    
    # Override parameters
    python train_with_hydra.py data.epochs=20 model.hidden_dim=512
    
    # Use different config
    python train_with_hydra.py experiment=fast_pilot
    
    # Multi-run sweep
    python train_with_hydra.py -m model.hidden_dim=128,256,512
    ```
    """
    pass


# Made with Bob