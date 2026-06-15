"""
Experiment Type Validation UI Components

This module provides UI components for marking and validating experiments
according to the theoretical framework for uncertainty disentanglement.

Theoretical Framework:
- Experiment 1 (Epistemic): Vary training size to control epistemic uncertainty (Ue)
- Experiment 2 (Aleatoric): Vary label noise to control aleatoric uncertainty (Ua)

Validation Criteria:
- Consistency: (C1) ua ∼∝ Ua, (C2) ue ∼∝ Ue
- Orthogonality: (O1) ua ⊥ Ue, (O2) ue ⊥ Ua
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Any

from .config_types import ValidationConfig


def render_experiment_type_validation(
    key_prefix: str = "default"
) -> ValidationConfig:
    """
    Render experiment type validation checkboxes.
    
    This allows users to mark experiments as testing specific uncertainty types
    according to the theoretical framework.
    
    Args:
        key_prefix: Prefix for widget keys to avoid duplicates
    
    Returns:
        ValidationConfig dataclass with validation configuration
    """
    st.markdown("### 🔬 Experiment Type Validation")
    st.info("""
    💡 **Theoretical Framework**: Mark this experiment to validate uncertainty disentanglement
    
    - **Epistemic Sweep**: Varies training data size to control model uncertainty (Ue)
    - **Aleatoric Sweep**: Varies label noise to control data uncertainty (Ua)
    """)
    
    col1, col2 = st.columns(2)
    
    # Initialize parameters
    epistemic_param: Optional[str] = None
    aleatoric_param: Optional[str] = None
    
    with col1:
        st.markdown("#### 📊 Epistemic Sweep")
        is_epistemic = st.checkbox(
            "Mark as Epistemic Sweep",
            value=False,
            key=f"{key_prefix}_is_epistemic",
            help="Check if this experiment varies training data size to test epistemic uncertainty"
        )
        
        if is_epistemic:
            epistemic_param = st.selectbox(
                "Swept Parameter",
                ["under_train_per_class", "regular_train_per_class", "both"],
                key=f"{key_prefix}_epistemic_param",
                help="Which parameter controls epistemic uncertainty"
            )
            
            st.caption("""
            ✅ **Expected Behavior:**
            - C2: ue ↑ as training size ↓
            - O1: ua stays constant
            """)
    
    with col2:
        st.markdown("#### 🎲 Aleatoric Sweep")
        is_aleatoric = st.checkbox(
            "Mark as Aleatoric Sweep",
            value=False,
            key=f"{key_prefix}_is_aleatoric",
            help="Check if this experiment varies label noise to test aleatoric uncertainty"
        )
        
        if is_aleatoric:
            aleatoric_param = st.selectbox(
                "Swept Parameter",
                ["aleatoric_noise_percentage", "cifar10n_noise_type"],
                key=f"{key_prefix}_aleatoric_param",
                help="Which parameter controls aleatoric uncertainty"
            )
            
            st.caption("""
            ✅ **Expected Behavior:**
            - C1: ua ↑ as noise ↑
            - O2: ue stays constant
            """)
    
    # Show warning if both are selected (2D grid)
    if is_epistemic and is_aleatoric:
        st.success("""
        ✅ **2D Grid Sweep Detected**: This experiment will test both epistemic and aleatoric uncertainty.
        
        Validation will check:
        - C1: ua correlates with noise level
        - C2: ue correlates with training size
        - O1: ua independent of training size
        - O2: ue independent of noise level
        """)
    
    return ValidationConfig(
        validation_enabled=is_epistemic or is_aleatoric,
        is_epistemic_sweep=is_epistemic,
        is_aleatoric_sweep=is_aleatoric,
        epistemic_parameter=epistemic_param,
        aleatoric_parameter=aleatoric_param
    )


def render_validation_summary(
    validation_config: ValidationConfig,
    experiment_config: Dict[str, Any]
) -> None:
    """
    Render a summary of validation configuration and expected outcomes.
    
    Args:
        validation_config: ValidationConfig dataclass
        experiment_config: Full experiment configuration
    """
    if not validation_config.validation_enabled:
        return
    
    st.markdown("---")
    st.markdown("### 📋 Validation Summary")
    
    # Create summary based on validation type
    if validation_config.is_epistemic_sweep and validation_config.is_aleatoric_sweep:
        st.info("""
        **2D Grid Sweep Validation**
        
        This experiment will validate the full uncertainty disentanglement framework:
        
        **Consistency Checks:**
        - ✅ C1: Aleatoric signal (ua) should correlate with noise level
        - ✅ C2: Epistemic signal (ue) should correlate with training size
        
        **Orthogonality Checks:**
        - ✅ O1: Aleatoric signal (ua) should be independent of training size
        - ✅ O2: Epistemic signal (ue) should be independent of noise level
        
        **Metrics:**
        - Pearson correlation coefficients (ρ) for each condition
        - UDE (Uncertainty Disentanglement Error) aggregate score
        """)
    
    elif validation_config.is_epistemic_sweep:
        param = validation_config.epistemic_parameter
        st.info(f"""
        **Epistemic Sweep Validation**
        
        Swept parameter: `{param}`
        
        **Expected Outcomes:**
        - ✅ C2: Epistemic signal (ue) should increase as `{param}` decreases
        - ✅ O1: Aleatoric signal (ua) should remain constant across sweep
        
        **Validation Metrics:**
        - ρ(ue, Ue): Should be high (> 0.7) for consistency
        - ρ(ua, Ue): Should be low (< 0.3) for orthogonality
        """)
    
    elif validation_config.is_aleatoric_sweep:
        param = validation_config.aleatoric_parameter
        st.info(f"""
        **Aleatoric Sweep Validation**
        
        Swept parameter: `{param}`
        
        **Expected Outcomes:**
        - ✅ C1: Aleatoric signal (ua) should increase as `{param}` increases
        - ✅ O2: Epistemic signal (ue) should remain constant across sweep
        
        **Validation Metrics:**
        - ρ(ua, Ua): Should be high (> 0.7) for consistency
        - ρ(ue, Ua): Should be low (< 0.3) for orthogonality
        """)
    
    # Show configuration details
    with st.expander("📊 Configuration Details"):
        st.json({
            'validation_enabled': validation_config.validation_enabled,
            'is_epistemic_sweep': validation_config.is_epistemic_sweep,
            'is_aleatoric_sweep': validation_config.is_aleatoric_sweep,
            'epistemic_parameter': validation_config.epistemic_parameter,
            'aleatoric_parameter': validation_config.aleatoric_parameter
        })


def get_validation_badge(validation_config: ValidationConfig) -> str:
    """
    Get a badge string for the experiment type.
    
    DEPRECATED: Use ValidationConfig.get_badge() method instead.
    
    Args:
        validation_config: ValidationConfig dataclass
    
    Returns:
        Badge string (e.g., "🔬 Epistemic", "🎲 Aleatoric", "📊 2D Grid")
    """
    return validation_config.get_badge()


def validate_sweep_configuration(
    validation_config: ValidationConfig,
    experiment_config: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate that the experiment configuration matches the declared sweep type.
    
    Args:
        validation_config: ValidationConfig dataclass
        experiment_config: Full experiment configuration
    
    Returns:
        Tuple of (is_valid, warnings)
        - is_valid: Whether configuration is valid for declared sweep type
        - warnings: List of warning messages
    """
    warnings = []
    
    if not validation_config.validation_enabled:
        return True, warnings
    
    # Check epistemic sweep
    if validation_config.is_epistemic_sweep:
        param = validation_config.epistemic_parameter
        
        # Check if parameter is actually being swept
        if param == 'under_train_per_class':
            # In batch experiments, this should be in sweep_definitions
            # In single experiments, this is just a marker
            pass
        elif param == 'regular_train_per_class':
            pass
        elif param == 'both':
            pass
    
    # Check aleatoric sweep
    if validation_config.is_aleatoric_sweep:
        param = validation_config.aleatoric_parameter
        
        if param == 'aleatoric_noise_percentage':
            # Should have custom noise enabled
            noise_pct = experiment_config.get('aleatoric_noise_percentage', 0)
            if noise_pct == 0:
                warnings.append("⚠️ Aleatoric sweep marked but aleatoric_noise_percentage is 0")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


# Made with Bob