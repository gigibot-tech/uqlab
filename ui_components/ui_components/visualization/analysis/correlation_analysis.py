"""
Correlation Analysis Module for Uncertainty Disentanglement Validation.

This module implements correlation analysis to validate uncertainty disentanglement
according to the theoretical framework with four conditions:
- C1: Aleatoric signal correlates with noise level
- C2: Epistemic signal correlates with training size
- O1: Aleatoric signal independent of training size
- O2: Epistemic signal independent of noise level
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy.stats import pearsonr


@dataclass
class CorrelationResult:
    """Result of correlation analysis between signal and parameter."""
    parameter_name: str  # e.g., "under_train_per_class" or "aleatoric_noise_percentage"
    signal_name: str  # e.g., "epistemic_uncertainty" or "aleatoric_uncertainty"
    correlation: float  # Pearson correlation coefficient
    p_value: float  # Statistical significance
    n_samples: int  # Number of data points
    
    @property
    def is_significant(self) -> bool:
        """Check if correlation is statistically significant (p < 0.05)."""
        return self.p_value < 0.05
    
    @property
    def strength(self) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(self.correlation)
        if abs_corr > 0.7:
            return "Strong"
        elif abs_corr > 0.4:
            return "Moderate"
        elif abs_corr > 0.2:
            return "Weak"
        else:
            return "Negligible"


@dataclass
class ValidationResult:
    """Complete validation result for an experiment sweep."""
    experiment_type: str  # "epistemic", "aleatoric", or "2d_grid"
    
    # Consistency checks
    c1_result: Optional[CorrelationResult] = None  # ua ~ Ua
    c2_result: Optional[CorrelationResult] = None  # ue ~ Ue
    
    # Orthogonality checks
    o1_result: Optional[CorrelationResult] = None  # ua ⊥ Ue
    o2_result: Optional[CorrelationResult] = None  # ue ⊥ Ua
    
    @property
    def c1_pass(self) -> bool:
        """Check if C1 condition passes (ρ > 0.7)."""
        return self.c1_result is not None and self.c1_result.correlation > 0.7
    
    @property
    def c2_pass(self) -> bool:
        """Check if C2 condition passes (ρ > 0.7)."""
        return self.c2_result is not None and self.c2_result.correlation > 0.7
    
    @property
    def o1_pass(self) -> bool:
        """Check if O1 condition passes (|ρ| < 0.3)."""
        return self.o1_result is not None and abs(self.o1_result.correlation) < 0.3
    
    @property
    def o2_pass(self) -> bool:
        """Check if O2 condition passes (|ρ| < 0.3)."""
        return self.o2_result is not None and abs(self.o2_result.correlation) < 0.3
    
    @property
    def ude_score(self) -> float:
        """
        Calculate Uncertainty Disentanglement Error (UDE).
        Lower is better. Perfect disentanglement = 0.
        """
        errors = []
        
        # Consistency errors (should be high, so error = 1 - ρ)
        if self.c1_result:
            errors.append(max(0, 0.7 - self.c1_result.correlation))
        if self.c2_result:
            errors.append(max(0, 0.7 - self.c2_result.correlation))
        
        # Orthogonality errors (should be low, so error = |ρ|)
        if self.o1_result:
            errors.append(max(0, abs(self.o1_result.correlation) - 0.3))
        if self.o2_result:
            errors.append(max(0, abs(self.o2_result.correlation) - 0.3))
        
        return np.mean(errors) if errors else 0.0


def calculate_correlation(
    signal_values: List[float],
    parameter_values: List[float],
    signal_name: str,
    parameter_name: str
) -> CorrelationResult:
    """
    Calculate Pearson correlation between signal and parameter.
    
    Args:
        signal_values: Uncertainty signal values (e.g., epistemic uncertainty)
        parameter_values: Swept parameter values (e.g., training size)
        signal_name: Name of the signal
        parameter_name: Name of the parameter
    
    Returns:
        CorrelationResult with correlation coefficient and p-value
    """
    if len(signal_values) != len(parameter_values):
        raise ValueError("Signal and parameter must have same length")
    
    if len(signal_values) < 3:
        raise ValueError("Need at least 3 data points for correlation")
    
    correlation, p_value = pearsonr(signal_values, parameter_values)
    
    return CorrelationResult(
        parameter_name=parameter_name,
        signal_name=signal_name,
        correlation=correlation,
        p_value=p_value,
        n_samples=len(signal_values)
    )


def analyze_epistemic_sweep(
    experiments: List[Dict],
    epistemic_parameter: str = "under_train_per_class"
) -> ValidationResult:
    """
    Analyze epistemic sweep experiments.
    
    Validates:
    - C2: ue should correlate with training size
    - O1: ua should be independent of training size
    
    Args:
        experiments: List of experiment results with configs and AUROC scores
        epistemic_parameter: Name of the swept epistemic parameter
    
    Returns:
        ValidationResult with C2 and O1 checks
    """
    # Extract data
    param_values = []
    ue_values = []  # Epistemic AUROC
    ua_values = []  # Aleatoric AUROC
    
    for exp in experiments:
        config = exp.get('config', {})
        param_val = config.get(epistemic_parameter)
        ue = exp.get('epistemic_auroc')
        ua = exp.get('aleatoric_auroc')
        
        if param_val is not None and ue is not None and ua is not None:
            param_values.append(float(param_val))
            ue_values.append(float(ue))
            ua_values.append(float(ua))
    
    if len(param_values) < 3:
        raise ValueError(f"Need at least 3 experiments, got {len(param_values)}")
    
    # C2: ue ~ Ue (epistemic signal correlates with training size)
    # Note: Inverse correlation expected (more training → less uncertainty)
    c2_result = calculate_correlation(
        ue_values, param_values,
        "epistemic_uncertainty", epistemic_parameter
    )
    
    # O1: ua ⊥ Ue (aleatoric signal independent of training size)
    o1_result = calculate_correlation(
        ua_values, param_values,
        "aleatoric_uncertainty", epistemic_parameter
    )
    
    return ValidationResult(
        experiment_type="epistemic",
        c2_result=c2_result,
        o1_result=o1_result
    )


def analyze_aleatoric_sweep(
    experiments: List[Dict],
    aleatoric_parameter: str = "aleatoric_noise_percentage"
) -> ValidationResult:
    """
    Analyze aleatoric sweep experiments.
    
    Validates:
    - C1: ua should correlate with noise level
    - O2: ue should be independent of noise level
    
    Args:
        experiments: List of experiment results with configs and AUROC scores
        aleatoric_parameter: Name of the swept aleatoric parameter
    
    Returns:
        ValidationResult with C1 and O2 checks
    """
    # Extract data
    param_values = []
    ue_values = []
    ua_values = []
    
    for exp in experiments:
        config = exp.get('config', {})
        param_val = config.get(aleatoric_parameter)
        ue = exp.get('epistemic_auroc')
        ua = exp.get('aleatoric_auroc')
        
        if param_val is not None and ue is not None and ua is not None:
            param_values.append(float(param_val))
            ue_values.append(float(ue))
            ua_values.append(float(ua))
    
    if len(param_values) < 3:
        raise ValueError(f"Need at least 3 experiments, got {len(param_values)}")
    
    # C1: ua ~ Ua (aleatoric signal correlates with noise level)
    c1_result = calculate_correlation(
        ua_values, param_values,
        "aleatoric_uncertainty", aleatoric_parameter
    )
    
    # O2: ue ⊥ Ua (epistemic signal independent of noise level)
    o2_result = calculate_correlation(
        ue_values, param_values,
        "epistemic_uncertainty", aleatoric_parameter
    )
    
    return ValidationResult(
        experiment_type="aleatoric",
        c1_result=c1_result,
        o2_result=o2_result
    )


def analyze_2d_grid(
    experiments: List[Dict],
    epistemic_parameter: str = "under_train_per_class",
    aleatoric_parameter: str = "aleatoric_noise_percentage"
) -> ValidationResult:
    """
    Analyze 2D grid sweep experiments.
    
    Validates all four conditions: C1, C2, O1, O2
    
    Args:
        experiments: List of experiment results from 2D grid
        epistemic_parameter: Name of epistemic parameter
        aleatoric_parameter: Name of aleatoric parameter
    
    Returns:
        ValidationResult with all four checks
    """
    # Extract data
    epis_values = []
    alea_values = []
    ue_values = []
    ua_values = []
    
    for exp in experiments:
        config = exp.get('config', {})
        epis_val = config.get(epistemic_parameter)
        alea_val = config.get(aleatoric_parameter)
        ue = exp.get('epistemic_auroc')
        ua = exp.get('aleatoric_auroc')
        
        if all(v is not None for v in [epis_val, alea_val, ue, ua]):
            epis_values.append(float(epis_val))
            alea_values.append(float(alea_val))
            ue_values.append(float(ue))
            ua_values.append(float(ua))
    
    if len(epis_values) < 3:
        raise ValueError(f"Need at least 3 experiments, got {len(epis_values)}")
    
    # C1: ua ~ Ua
    c1_result = calculate_correlation(
        ua_values, alea_values,
        "aleatoric_uncertainty", aleatoric_parameter
    )
    
    # C2: ue ~ Ue
    c2_result = calculate_correlation(
        ue_values, epis_values,
        "epistemic_uncertainty", epistemic_parameter
    )
    
    # O1: ua ⊥ Ue
    o1_result = calculate_correlation(
        ua_values, epis_values,
        "aleatoric_uncertainty", epistemic_parameter
    )
    
    # O2: ue ⊥ Ua
    o2_result = calculate_correlation(
        ue_values, alea_values,
        "epistemic_uncertainty", aleatoric_parameter
    )
    
    return ValidationResult(
        experiment_type="2d_grid",
        c1_result=c1_result,
        c2_result=c2_result,
        o1_result=o1_result,
        o2_result=o2_result
    )

# Made with Bob
