"""
Validation Visualization Module.

This module provides visualization components for displaying correlation analysis
results and validation summaries in the Streamlit UI.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Optional
from ..analysis.correlation_analysis import ValidationResult, CorrelationResult


def render_correlation_scatter(
    param_values: List[float],
    signal_values: List[float],
    param_name: str,
    signal_name: str,
    correlation_result: CorrelationResult
) -> None:
    """
    Render scatter plot with correlation line.
    
    Args:
        param_values: Parameter values (x-axis)
        signal_values: Signal values (y-axis)
        param_name: Name of parameter for axis label
        signal_name: Name of signal for axis label
        correlation_result: Correlation analysis result
    """
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=param_values,
        y=signal_values,
        mode='markers',
        name='Experiments',
        marker=dict(size=10, color='blue', opacity=0.7)
    ))
    
    # Trend line
    z = np.polyfit(param_values, signal_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(param_values), max(param_values), 100)
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Title with correlation info
    title_text = (
        f"{signal_name} vs {param_name}<br>"
        f"<sub>ρ = {correlation_result.correlation:.3f}, "
        f"p = {correlation_result.p_value:.3f}, "
        f"strength = {correlation_result.strength}</sub>"
    )
    
    fig.update_layout(
        title=title_text,
        xaxis_title=param_name,
        yaxis_title=signal_name,
        height=400,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_validation_summary(validation_result: ValidationResult) -> None:
    """
    Render validation summary with pass/fail indicators.
    
    Args:
        validation_result: Complete validation result with all checks
    """
    st.markdown("### 📊 Validation Summary")
    
    # UDE Score - prominent display
    ude = validation_result.ude_score
    if ude < 0.1:
        ude_color = "green"
        ude_status = "Excellent"
    elif ude < 0.3:
        ude_color = "orange"
        ude_status = "Good"
    else:
        ude_color = "red"
        ude_status = "Poor"
    
    st.markdown(
        f"**Uncertainty Disentanglement Error (UDE):** "
        f"<span style='color:{ude_color}; font-size:1.5em; font-weight:bold'>{ude:.3f}</span> "
        f"<span style='color:{ude_color}'>({ude_status})</span>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Condition checks in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Consistency Conditions")
        st.markdown("*Signals should correlate with their sources (ρ > 0.7)*")
        
        # C1: ua ~ Ua
        if validation_result.c1_result:
            icon = "✅" if validation_result.c1_pass else "❌"
            corr = validation_result.c1_result.correlation
            st.markdown(
                f"{icon} **C1**: Aleatoric ∼ Noise Level  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;ρ = {corr:.3f} "
                f"({validation_result.c1_result.strength})"
            )
        else:
            st.markdown("⚪ **C1**: Not tested (aleatoric sweep needed)")
        
        # C2: ue ~ Ue
        if validation_result.c2_result:
            icon = "✅" if validation_result.c2_pass else "❌"
            corr = validation_result.c2_result.correlation
            st.markdown(
                f"{icon} **C2**: Epistemic ∼ Training Size  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;ρ = {corr:.3f} "
                f"({validation_result.c2_result.strength})"
            )
        else:
            st.markdown("⚪ **C2**: Not tested (epistemic sweep needed)")
    
    with col2:
        st.markdown("#### 🔀 Orthogonality Conditions")
        st.markdown("*Signals should be independent of other sources (|ρ| < 0.3)*")
        
        # O1: ua ⊥ Ue
        if validation_result.o1_result:
            icon = "✅" if validation_result.o1_pass else "❌"
            corr = validation_result.o1_result.correlation
            st.markdown(
                f"{icon} **O1**: Aleatoric ⊥ Training Size  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;ρ = {corr:.3f} "
                f"({validation_result.o1_result.strength})"
            )
        else:
            st.markdown("⚪ **O1**: Not tested (epistemic sweep needed)")
        
        # O2: ue ⊥ Ua
        if validation_result.o2_result:
            icon = "✅" if validation_result.o2_pass else "❌"
            corr = validation_result.o2_result.correlation
            st.markdown(
                f"{icon} **O2**: Epistemic ⊥ Noise Level  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;ρ = {corr:.3f} "
                f"({validation_result.o2_result.strength})"
            )
        else:
            st.markdown("⚪ **O2**: Not tested (aleatoric sweep needed)")
    
    st.markdown("---")
    
    # Overall assessment
    st.markdown("#### 📈 Overall Assessment")
    
    # Count passes
    conditions_tested = []
    conditions_passed = []
    
    if validation_result.c1_result:
        conditions_tested.append("C1")
        if validation_result.c1_pass:
            conditions_passed.append("C1")
    
    if validation_result.c2_result:
        conditions_tested.append("C2")
        if validation_result.c2_pass:
            conditions_passed.append("C2")
    
    if validation_result.o1_result:
        conditions_tested.append("O1")
        if validation_result.o1_pass:
            conditions_passed.append("O1")
    
    if validation_result.o2_result:
        conditions_tested.append("O2")
        if validation_result.o2_pass:
            conditions_passed.append("O2")
    
    if conditions_tested:
        pass_rate = len(conditions_passed) / len(conditions_tested)
        st.markdown(
            f"**Conditions Passed:** {len(conditions_passed)}/{len(conditions_tested)} "
            f"({pass_rate*100:.0f}%)"
        )
        
        if pass_rate == 1.0:
            st.success("🎉 Perfect disentanglement! All tested conditions pass.")
        elif pass_rate >= 0.75:
            st.info("👍 Good disentanglement. Most conditions pass.")
        elif pass_rate >= 0.5:
            st.warning("⚠️ Moderate disentanglement. Some conditions fail.")
        else:
            st.error("❌ Poor disentanglement. Most conditions fail.")
    else:
        st.info("No conditions tested yet. Run experiments to validate.")


def render_correlation_details(correlation_result: CorrelationResult) -> None:
    """
    Render detailed information about a correlation result.
    
    Args:
        correlation_result: Correlation analysis result
    """
    with st.expander(f"📊 Details: {correlation_result.signal_name} vs {correlation_result.parameter_name}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Correlation (ρ)", f"{correlation_result.correlation:.3f}")
        
        with col2:
            st.metric("P-value", f"{correlation_result.p_value:.4f}")
        
        with col3:
            st.metric("Samples", correlation_result.n_samples)
        
        st.markdown(f"**Strength:** {correlation_result.strength}")
        st.markdown(
            f"**Significant:** {'Yes ✅' if correlation_result.is_significant else 'No ❌'} "
            f"(p < 0.05)"
        )
        
        # Interpretation
        st.markdown("**Interpretation:**")
        if abs(correlation_result.correlation) > 0.7:
            st.markdown("Strong correlation detected. Signals are highly related.")
        elif abs(correlation_result.correlation) < 0.3:
            st.markdown("Weak correlation detected. Signals are largely independent.")
        else:
            st.markdown("Moderate correlation detected. Some relationship exists.")


def render_compliance_dashboard(validation_result: ValidationResult) -> None:
    """
    Render comprehensive compliance dashboard with all validation checks.
    
    This is the main entry point for displaying validation results.
    Shows UDE score, condition checks, detailed statistics, and recommendations.
    """
    st.markdown("## 🎯 Validation Compliance Dashboard")
    st.markdown("---")
    
    # Top-level UDE score with color-coded badge
    render_ude_score_badge(validation_result)
    
    st.markdown("---")
    
    # Condition checks in a grid layout
    render_condition_grid(validation_result)
    
    st.markdown("---")
    
    # Detailed statistics in expandable sections
    render_detailed_statistics(validation_result)
    
    st.markdown("---")
    
    # Recommendations based on results
    render_recommendations(validation_result)


def render_ude_score_badge(validation_result: ValidationResult) -> None:
    """Render prominent UDE score badge with interpretation."""
    ude = validation_result.ude_score
    
    # Determine status and color
    if ude < 0.1:
        status = "Excellent"
        color = "#4CAF50"  # Green
        icon = "🌟"
    elif ude < 0.2:
        status = "Good"
        color = "#8BC34A"  # Light green
        icon = "✅"
    elif ude < 0.3:
        status = "Fair"
        color = "#FFC107"  # Amber
        icon = "⚠️"
    else:
        status = "Poor"
        color = "#F44336"  # Red
        icon = "❌"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
        border-left: 5px solid {color};
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h3 style="margin: 0; color: {color};">{icon} Uncertainty Disentanglement Error (UDE)</h3>
                <p style="margin: 5px 0 0 0; color: #666;">
                    Lower is better. Perfect disentanglement = 0.0
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 3em; font-weight: bold; color: {color};">
                    {ude:.3f}
                </div>
                <div style="font-size: 1.2em; color: {color}; font-weight: 600;">
                    {status}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_condition_grid(validation_result: ValidationResult) -> None:
    """Render 2x2 grid of condition checks."""
    st.markdown("### 📋 Validation Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔗 Consistency Conditions")
        st.caption("Signals should correlate with their sources")
        
        # C1: ua ~ Ua
        if validation_result.c1_result:
            render_condition_card(
                "C1",
                "Aleatoric ∼ Noise Level",
                validation_result.c1_result,
                validation_result.c1_pass,
                threshold=0.7,
                direction="positive"
            )
        
        # C2: ue ~ Ue
        if validation_result.c2_result:
            render_condition_card(
                "C2",
                "Epistemic ∼ Training Size",
                validation_result.c2_result,
                validation_result.c2_pass,
                threshold=0.7,
                direction="positive"
            )
    
    with col2:
        st.markdown("#### ⊥ Orthogonality Conditions")
        st.caption("Signals should be independent of other sources")
        
        # O1: ua ⊥ Ue
        if validation_result.o1_result:
            render_condition_card(
                "O1",
                "Aleatoric ⊥ Training Size",
                validation_result.o1_result,
                validation_result.o1_pass,
                threshold=0.3,
                direction="negative"
            )
        
        # O2: ue ⊥ Ua
        if validation_result.o2_result:
            render_condition_card(
                "O2",
                "Epistemic ⊥ Noise Level",
                validation_result.o2_result,
                validation_result.o2_pass,
                threshold=0.3,
                direction="negative"
            )


def render_condition_card(
    condition_id: str,
    description: str,
    result: CorrelationResult,
    passed: bool,
    threshold: float,
    direction: str
) -> None:
    """Render individual condition check card."""
    icon = "✅" if passed else "❌"
    color = "#4CAF50" if passed else "#F44336"
    
    # Format threshold text
    if direction == "positive":
        threshold_text = f"ρ > {threshold}"
    else:
        threshold_text = f"|ρ| < {threshold}"
    
    st.markdown(f"""
    <div style="
        background: {'#E8F5E9' if passed else '#FFEBEE'};
        border-left: 4px solid {color};
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong style="font-size: 1.1em;">{icon} {condition_id}</strong>
                <div style="color: #666; font-size: 0.9em; margin-top: 5px;">
                    {description}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.5em; font-weight: bold; color: {color};">
                    ρ = {result.correlation:.3f}
                </div>
                <div style="font-size: 0.8em; color: #666;">
                    {threshold_text}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_detailed_statistics(validation_result: ValidationResult) -> None:
    """Render detailed statistics in expandable sections."""
    st.markdown("### 📊 Detailed Statistics")
    
    results = [
        ("C1: Aleatoric ∼ Noise", validation_result.c1_result),
        ("C2: Epistemic ∼ Training", validation_result.c2_result),
        ("O1: Aleatoric ⊥ Training", validation_result.o1_result),
        ("O2: Epistemic ⊥ Noise", validation_result.o2_result),
    ]
    
    for label, result in results:
        if result:
            with st.expander(f"📈 {label}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Correlation (ρ)", f"{result.correlation:.4f}")
                
                with col2:
                    st.metric("P-value", f"{result.p_value:.4f}")
                    significance = "✅ Significant" if result.is_significant else "⚠️ Not significant"
                    st.caption(significance)
                
                with col3:
                    st.metric("Sample Size", result.n_samples)
                    st.caption(f"Strength: {result.strength}")
                
                # Interpretation
                st.markdown("**Interpretation:**")
                if result.is_significant:
                    if abs(result.correlation) > 0.7:
                        st.success(f"Strong {result.strength.lower()} correlation detected (p < 0.05)")
                    elif abs(result.correlation) > 0.4:
                        st.info(f"Moderate correlation detected (p < 0.05)")
                    else:
                        st.warning(f"Weak correlation detected (p < 0.05)")
                else:
                    st.warning("Correlation is not statistically significant (p ≥ 0.05)")


def render_recommendations(validation_result: ValidationResult) -> None:
    """Render actionable recommendations based on validation results."""
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    
    # Check C1
    if validation_result.c1_result and not validation_result.c1_pass:
        recommendations.append({
            "type": "warning",
            "title": "C1 Condition Failed",
            "message": "Aleatoric uncertainty signal does not correlate strongly with noise level.",
            "actions": [
                "Verify that label noise is properly injected into the dataset",
                "Check if aleatoric uncertainty estimation method is appropriate",
                "Consider increasing the range of noise levels in the sweep"
            ]
        })
    
    # Check C2
    if validation_result.c2_result and not validation_result.c2_pass:
        recommendations.append({
            "type": "warning",
            "title": "C2 Condition Failed",
            "message": "Epistemic uncertainty signal does not correlate strongly with training size.",
            "actions": [
                "Verify that training set size variation is sufficient",
                "Check if epistemic uncertainty estimation method is appropriate",
                "Consider using a wider range of training sizes"
            ]
        })
    
    # Check O1
    if validation_result.o1_result and not validation_result.o1_pass:
        recommendations.append({
            "type": "warning",
            "title": "O1 Condition Failed",
            "message": "Aleatoric uncertainty is not independent of training size.",
            "actions": [
                "This suggests aleatoric signal is contaminated with epistemic uncertainty",
                "Review the uncertainty decomposition method",
                "Consider using a different uncertainty quantification approach"
            ]
        })
    
    # Check O2
    if validation_result.o2_result and not validation_result.o2_pass:
        recommendations.append({
            "type": "warning",
            "title": "O2 Condition Failed",
            "message": "Epistemic uncertainty is not independent of noise level.",
            "actions": [
                "This suggests epistemic signal is contaminated with aleatoric uncertainty",
                "Review the uncertainty decomposition method",
                "Consider using a different uncertainty quantification approach"
            ]
        })
    
    # Success case
    if not recommendations:
        st.success("🎉 **All validation conditions passed!** Your uncertainty quantification method successfully disentangles epistemic and aleatoric uncertainty.")
        st.info("**Next Steps:**\n- Document these results\n- Consider testing on additional datasets\n- Explore edge cases and failure modes")
    else:
        for rec in recommendations:
            if rec["type"] == "warning":
                st.warning(f"**{rec['title']}**\n\n{rec['message']}")
            
            with st.expander("📋 Suggested Actions"):
                for action in rec["actions"]:
                    st.markdown(f"- {action}")


def render_full_validation_report(
    validation_result: ValidationResult,
    experiments: List[dict],
    epistemic_param: Optional[str] = None,
    aleatoric_param: Optional[str] = None
) -> None:
    """
    Render complete validation report with dashboard and correlation plots.
    
    This is the main entry point that combines the compliance dashboard
    with detailed correlation scatter plots.
    """
    # Main compliance dashboard
    render_compliance_dashboard(validation_result)
    
    st.markdown("---")
    st.markdown("## 📈 Correlation Analysis")
    
    # Extract data for plots
    if epistemic_param:
        epis_values = [float(exp['config'].get(epistemic_param, 0)) for exp in experiments if exp['config'].get(epistemic_param) is not None]
    else:
        epis_values = []
    
    if aleatoric_param:
        alea_values = [float(exp['config'].get(aleatoric_param, 0)) for exp in experiments if exp['config'].get(aleatoric_param) is not None]
    else:
        alea_values = []
    
    ue_values = [float(exp.get('epistemic_auroc', 0)) for exp in experiments if exp.get('epistemic_auroc') is not None]
    ua_values = [float(exp.get('aleatoric_auroc', 0)) for exp in experiments if exp.get('aleatoric_auroc') is not None]
    
    # Render correlation plots based on experiment type
    if validation_result.experiment_type == "epistemic":
        st.markdown("### Epistemic Sweep Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if validation_result.c2_result and epis_values and ue_values:
                st.markdown("#### C2: Epistemic ∼ Training Size")
                render_correlation_scatter(
                    epis_values, ue_values,
                    epistemic_param or "Training Size", "Epistemic AUROC",
                    validation_result.c2_result
                )
        
        with col2:
            if validation_result.o1_result and epis_values and ua_values:
                st.markdown("#### O1: Aleatoric ⊥ Training Size")
                render_correlation_scatter(
                    epis_values, ua_values,
                    epistemic_param or "Training Size", "Aleatoric AUROC",
                    validation_result.o1_result
                )
    
    elif validation_result.experiment_type == "aleatoric":
        st.markdown("### Aleatoric Sweep Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            if validation_result.c1_result and alea_values and ua_values:
                st.markdown("#### C1: Aleatoric ∼ Noise Level")
                render_correlation_scatter(
                    alea_values, ua_values,
                    aleatoric_param or "Noise Level", "Aleatoric AUROC",
                    validation_result.c1_result
                )
        
        with col2:
            if validation_result.o2_result and alea_values and ue_values:
                st.markdown("#### O2: Epistemic ⊥ Noise Level")
                render_correlation_scatter(
                    alea_values, ue_values,
                    aleatoric_param or "Noise Level", "Epistemic AUROC",
                    validation_result.o2_result
                )
    
    elif validation_result.experiment_type == "2d_grid":
        st.markdown("### 2D Grid Analysis")
        
        # All four plots in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            if validation_result.c1_result and alea_values and ua_values:
                st.markdown("#### C1: Aleatoric ∼ Noise")
                render_correlation_scatter(
                    alea_values, ua_values,
                    aleatoric_param or "Noise Level", "Aleatoric AUROC",
                    validation_result.c1_result
                )
            
            if validation_result.o1_result and epis_values and ua_values:
                st.markdown("#### O1: Aleatoric ⊥ Training")
                render_correlation_scatter(
                    epis_values, ua_values,
                    epistemic_param or "Training Size", "Aleatoric AUROC",
                    validation_result.o1_result
                )
        
        with col2:
            if validation_result.c2_result and epis_values and ue_values:
                st.markdown("#### C2: Epistemic ∼ Training")
                render_correlation_scatter(
                    epis_values, ue_values,
                    epistemic_param or "Training Size", "Epistemic AUROC",
                    validation_result.c2_result
                )
            
            if validation_result.o2_result and alea_values and ue_values:
                st.markdown("#### O2: Epistemic ⊥ Noise")
                render_correlation_scatter(
                    alea_values, ue_values,
                    aleatoric_param or "Noise Level", "Epistemic AUROC",
                    validation_result.o2_result
                )

# Made with Bob
