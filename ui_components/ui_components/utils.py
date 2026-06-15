"""
Utility UI Components

This module contains utility functions and helper components used across
the application, including progress tracking and ROC explanation.
"""

import streamlit as st
import pandas as pd
from typing import Dict


def render_configuration_progress(
    config_progress: Dict[str, bool],
    api_base_url: str
) -> None:
    """
    Render configuration progress sidebar with checkmarks and progress bar.
    
    Args:
        config_progress: Dictionary with completion status for each section
        api_base_url: API endpoint URL to display
    """
    dataset_icon = "✅" if config_progress['dataset'] else "⭕"
    epistemic_icon = "✅" if config_progress['epistemic'] else "⭕"
    aleatoric_icon = "✅" if config_progress['aleatoric'] else "⭕"
    model_icon = "✅" if config_progress['model'] else "⭕"
    evaluation_icon = "✅" if config_progress['evaluation'] else "⭕"
    
    st.markdown(f"{dataset_icon} Dataset")
    st.markdown(f"{epistemic_icon} Epistemic")
    st.markdown(f"{aleatoric_icon} Aleatoric")
    st.markdown(f"{model_icon} Model")
    st.markdown(f"{evaluation_icon} Evaluation")
    
    # Calculate overall completion
    completed = sum(config_progress.values())
    total = len(config_progress)
    progress_pct = completed / total
    
    st.progress(progress_pct)
    st.caption(f"{int(progress_pct * 100)}% Complete")
    
    st.markdown("---")
    st.markdown(f"**API Endpoint:** `{api_base_url}`")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()


def render_roc_explanation(
    under_supported: str,
    class_names: list[str],
    noise_source: str,
    custom_noise_rate: float
) -> None:
    """
    Render ROC calculation walkthrough with crystal-clear configuration-based explanation.
    
    Args:
        under_supported: Under-supported classes configuration
        class_names: List of class names for display
        noise_source: Noise source selection ("Use CIFAR-10N noise" or "Random noise")
        custom_noise_rate: Custom noise rate if random noise is used
    """
    with st.expander("📐 Understanding AUROC Calculation (Click to Expand)"):
        st.markdown("### How AUROC Measures Uncertainty Detection")
        
        # Parse under-supported classes for display
        if under_supported.startswith("random:"):
            num_under = int(under_supported.split(":")[1])
            under_classes_display = f"random {num_under} classes"
            under_indices = []
        else:
            under_indices = [int(idx.strip()) for idx in under_supported.split(",") if idx.strip()]
            num_under = len(under_indices)
            under_classes_display = ", ".join([f"{class_names[idx]}" for idx in under_indices])
        
        num_regular = 10 - num_under
        
        # Configuration context
        st.info("""
        **AUROC (Area Under ROC Curve)** measures how well uncertainty signals can
        **rank samples by their uncertainty** based on your specific configuration.
        """)
        
        # Display configuration context
        st.markdown("#### 🎯 AUROC Calculation Based on Your Configuration")
        
        noise_desc = "CIFAR-10N synthetic noise" if noise_source.startswith("Use CIFAR-10N noise") else f"{custom_noise_rate}% random label flipping"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🟡 Aleatoric AUROC** (Detecting Noisy Labels)
            
            **Positive Class (should have HIGH uncertainty):**
            - 🟡 Aleatoric group
            - Regular classes + noisy labels
            - {num_regular} well-supported classes with {noise_desc}
            
            **Negative Class (should have LOW uncertainty):**
            - 🟢 Clean group (regular classes, clean labels)
            - 🔴 Epistemic group (under-supported classes, clean labels)
            
            **Question:** Can we detect mislabeled data in well-supported classes?
            """)
        
        with col2:
            st.markdown(f"""
            **🔴 Epistemic AUROC** (Detecting Unfamiliar Classes)
            
            **Positive Class (should have HIGH uncertainty):**
            - 🔴 Epistemic group
            - Under-supported classes: {under_classes_display}
            - Clean labels but model undertrained
            
            **Negative Class (should have LOW uncertainty):**
            - 🟢 Clean group (regular classes, clean labels)
            - 🟡 Aleatoric group (regular classes, noisy labels)
            
            **Question:** Can we detect samples from undertrained classes?
            """)
        
        st.warning("""
        ⚠️ **Key Point**: AUROC scores depend entirely on your configuration.
        Different under-supported classes or noise rates will produce different results.
        """)
        
        st.markdown("---")
        
        # Step-by-step explanation
        st.markdown("#### 📊 How AUROC is Calculated")
        
        st.markdown("**Step 1: Compute Uncertainty Scores**")
        st.code("""
# For each evaluation sample, compute uncertainty signal
for sample in evaluation_dataset:
    uncertainty_score = model.compute_uncertainty(sample)
    
# Higher score = more uncertain
# Example:
#   Clean sample (regular class):     0.15 (low uncertainty)
#   Aleatoric sample (noisy label):   0.82 (high uncertainty)
#   Epistemic sample (under-trained): 0.91 (high uncertainty)
        """, language="python")
        
        st.markdown("**Step 2: Rank Samples by Uncertainty**")
        
        # Sample data preview
        sample_data = pd.DataFrame({
            'Sample': ['img_001', 'img_002', 'img_003', 'img_004', 'img_005', 'img_006'],
            'True Group': ['🟢 Clean', '🟡 Aleatoric', '🔴 Epistemic', '🟢 Clean', '🟡 Aleatoric', '🔴 Epistemic'],
            'Uncertainty': [0.15, 0.82, 0.91, 0.19, 0.78, 0.88],
            'Rank': [1, 4, 6, 2, 3, 5]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("**Step 3: Calculate AUROC**")
        
        st.markdown("""
        AUROC measures: **"What fraction of positive samples are ranked higher than negative samples?"**
        
        For **Aleatoric AUROC**:
        - Positive = 🟡 Aleatoric (should rank high)
        - Negative = 🟢 Clean + 🔴 Epistemic (should rank low)
        - Perfect score: All aleatoric samples ranked above all clean/epistemic
        
        For **Epistemic AUROC**:
        - Positive = 🔴 Epistemic (should rank high)
        - Negative = 🟢 Clean + 🟡 Aleatoric (should rank low)
        - Perfect score: All epistemic samples ranked above all clean/aleatoric
        """)
        
        st.markdown("#### 📈 Interpretation")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.success("""
            **AUROC Scores:**
            - **1.0**: Perfect detection
            - **0.9**: Excellent (90% correct ranking)
            - **0.7**: Good (70% correct ranking)
            - **0.5**: Random guessing (no signal)
            """)
        with col_b:
            st.info("""
            **What This Means:**
            - High Aleatoric AUROC → Can detect bad labels
            - High Epistemic AUROC → Can detect knowledge gaps
            - Compare signals to find best uncertainty measure
            """)
        
        st.markdown("---")
        st.markdown("#### 💡 Practical Example")
        st.code("""
# Your configuration creates these groups:
Clean:      600 samples from regular classes (clean)
Aleatoric:  600 samples from regular classes (noisy)
Epistemic:  600 samples from under-supported classes (clean)

# Model computes uncertainty for all 1800 samples
# AUROC tells us: "Can uncertainty scores separate the groups?"

# Good Aleatoric AUROC (0.85):
#   → 85% of noisy samples ranked higher than clean samples
#   → Uncertainty signal detects mislabeled data well

# Good Epistemic AUROC (0.92):
#   → 92% of under-supported samples ranked higher
#   → Uncertainty signal detects unfamiliar classes well
        """, language="python")

# Made with Bob
