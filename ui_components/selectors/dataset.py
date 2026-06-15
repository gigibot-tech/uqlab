"""
Dataset Selection and Configuration UI Components

This module contains UI components for dataset selection, configuration,
and comparison visualization.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable


def render_dataset_selection(
    dataset_name: str,
    noise_type: str,
    stats_fetcher: Callable[[str, str], Optional[Dict]]
) -> Tuple[str, str, Optional[Dict]]:
    """
    Render dataset selection section at the top of the page.
    
    Args:
        dataset_name: Default dataset name
        noise_type: Default noise type
        stats_fetcher: Function to fetch dataset statistics
    
    Returns:
        Tuple of (dataset_name, noise_type, stats_dict)
    """
    col_selector, col_description = st.columns([1, 2])
    
    with col_selector:
        st.markdown("**Base Dataset:** CIFAR-10 (clean)")
        st.caption("50,000 training images, 10 classes")
        
        st.markdown("---")
        
        st.markdown("**CIFAR-10N Reference** (optional)")
        st.caption("View pre-existing noise patterns for comparison")
        
        noise_type = st.selectbox(
            "Noise Type (reference only)",
            ["worse_label", "aggre_label", "random_label1", "random_label2", "random_label3"],
            help="CIFAR-10N noise patterns - shown for reference. Use 'Aleatoric Uncertainty' section below to add noise."
        )
    
    with col_description:
        st.success("""
        **✅ Training Dataset: Clean CIFAR-10**
        - 📦 50,000 training images across 10 classes
        - ✨ Clean labels (no noise by default)
        - 🏷️ Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        
        **💡 To add label noise:** Configure in "Aleatoric Uncertainty" section below
        """)
        
        # Fetch CIFAR-10N stats for reference only
        dataset_name = "cifar10n"  # Fixed - only for stats reference
        stats = stats_fetcher(dataset_name, noise_type)
        if stats:
            noise_rate = stats.get('noise_rate', 0) * 100
            st.caption(f"📊 CIFAR-10N '{noise_type}' reference: {noise_rate:.1f}% noise")
    
    st.markdown("---")
    
    return dataset_name, noise_type, stats


def render_dataset_comparison(
    under_supported: str,
    under_train_per_class: int,
    regular_train_per_class: int,
    noise_source: str,
    stats: Optional[Dict],
    noise_type: str,
    custom_noise_rate: float,
    class_names: List[str],
    eval_per_group: int
) -> None:
    """
    Render dynamic dataset calculations based on user configuration.
    All values are computed from actual configuration parameters - no hardcoded examples.
    
    Args:
        under_supported: Under-supported classes specification
        under_train_per_class: Samples per under-supported class
        regular_train_per_class: Samples per regular class
        noise_source: Source of noise
        stats: Dataset statistics (only for total dataset info)
        noise_type: Type of noise pattern
        custom_noise_rate: Custom noise rate
        class_names: List of class names
        eval_per_group: Samples per evaluation group
    """
    st.markdown("---")
    st.markdown("### 📊 Dataset Configuration (Fully Dynamic)")
    
    # Parse under-supported classes and get actual indices
    if under_supported.startswith("random:"):
        num_under = int(under_supported.split(":")[1])
        under_supported_list = []  # Random selection, indices not yet determined
        under_classes_display = f"{num_under} randomly selected classes"
    else:
        under_supported_list = [int(x.strip()) for x in under_supported.split(",") if x.strip()]
        num_under = len(under_supported_list)
        # Show actual class names
        if under_supported_list:
            class_display = ", ".join([f"{i} ({class_names[i]})" for i in under_supported_list])
            under_classes_display = class_display
        else:
            under_classes_display = "None selected"
    
    num_regular = 10 - num_under
    
    # Get actual noise rate from configuration
    if noise_source.startswith("Use CIFAR-10N noise") and stats:
        noise_rate = stats.get('noise_rate', 0)
        noise_source_display = f"CIFAR-10N {noise_type}"
    else:
        noise_rate = custom_noise_rate / 100.0
        noise_source_display = f"Random label flipping at {custom_noise_rate}%"
    
    # Calculate training dataset sizes dynamically
    under_samples = num_under * under_train_per_class
    regular_clean_samples = int(num_regular * regular_train_per_class * (1 - noise_rate))
    regular_noisy_samples = int(num_regular * regular_train_per_class * noise_rate)
    total_regular_samples = num_regular * regular_train_per_class
    total_train = under_samples + total_regular_samples
    
    # Calculate evaluation dataset sizes dynamically
    total_eval = 3 * eval_per_group
    
    st.info("""
    **CIFAR-10 Base Dataset:**
    - Training: 50,000 samples (5,000 per class)
    - Test: 10,000 samples (1,000 per class)
    
    Your configuration creates custom training and evaluation splits from this base.
    """)
    
    # Show selected classes
    st.markdown("#### 🎯 Class Distribution")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        **Under-supported Classes ({num_under}):**
        {under_classes_display}
        """)
    with col_b:
        st.markdown(f"""
        **Regular Classes ({num_regular}):**
        All remaining classes
        """)
    
    # Training Dataset Section
    st.markdown("#### 🎯 Training Dataset (Based on Your Configuration)")
    st.markdown(f"""
    **Formula:** `Total = (Under-supported samples) + (Regular samples)`
    
    **Calculation:**
    ```
    Under-supported: {num_under} classes × {under_train_per_class} samples = {under_samples:,} samples
    Regular classes: {num_regular} classes × {regular_train_per_class} samples = {total_regular_samples:,} samples
      ├─ Clean: {total_regular_samples:,} × {(1-noise_rate):.1%} = {regular_clean_samples:,} samples
      └─ Noisy: {total_regular_samples:,} × {noise_rate:.1%} = {regular_noisy_samples:,} samples
    
    Total Training = {under_samples:,} + {total_regular_samples:,} = {total_train:,} samples
    ```
    
    **Noise Source:** {noise_source_display}
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Training", f"{total_train:,}")
    with col2:
        st.metric("Under-supported", f"{under_samples:,}",
                  delta=f"{num_under} classes")
    with col3:
        st.metric("Regular (Clean+Noisy)", f"{total_regular_samples:,}",
                  delta=f"{num_regular} classes")
    
    # Evaluation Dataset Section
    st.markdown("#### 📊 Evaluation Dataset (3 Groups)")
    st.markdown(f"""
    **Formula:** `Total = 3 × {eval_per_group:,} samples per group`
    
    **Group Assignment:**
    ```
    🟢 Clean Group:      {eval_per_group:,} samples from regular classes (clean labels)
    🟡 Aleatoric Group:  {eval_per_group:,} samples from regular classes (noisy labels)
    🔴 Epistemic Group:  {eval_per_group:,} samples from under-supported classes (clean labels)
    
    Total Evaluation = 3 × {eval_per_group:,} = {total_eval:,} samples
    ```
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Evaluation", f"{total_eval:,}")
    with col2:
        st.metric("🟢 Clean", f"{eval_per_group:,}")
    with col3:
        st.metric("🟡 Aleatoric", f"{eval_per_group:,}")
    with col4:
        st.metric("🔴 Epistemic", f"{eval_per_group:,}")
    
    st.warning("""
    ⚠️ **Important**: Evaluation samples are **separate from training** and never overlap.
    This ensures unbiased measurement of uncertainty detection capabilities.
    """)

# Made with Bob
