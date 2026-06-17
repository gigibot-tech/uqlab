"""
Training Data Inspection UI Components

This module provides UI components for inspecting training data,
including label flip statistics and sample-level inspection tables.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def parse_training_data_stats(experiment_id: str) -> Optional[Dict]:
    """
    Parse training data statistics from experiment results.
    
    Args:
        experiment_id: Experiment ID
        
    Returns:
        Dictionary with training data statistics, or None if not available
    """
    # Try multiple possible locations for training data
    possible_paths = [
        Path(f"data/experiments/{experiment_id}/results/training_data.csv"),
        Path(f"data/experiments/{experiment_id}/results/per_sample_signals.csv"),
        Path(f"/tmp/uqlab_experiments/{experiment_id}/results/training_data.csv"),
        Path(f"/tmp/uqlab_experiments/{experiment_id}/results/per_sample_signals.csv"),
    ]
    
    train_data_file = None
    for path in possible_paths:
        if path.exists():
            train_data_file = path
            break
    
    if not train_data_file:
        return None
    
    try:
        # Load training data
        df = pd.read_csv(train_data_file)
        
        # Validate required columns
        required_cols = ['clean_label', 'noisy_label']
        if not all(col in df.columns for col in required_cols):
            return None
        
        # Calculate statistics
        total_samples = len(df)
        
        # Check if is_noisy column exists, otherwise infer from labels
        if 'is_noisy' in df.columns:
            noisy_samples = df['is_noisy'].sum()
        else:
            # Infer noisy samples from label mismatch
            df['is_noisy'] = df['clean_label'] != df['noisy_label']
            noisy_samples = df['is_noisy'].sum()
        
        clean_samples = total_samples - noisy_samples
        noise_rate = noisy_samples / total_samples if total_samples > 0 else 0
        
        # Per-class statistics
        class_stats = []
        for class_idx in range(10):  # CIFAR-10 has 10 classes
            class_mask = df['clean_label'] == class_idx
            class_total = class_mask.sum()
            class_noisy = ((class_mask) & (df['is_noisy'])).sum()
            
            class_stats.append({
                'class_idx': class_idx,
                'class_name': CIFAR10_CLASSES[class_idx],
                'total_samples': int(class_total),
                'clean_samples': int(class_total - class_noisy),
                'noisy_samples': int(class_noisy),
                'noise_rate': class_noisy / class_total if class_total > 0 else 0
            })
        
        return {
            'total_samples': int(total_samples),
            'clean_samples': int(clean_samples),
            'noisy_samples': int(noisy_samples),
            'noise_rate': float(noise_rate),
            'class_stats': class_stats,
            'samples_df': df  # Full dataframe for table display
        }
    
    except Exception as e:
        st.error(f"Error parsing training data: {str(e)}")
        return None


def render_training_data_stats(experiment_id: str) -> None:
    """
    Render training data statistics and inspection table.
    
    Args:
        experiment_id: Experiment ID
    """
    st.markdown("### 📊 Training Data Inspection")
    
    # Parse training data
    train_stats = parse_training_data_stats(experiment_id)
    
    if not train_stats:
        st.info("Training data statistics not available for this experiment")
        return
    
    # Overall statistics
    st.markdown("#### Overall Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{train_stats['total_samples']:,}")
    with col2:
        st.metric("Clean Samples", f"{train_stats['clean_samples']:,}")
    with col3:
        st.metric("Noisy Samples", f"{train_stats['noisy_samples']:,}")
    with col4:
        st.metric("Noise Rate", f"{train_stats['noise_rate']:.1%}")
    
    # Per-class statistics
    st.markdown("#### Per-Class Statistics")
    
    class_stats_df = pd.DataFrame(train_stats['class_stats'])
    
    # Format for display
    display_df = class_stats_df[['class_name', 'total_samples', 'clean_samples', 'noisy_samples', 'noise_rate']].copy()
    display_df.columns = ['Class', 'Total', 'Clean', 'Noisy', 'Noise %']
    display_df['Noise %'] = display_df['Noise %'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Sample-level inspection table
    st.markdown("#### Sample-Level Inspection")
    st.caption("Scrollable table showing clean vs noisy labels for each training sample")
    
    samples_df = train_stats['samples_df']
    
    # Prepare display dataframe
    if 'is_noisy' in samples_df.columns:
        # Determine index column name
        index_col = None
        for possible_col in ['dataset_index', 'index', 'sample_index']:
            if possible_col in samples_df.columns:
                index_col = possible_col
                break
        
        if index_col:
            display_samples = samples_df[[index_col, 'clean_label', 'noisy_label', 'is_noisy']].copy()
            display_samples.columns = ['Sample Index', 'Clean Label', 'Noisy Label', 'Is Noisy']
        else:
            # Use dataframe index as sample index
            display_samples = samples_df[['clean_label', 'noisy_label', 'is_noisy']].copy()
            display_samples.insert(0, 'Sample Index', samples_df.index)
            display_samples.columns = ['Sample Index', 'Clean Label', 'Noisy Label', 'Is Noisy']
        
        # Add class names
        display_samples['Clean Class'] = display_samples['Clean Label'].apply(
            lambda x: CIFAR10_CLASSES[int(x)] if 0 <= x < 10 else 'Unknown'
        )
        display_samples['Noisy Class'] = display_samples['Noisy Label'].apply(
            lambda x: CIFAR10_CLASSES[int(x)] if 0 <= x < 10 else 'Unknown'
        )
        
        # Reorder columns
        display_samples = display_samples[['Sample Index', 'Clean Label', 'Clean Class', 'Noisy Label', 'Noisy Class', 'Is Noisy']]
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_only_noisy = st.checkbox("Show only noisy samples", value=False, key=f"show_noisy_{experiment_id}")
        with col2:
            selected_class = st.selectbox(
                "Filter by clean class",
                options=["All"] + CIFAR10_CLASSES,
                index=0,
                key=f"filter_class_{experiment_id}"
            )
        
        # Apply filters
        filtered_df = display_samples.copy()
        if show_only_noisy:
            filtered_df = filtered_df[filtered_df['Is Noisy'] == True]
        if selected_class != "All":
            class_idx = CIFAR10_CLASSES.index(selected_class)
            filtered_df = filtered_df[filtered_df['Clean Label'] == class_idx]
        
        st.caption(f"Showing {len(filtered_df):,} of {len(display_samples):,} samples")
        
        # Display table with scrolling
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"training_data_{experiment_id[:8]}.csv",
            mime="text/csv",
            key=f"download_train_{experiment_id}"
        )
    else:
        st.info("Detailed sample information not available")


def generate_training_stats_from_config(experiment_config: Dict) -> Optional[Dict]:
    """
    Generate training statistics from experiment configuration.
    
    This is a fallback when actual training data file is not available.
    
    Args:
        experiment_config: Experiment configuration dictionary
        
    Returns:
        Dictionary with estimated training data statistics
    """
    try:
        # Extract configuration
        under_supported = experiment_config.get('under_supported', 'random:2')
        under_train_per_class = experiment_config.get('under_train_per_class', 50)
        regular_train_per_class = experiment_config.get('regular_train_per_class', 300)
        noise_rate = experiment_config.get('aleatoric_noise_percentage', 0) / 100.0
        
        # Parse under-supported classes
        if under_supported.startswith('random:'):
            num_under = int(under_supported.split(':')[1])
            under_classes = list(range(num_under))  # Placeholder
        else:
            under_classes = [int(x.strip()) for x in under_supported.split(',')]
        
        # Calculate statistics
        total_samples = (len(under_classes) * under_train_per_class + 
                        (10 - len(under_classes)) * regular_train_per_class)
        
        # Estimate noisy samples (only in regular classes)
        regular_samples = (10 - len(under_classes)) * regular_train_per_class
        noisy_samples = int(regular_samples * noise_rate)
        clean_samples = total_samples - noisy_samples
        
        return {
            'total_samples': total_samples,
            'clean_samples': clean_samples,
            'noisy_samples': noisy_samples,
            'noise_rate': noisy_samples / total_samples if total_samples > 0 else 0,
            'estimated': True  # Flag that this is estimated, not actual
        }
    except Exception:
        return None


# Made with Bob