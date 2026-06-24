"""Per-class configuration table for Step 3 uncertainty configuration.

This module provides an editable table UI for configuring training samples,
label noise, and sweep participation for each CIFAR-10 class individually.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd

from uqlab_orchestrator.config import PerClassConfig


# CIFAR-10 class names
CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def _create_default_per_class_config() -> Dict[int, PerClassConfig]:
    """Create default per-class configuration (Four-Region Partition).
    
    Returns:
        Dict mapping class ID to PerClassConfig with four-region settings:
        - Noisy (0-3): 300 samples, 30% label noise → aleatoric uncertainty
        - Sparse (4-5): 30 samples (10% of 300), 0% noise → epistemic uncertainty
        - Clean (6-7): 300 samples, 0% noise → low uncertainty baseline
        - OOD (8-9): 0 samples, 0% noise → out-of-distribution (withheld from training)
    """
    config = {}
    
    for class_id in range(10):
        if 0 <= class_id <= 3:  # Noisy region
            config[class_id] = PerClassConfig(
                train_samples=300,
                label_noise_pct=30.0,
                sweep_epistemic=False,
                sweep_aleatoric=False,
            )
        elif 4 <= class_id <= 5:  # Sparse region
            config[class_id] = PerClassConfig(
                train_samples=30,  # 10% of 300
                label_noise_pct=0.0,
                sweep_epistemic=False,
                sweep_aleatoric=False,
            )
        elif 6 <= class_id <= 7:  # Clean region
            config[class_id] = PerClassConfig(
                train_samples=300,
                label_noise_pct=0.0,
                sweep_epistemic=False,
                sweep_aleatoric=False,
            )
        else:  # OOD region (8-9)
            config[class_id] = PerClassConfig(
                train_samples=0,  # Withheld from training
                label_noise_pct=0.0,
                sweep_epistemic=False,
                sweep_aleatoric=False,
            )
    
    return config


def _create_balanced_config() -> Dict[int, PerClassConfig]:
    """Create balanced per-class configuration.
    
    Returns:
        Dict with all classes having 300 samples, 0% noise, no sweeps
    """
    return {
        class_id: PerClassConfig(
            train_samples=300,
            label_noise_pct=0.0,
            sweep_epistemic=False,
            sweep_aleatoric=False,
        )
        for class_id in range(10)
    }


def _config_to_dataframe(config: Dict[int, PerClassConfig]) -> pd.DataFrame:
    """Convert per-class config dict to pandas DataFrame for display.
    
    Args:
        config: Dict mapping class ID to PerClassConfig
        
    Returns:
        DataFrame with columns: ID, Class, Train Samples, Label Noise %, 
        Sweep Epistemic, Sweep Aleatoric
    """
    rows = []
    for class_id in range(10):
        cfg = config.get(class_id, PerClassConfig())
        rows.append({
            "ID": class_id,
            "Class": CIFAR10_CLASS_NAMES[class_id],
            "Train Samples": cfg.train_samples,
            "Label Noise %": cfg.label_noise_pct,
            "Sweep Epistemic": cfg.sweep_epistemic,
            "Sweep Aleatoric": cfg.sweep_aleatoric,
        })
    
    return pd.DataFrame(rows)


def _dataframe_to_config(df: pd.DataFrame) -> Dict[int, PerClassConfig]:
    """Convert edited DataFrame back to per-class config dict.
    
    Args:
        df: DataFrame with edited values
        
    Returns:
        Dict mapping class ID to PerClassConfig
    """
    config = {}
    for _, row in df.iterrows():
        class_id = int(row["ID"])
        config[class_id] = PerClassConfig(
            train_samples=int(row["Train Samples"]),
            label_noise_pct=float(row["Label Noise %"]),
            sweep_epistemic=bool(row["Sweep Epistemic"]),
            sweep_aleatoric=bool(row["Sweep Aleatoric"]),
        )
    
    return config


def render_per_class_table(
    session_key: str = "per_class_config",
) -> Tuple[Dict[int, PerClassConfig], bool]:
    """Render editable per-class configuration table.
    
    Args:
        session_key: Session state key for storing config
        
    Returns:
        Tuple of (per_class_config dict, config_changed flag)
    """
    st.markdown("### 📊 Per-Class Configuration")
    st.caption(
        "Configure training samples, label noise, and sweep participation "
        "for each class individually"
    )
    
    # Initialize session state if needed
    if session_key not in st.session_state:
        st.session_state[session_key] = _create_default_per_class_config()
    
    # Preset buttons
    col1, col2, col3 = st.columns([3, 3, 6])
    
    with col1:
        if st.button(
            "🎯 Four-Region Default",
            help="Noisy (0-3): 300 samples, 30% noise | Sparse (4-5): 30 samples | Clean (6-7): 300 samples | OOD (8-9): 0 samples",
            use_container_width=True
        ):
            st.session_state[session_key] = _create_default_per_class_config()
            st.rerun()
    
    with col2:
        if st.button(
            "⚖️ Balanced",
            help="All classes: 300 samples, 0% noise",
            use_container_width=True
        ):
            st.session_state[session_key] = _create_balanced_config()
            st.rerun()
    
    # Convert config to DataFrame for editing
    current_config = st.session_state[session_key]
    df = _config_to_dataframe(current_config)
    
    # Display editable table
    st.markdown("#### Edit Configuration")
    st.caption("💡 Click cells to edit values. Check boxes to enable sweeps for that class.")
    
    edited_df = st.data_editor(
        df,
        column_config={
            "ID": st.column_config.NumberColumn(
                "ID",
                help="Class ID (0-9)",
                disabled=True,
                width="small",
            ),
            "Class": st.column_config.TextColumn(
                "Class",
                help="CIFAR-10 class name",
                disabled=True,
                width="medium",
            ),
            "Train Samples": st.column_config.NumberColumn(
                "Train Samples",
                help="Number of training samples for this class",
                min_value=0,
                max_value=5000,
                step=10,
                width="medium",
            ),
            "Label Noise %": st.column_config.NumberColumn(
                "Label Noise %",
                help="Label noise percentage (0-100) for this class",
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                format="%.1f",
                width="medium",
            ),
            "Sweep Epistemic": st.column_config.CheckboxColumn(
                "Sweep Epistemic",
                help="Enable epistemic uncertainty sweep for this class",
                width="small",
            ),
            "Sweep Aleatoric": st.column_config.CheckboxColumn(
                "Sweep Aleatoric",
                help="Enable aleatoric uncertainty sweep for this class",
                width="small",
            ),
        },
        hide_index=True,
        use_container_width=True,
        key=f"{session_key}_editor",
    )
    
    # Convert edited DataFrame back to config
    new_config = _dataframe_to_config(edited_df)
    
    # Check if config changed
    config_changed = new_config != current_config
    
    # Update session state if changed
    if config_changed:
        st.session_state[session_key] = new_config
    
    # Display summary statistics
    st.markdown("#### Configuration Summary")
    
    total_samples = sum(cfg.train_samples for cfg in new_config.values())
    sparse_classes = [
        class_id for class_id, cfg in new_config.items()
        if cfg.train_samples < 200
    ]
    noisy_classes = [
        class_id for class_id, cfg in new_config.items()
        if cfg.label_noise_pct > 0
    ]
    epistemic_sweep_classes = [
        class_id for class_id, cfg in new_config.items()
        if cfg.sweep_epistemic
    ]
    aleatoric_sweep_classes = [
        class_id for class_id, cfg in new_config.items()
        if cfg.sweep_aleatoric
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Training Samples", f"{total_samples:,}")
        st.caption(f"Sparse classes (<200): {len(sparse_classes)}")
    
    with col2:
        avg_noise = sum(cfg.label_noise_pct for cfg in new_config.values()) / 10
        st.metric("Avg Label Noise", f"{avg_noise:.1f}%")
        st.caption(f"Noisy classes (>0%): {len(noisy_classes)}")
    
    with col3:
        st.metric("Sweep Classes", f"{len(epistemic_sweep_classes) + len(aleatoric_sweep_classes)}")
        st.caption(
            f"Epistemic: {len(epistemic_sweep_classes)}, "
            f"Aleatoric: {len(aleatoric_sweep_classes)}"
        )
    
    # Display sweep details if any sweeps enabled
    if epistemic_sweep_classes or aleatoric_sweep_classes:
        with st.expander("🔍 Sweep Details", expanded=False):
            if epistemic_sweep_classes:
                st.markdown("**Epistemic Sweep Classes:**")
                for class_id in epistemic_sweep_classes:
                    cfg = new_config[class_id]
                    st.write(
                        f"- Class {class_id} ({CIFAR10_CLASS_NAMES[class_id]}): "
                        f"{cfg.train_samples} samples"
                    )
            
            if aleatoric_sweep_classes:
                st.markdown("**Aleatoric Sweep Classes:**")
                for class_id in aleatoric_sweep_classes:
                    cfg = new_config[class_id]
                    st.write(
                        f"- Class {class_id} ({CIFAR10_CLASS_NAMES[class_id]}): "
                        f"{cfg.label_noise_pct:.1f}% noise"
                    )
    
    return new_config, config_changed


def get_per_class_config_summary(config: Dict[int, PerClassConfig]) -> str:
    """Generate a human-readable summary of per-class configuration.
    
    Args:
        config: Per-class configuration dict
        
    Returns:
        Summary string for display in collapsed step view
    """
    total_samples = sum(cfg.train_samples for cfg in config.values())
    sparse_count = sum(1 for cfg in config.values() if cfg.train_samples < 200)
    avg_noise = sum(cfg.label_noise_pct for cfg in config.values()) / 10
    
    epistemic_count = sum(1 for cfg in config.values() if cfg.sweep_epistemic)
    aleatoric_count = sum(1 for cfg in config.values() if cfg.sweep_aleatoric)
    
    parts = [
        f"{total_samples:,} total samples",
        f"{sparse_count} sparse classes",
        f"{avg_noise:.1f}% avg noise",
    ]
    
    if epistemic_count > 0:
        parts.append(f"{epistemic_count} epistemic sweeps")
    if aleatoric_count > 0:
        parts.append(f"{aleatoric_count} aleatoric sweeps")
    
    return ", ".join(parts)


# Example usage for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Per-Class Config Test", layout="wide")
    st.title("Per-Class Configuration Table Test")
    
    config, changed = render_per_class_table()
    
    if changed:
        st.success("✅ Configuration updated!")
    
    st.markdown("---")
    st.markdown("### Current Configuration (JSON)")
    st.json({
        str(k): {
            "train_samples": v.train_samples,
            "label_noise_pct": v.label_noise_pct,
            "sweep_epistemic": v.sweep_epistemic,
            "sweep_aleatoric": v.sweep_aleatoric,
        }
        for k, v in config.items()
    })

# Made with Bob
