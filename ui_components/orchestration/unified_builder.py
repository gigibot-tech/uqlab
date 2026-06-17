"""
Unified Experiment Builder UI Component

This module implements the unified experiment builder that replaces the 4-tab system
with a single, dynamic configuration view where users can enable sweeps via checkboxes.

Phase A of the UX redesign plan.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable

from ..config.experiment_config import (
    build_base_experiment_config,
    render_model_config,
    render_training_config,
    render_evaluation_config,
)
from ..config.config_types import (
    ValidationConfig,
    SweepConfig,
    EpistemicConfig,
    AleatoricConfig,
    UnifiedBuilderConfig,
)


def render_unified_builder(
    dataset_name: str,
    noise_type: str,
    stats: Optional[Dict],
    class_names: List[str],
    fetch_dataset_stats: Callable,
    api_base_url: str,
    get_headers: Callable
) -> Optional[UnifiedBuilderConfig]:
    """
    Render the unified experiment builder interface.
    
    This is the main entry point for the unified builder that combines
    single experiment and sweep configuration into one dynamic view.
    
    Args:
        dataset_name: Name of the dataset (e.g., "cifar10n")
        noise_type: Type of noise (e.g., "worse_label")
        stats: Dataset statistics dictionary
        class_names: List of class names
        fetch_dataset_stats: Function to fetch dataset statistics
        api_base_url: Base URL for API calls
        get_headers: Function to get request headers
    
    Returns:
        UnifiedBuilderConfig if form submitted, None otherwise
    """
    st.markdown("## 🚀 Custom experiments (API)")
    st.caption(
        "Preset hypothesis sweeps (same as the Hypothesis Validation tab) run locally and "
        "write to `results/validation/`. Below: custom epistemic/aleatoric grids via the API."
    )

    from ..orchestration.validation_runner import render_preset_validation_sweeps

    with st.expander("Preset validation sweeps (recommended first)", expanded=True):
        render_preset_validation_sweeps(key_prefix="ub_preset")

    st.markdown("---")

    with st.form("unified_builder_form"):
        # ========== A. DATASET CONFIGURATION (COMPACT) ==========
        st.markdown("### 📊 Dataset Configuration")
        render_dataset_config_compact(dataset_name, stats, class_names)
        
        st.markdown("---")
        
        # ========== B. UNCERTAINTY CONFIGURATION (CORE INNOVATION) ==========
        st.markdown("### 🎯 Uncertainty Configuration")
        
        col_epistemic, col_aleatoric = st.columns(2)
        
        # Epistemic Section
        with col_epistemic:
            epistemic_config = render_epistemic_with_sweep(class_names)
        
        # Aleatoric Section
        with col_aleatoric:
            aleatoric_config = render_aleatoric_with_sweep(
                stats, noise_type, class_names, fetch_dataset_stats, dataset_name
            )
        
        st.markdown("---")
        
        # ========== C. MODEL & TRAINING (COLLAPSIBLE) ==========
        with st.expander("🧠 Model & Training Configuration", expanded=False):
            st.markdown("**Model Architecture**")
            architecture, training_mode, dinov2_model, hidden_dim, dropout, use_untrained_resnet, num_conv_layers, conv_channels = render_model_config()
            
            epochs, learning_rate, weight_decay, train_batch_size = render_training_config()
        
        st.markdown("---")
        
        # ========== D. EVALUATION (COLLAPSIBLE) ==========
        with st.expander("📊 Evaluation Configuration", expanded=False):
            st.info("💡 Configure how uncertainty is measured and evaluated")
            
            # Get noise rate for evaluation
            if aleatoric_config['noise_source'] == "Use CIFAR-10N noise" and stats:
                noise_rate_for_eval = stats.get('noise_rate', 0)
            else:
                noise_rate_for_eval = aleatoric_config['custom_noise_rate'] / 100.0
            
            mc_passes, selected_signals, eval_per_group = render_evaluation_config(
                under_supported_list=None,  # Will be determined at runtime
                under_train_per_class=epistemic_config['under_train_per_class'],
                regular_train_per_class=epistemic_config['regular_train_per_class'],
                noise_rate=noise_rate_for_eval,
                key_prefix="unified"
            )
        
        st.markdown("---")
        
        # ========== E. DYNAMIC EXPERIMENT TYPE DETECTION ==========
        experiment_type, num_experiments, validation_info = detect_experiment_type(
            epistemic_config, aleatoric_config
        )
        
        render_experiment_type_summary(
            experiment_type, num_experiments, validation_info,
            epistemic_config, aleatoric_config
        )
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Create Experiment(s)",
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            # Build the unified configuration
            config = build_unified_config(
                epistemic_config=epistemic_config,
                aleatoric_config=aleatoric_config,
                validation_info=validation_info,
                noise_type=noise_type,
                dinov2_model=dinov2_model,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_untrained_resnet=use_untrained_resnet,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                train_batch_size=train_batch_size,
                eval_per_group=eval_per_group,
                mc_passes=mc_passes,
                architecture=architecture,
                training_mode=training_mode,
                num_conv_layers=num_conv_layers,
                conv_channels=conv_channels,
            )
            
            # Handle submission with the config
            handle_unified_builder_submission(
                config=config,
                api_base_url=api_base_url,
                get_headers=get_headers
            )
            
            return config
    
    # Show execution panel outside form (always visible)
    render_experiment_execution_panel(api_base_url, get_headers)
    
    return None


def render_dataset_config_compact(
    dataset_name: str,
    stats: Optional[Dict],
    class_names: List[str]
) -> None:
    """
    Render compact dataset configuration section.
    
    Args:
        dataset_name: Name of the dataset
        stats: Dataset statistics
        class_names: List of class names
    """
    if stats:
        noise_rate = stats.get('noise_rate', 0) * 100
        st.info(f"**{dataset_name.upper()}**: {len(class_names)} classes, avg {noise_rate:.0f}% noise")
    else:
        st.info(f"**{dataset_name.upper()}**: {len(class_names)} classes")
    
    with st.expander("📋 Show Dataset Details"):
        st.markdown("**Available Classes:**")
        st.write(", ".join([f"{i}: {name}" for i, name in enumerate(class_names)]))
        
        if stats:
            noise_per_class = stats.get('noise_per_class', {})
            if noise_per_class:
                st.markdown("**Noise Distribution:**")
                table_data = []
                for class_id, data in noise_per_class.items():
                    class_idx = int(class_id)
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_id}"
                    rate = data.get('rate', 0) * 100
                    table_data.append({
                        "Class": class_name,
                        "Noise Rate": f"{rate:.1f}%"
                    })
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)


def render_epistemic_with_sweep(class_names: List[str]) -> Dict[str, Any]:
    """
    Render epistemic uncertainty configuration with sweep checkbox.
    
    Args:
        class_names: List of class names
    
    Returns:
        Dictionary with epistemic configuration including sweep settings
    """
    st.markdown("#### 🔬 Epistemic Uncertainty")
    st.caption("Model uncertainty from insufficient training data")
    
    # Under-supported classes configuration
    random_under_supported = st.checkbox(
        "Random under-supported classes",
        value=True,
        help="Randomly select classes to under-sample",
        key="unified_random_under"
    )
    
    if random_under_supported:
        num_under_supported = st.number_input(
            "Number of classes",
            min_value=1, max_value=5, value=2,
            help="How many classes to randomly under-sample",
            key="unified_num_under"
        )
        under_supported = f"random:{num_under_supported}"
    else:
        under_supported_list = st.multiselect(
            "Select classes",
            options=list(range(10)),
            format_func=lambda x: f"{x}: {class_names[x]}",
            default=[3, 5],
            key="unified_under_list"
        )
        under_supported = ",".join(map(str, under_supported_list)) if under_supported_list else "3,5"
    
    # Samples per class
    under_train_per_class = st.number_input(
        "Samples per under-supported class",
        min_value=10, max_value=500, value=50,
        help="Limited samples create epistemic uncertainty",
        key="unified_under_train"
    )
    
    # Sweep checkbox
    sweep_epistemic = st.checkbox(
        "☐ Sweep this parameter",
        value=False,
        help="Enable to create multiple experiments with different epistemic values",
        key="unified_sweep_epistemic"
    )
    
    epistemic_values = [under_train_per_class]
    if sweep_epistemic:
        st.markdown("**Sweep Values:**")
        sweep_values_str = st.text_input(
            "Values (comma-separated)",
            value="50, 100, 200",
            help="Enter values to sweep (e.g., 50, 100, 200)",
            key="unified_epistemic_sweep_values"
        )
        try:
            epistemic_values = [int(v.strip()) for v in sweep_values_str.split(",")]
            st.caption(f"→ Creates {len(epistemic_values)} experiments")
        except ValueError:
            st.error("Invalid values. Please enter comma-separated integers.")
            epistemic_values = [under_train_per_class]
    
    # Regular classes
    regular_train_per_class = st.number_input(
        "Samples per regular class",
        min_value=50, max_value=1000, value=300,
        help="Well-supported classes for comparison",
        key="unified_regular_train"
    )
    
    return {
        'under_supported': under_supported,
        'under_train_per_class': under_train_per_class,
        'regular_train_per_class': regular_train_per_class,
        'sweep_enabled': sweep_epistemic,
        'sweep_values': epistemic_values
    }


def render_aleatoric_with_sweep(
    stats: Optional[Dict],
    noise_type: str,
    class_names: List[str],
    fetch_dataset_stats: Callable,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Render aleatoric uncertainty configuration with sweep checkbox.
    
    Args:
        stats: Dataset statistics
        noise_type: Type of noise
        class_names: List of class names
        fetch_dataset_stats: Function to fetch stats
        dataset_name: Name of dataset
    
    Returns:
        Dictionary with aleatoric configuration including sweep settings
    """
    st.markdown("#### 🎲 Aleatoric Uncertainty (Label Noise)")
    st.info("💡 **Base Dataset**: Clean CIFAR-10 (no noise). Optionally add label noise below.")
    
    # Noise source selection with 3 options
    noise_source = st.radio(
        "Label Noise Strategy",
        [
            "No noise (0%, clean labels)",
            "CIFAR-10N pre-existing noise (~18-40%, not sweepable)",
            "Custom random flipping (0-100%, sweepable)"
        ],
        index=2,  # Default to custom flipping
        help="Choose label noise strategy. For sweeps, use custom flipping.",
        key="unified_noise_source_selection"
    )
    
    if noise_source.startswith("No noise"):
        # No noise - clean CIFAR-10
        custom_noise_rate = 0.0
        noise_level = 0.0
        st.success("✅ Using clean CIFAR-10 labels (no noise)")
        
    elif noise_source.startswith("CIFAR-10N"):
        # Use CIFAR-10N pre-existing noise
        custom_noise_rate = 0.0  # Backend will use CIFAR-10N instead
        noise_level = stats.get('noise_rate', 0.4) * 100 if stats else 40.0
        
        st.warning("⚠️ **Not sweepable**: CIFAR-10N noise is fixed per noise_type")
        if stats:
            avg_noise = stats.get('noise_rate', 0) * 100
            st.metric("CIFAR-10N Noise Rate", f"{avg_noise:.1f}%")
    else:
        # Custom random label flipping - RECOMMENDED for sweeps
        custom_noise_rate = st.slider(
            "Custom noise percentage",
            min_value=0, max_value=100, value=10,
            help=(
                "Percentage of labels to randomly flip (uniform across all classes). "
                "Range 0-100 — above 80% the clean-vs-noisy AUROC starts to saturate "
                "and at 100% the clean evaluation pool is empty."
            ),
            key="unified_custom_noise"
        )
        noise_level = custom_noise_rate
        
        if custom_noise_rate == 0:
            st.success("✅ Using clean CIFAR-10 labels (0% noise)")
        else:
            st.warning(f"⚠️ Will inject {custom_noise_rate}% uniform random label flipping into clean CIFAR-10")
    
    # Sweep checkbox
    sweep_aleatoric = st.checkbox(
        "☐ Sweep this parameter",
        value=False,
        help="Enable to create multiple experiments with different aleatoric values",
        key="unified_sweep_aleatoric"
    )
    
    aleatoric_values = [noise_level]
    if sweep_aleatoric:
        st.markdown("**Sweep Values:**")
        sweep_values_str = st.text_input(
            "Values (comma-separated)",
            value="0, 20, 40, 60",
            help="Enter noise percentages to sweep (e.g., 0, 20, 40, 60)",
            key="unified_aleatoric_sweep_values"
        )
        try:
            aleatoric_values = [float(v.strip()) for v in sweep_values_str.split(",")]
            st.caption(f"→ Creates {len(aleatoric_values)} experiments")
        except ValueError:
            st.error("Invalid values. Please enter comma-separated numbers.")
            aleatoric_values = [noise_level]
    
    return {
        'noise_source': noise_source,
        'custom_noise_rate': custom_noise_rate,
        'noise_level': noise_level,
        'sweep_enabled': sweep_aleatoric,
        'sweep_values': aleatoric_values
    }


def build_unified_config(
    epistemic_config: Dict[str, Any],
    aleatoric_config: Dict[str, Any],
    validation_info: Dict[str, Any],
    noise_type: str,
    dinov2_model: str,
    hidden_dim: int,
    dropout: float,
    use_untrained_resnet: bool,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    train_batch_size: int,
    eval_per_group: int,
    mc_passes: int,
    architecture: str = "dinov2_mlp",
    training_mode: str = "feature_space",
    num_conv_layers: int = 3,
    conv_channels: List[int] = [32, 64, 64],
) -> UnifiedBuilderConfig:
    """
    Build UnifiedBuilderConfig from form inputs.
    
    Args:
        epistemic_config: Epistemic configuration dict
        aleatoric_config: Aleatoric configuration dict
        validation_info: Validation information dict
        noise_type: Noise type
        dinov2_model: Model architecture
        hidden_dim: Hidden dimension
        dropout: Dropout rate (used for both training and MC dropout)
        use_untrained_resnet: Whether to use untrained ResNet
        epochs: Training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        train_batch_size: Batch size
        eval_per_group: Evaluation samples per group
        mc_passes: MC dropout passes
        architecture: Model architecture (dinov2_mlp, cnn_mcdropout, resnet18_mcdropout)
        training_mode: Training mode (feature_space or end_to_end)
        num_conv_layers: Number of convolutional layers (for CNN)
        conv_channels: Channel sizes for each conv layer (for CNN)
    
    Returns:
        UnifiedBuilderConfig dataclass
    """
    # Build epistemic config
    epistemic = EpistemicConfig(
        under_supported_classes=epistemic_config['under_supported'],
        samples_per_class=epistemic_config['under_train_per_class'],
        regular_samples_per_class=epistemic_config['regular_train_per_class'],
        sweep=SweepConfig(
            enabled=epistemic_config['sweep_enabled'],
            values=[float(v) for v in epistemic_config['sweep_values']]
        )
    )
    
    # Build aleatoric config
    aleatoric = AleatoricConfig(
        noise_source="cifar10n" if aleatoric_config['noise_source'] == "Use CIFAR-10N noise" else "random",
        noise_level=aleatoric_config['noise_level'],
        sweep=SweepConfig(
            enabled=aleatoric_config['sweep_enabled'],
            values=[float(v) for v in aleatoric_config['sweep_values']]
        )
    )
    
    # Build validation config
    validation = ValidationConfig(
        validation_enabled=validation_info['validation_enabled'],
        is_epistemic_sweep=validation_info['is_epistemic_sweep'],
        is_aleatoric_sweep=validation_info['is_aleatoric_sweep'],
        epistemic_parameter=validation_info['epistemic_parameter'],
        aleatoric_parameter=validation_info['aleatoric_parameter']
    )
    
    return UnifiedBuilderConfig(
        epistemic=epistemic,
        aleatoric=aleatoric,
        validation=validation,
        dinov2_model=dinov2_model,
        hidden_dim=hidden_dim,
        use_untrained_resnet=use_untrained_resnet,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        train_batch_size=train_batch_size,
        training_dropout=dropout,
        eval_per_group=eval_per_group,
        mc_dropout_enabled=True,
        mc_dropout_rate=dropout,  # Use same rate for now
        mc_passes=mc_passes,
        noise_type=noise_type
    )


def detect_experiment_type(
    epistemic_config: Dict[str, Any],
    aleatoric_config: Dict[str, Any]
) -> Tuple[str, int, Dict[str, Any]]:
    """
    Detect experiment type based on sweep configurations.
    
    Args:
        epistemic_config: Epistemic configuration
        aleatoric_config: Aleatoric configuration
    
    Returns:
        Tuple of (experiment_type, num_experiments, validation_info)
    """
    epis_sweep = epistemic_config['sweep_enabled']
    alea_sweep = aleatoric_config['sweep_enabled']
    
    epis_values = epistemic_config['sweep_values']
    alea_values = aleatoric_config['sweep_values']
    
    validation_info = {
        'validation_enabled': False,
        'is_epistemic_sweep': False,
        'is_aleatoric_sweep': False,
        'epistemic_parameter': None,
        'aleatoric_parameter': None
    }
    
    if not epis_sweep and not alea_sweep:
        return "single", 1, validation_info
    
    elif epis_sweep and not alea_sweep:
        validation_info.update({
            'validation_enabled': True,
            'is_epistemic_sweep': True,
            'epistemic_parameter': 'under_train_per_class'
        })
        return "1d_epistemic", len(epis_values), validation_info
    
    elif alea_sweep and not epis_sweep:
        validation_info.update({
            'validation_enabled': True,
            'is_aleatoric_sweep': True,
            'aleatoric_parameter': 'aleatoric_noise_percentage'
        })
        return "1d_aleatoric", len(alea_values), validation_info
    
    else:  # Both sweeps enabled
        validation_info.update({
            'validation_enabled': True,
            'is_epistemic_sweep': True,
            'is_aleatoric_sweep': True,
            'epistemic_parameter': 'under_train_per_class',
            'aleatoric_parameter': 'aleatoric_noise_percentage'
        })
        return "2d_grid", len(epis_values) * len(alea_values), validation_info


def render_experiment_type_summary(
    experiment_type: str,
    num_experiments: int,
    validation_info: Dict[str, Any],
    epistemic_config: Dict[str, Any],
    aleatoric_config: Dict[str, Any]
) -> None:
    """
    Render experiment type detection summary.
    
    Args:
        experiment_type: Detected experiment type
        num_experiments: Number of experiments to create
        validation_info: Validation configuration
        epistemic_config: Epistemic configuration
        aleatoric_config: Aleatoric configuration
    """
    st.markdown("### 🎯 Experiment Type: [Auto-detected]")
    
    if experiment_type == "single":
        st.success(f"""
        ✅ **Single Experiment**
        
        No sweeps enabled → Creates 1 experiment
        """)
    
    elif experiment_type == "1d_epistemic":
        epis_values = epistemic_config['sweep_values']
        st.success(f"""
        ✅ **1D Epistemic Sweep** ({num_experiments} experiments)
        
        - Epistemic values: {epis_values}
        - Validates: **C2** (ue ∼∝ Ue) and **O1** (ua ⊥ Ue)
        """)
    
    elif experiment_type == "1d_aleatoric":
        alea_values = aleatoric_config['sweep_values']
        st.success(f"""
        ✅ **1D Aleatoric Sweep** ({num_experiments} experiments)
        
        - Aleatoric values: {alea_values}
        - Validates: **C1** (ua ∼∝ Ua) and **O2** (ue ⊥ Ua)
        """)
    
    elif experiment_type == "2d_grid":
        epis_values = epistemic_config['sweep_values']
        alea_values = aleatoric_config['sweep_values']
        st.success(f"""
        ✅ **2D Grid Sweep** ({num_experiments} experiments)
        
        - Epistemic: {len(epis_values)} values {epis_values}
        - Aleatoric: {len(alea_values)} values {alea_values}
        - Validates: **C1, C2, O1, O2** ✅ Full validation
        
        **Estimated time**: ~{num_experiments * 5} minutes
        """)


def handle_unified_builder_submission(
    config: UnifiedBuilderConfig,
    api_base_url: str,
    get_headers: Callable
) -> None:
    """
    Handle form submission and create experiments.
    
    Args:
        config: UnifiedBuilderConfig with all experiment settings
        api_base_url: API base URL
        get_headers: Function to get headers
    """
    import requests
    
    experiment_type = config.experiment_type
    
    with st.spinner(f"Creating {experiment_type} experiment(s)..."):
        try:
            created_experiments = []
            
            # Get sweep values
            epis_values = config.epistemic.sweep.values
            alea_values = config.aleatoric.sweep.values
            
            # Generate timestamp for batch name
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            # Create experiments based on type
            for epis_val in epis_values:
                for alea_val in alea_values:
                    # Generate experiment name
                    if experiment_type == "Single":
                        exp_name = f"exp_{timestamp}"
                    elif experiment_type == "1D Epistemic":
                        exp_name = f"epis_sweep_{timestamp}_e{int(epis_val)}"
                    elif experiment_type == "1D Aleatoric":
                        exp_name = f"alea_sweep_{timestamp}_a{int(alea_val)}"
                    else:  # 2D Grid
                        exp_name = f"grid_{timestamp}_e{int(epis_val)}_a{int(alea_val)}"
                    
                    # Build experiment config
                    exp_config = build_base_experiment_config(
                        noise_type=config.noise_type,
                        under_supported=config.epistemic.under_supported_classes,
                        under_train_per_class=int(epis_val),
                        regular_train_per_class=config.epistemic.regular_samples_per_class,
                        dinov2_model=config.dinov2_model,
                        hidden_dim=config.hidden_dim,
                        dropout=config.training_dropout or 0.2,
                        epochs=config.epochs,
                        learning_rate=config.learning_rate,
                        weight_decay=config.weight_decay,
                        train_batch_size=config.train_batch_size,
                        eval_per_group=config.eval_per_group,
                        mc_passes=config.mc_passes,
                        use_untrained_resnet=config.use_untrained_resnet,
                        aleatoric_noise_percentage=alea_val,
                    )
                    
                    # Add validation metadata
                    if config.validation.validation_enabled:
                        exp_config['validation_metadata'] = {
                            'validation_enabled': config.validation.validation_enabled,
                            'is_epistemic_sweep': config.validation.is_epistemic_sweep,
                            'is_aleatoric_sweep': config.validation.is_aleatoric_sweep,
                            'epistemic_parameter': config.validation.epistemic_parameter,
                            'aleatoric_parameter': config.validation.aleatoric_parameter,
                        }
                    
                    experiment_data = {
                        "name": exp_name,
                        "config": exp_config
                    }
                    
                    # Create experiment
                    response = requests.post(
                        f"{api_base_url}/api/v1/experiments/no-auth",
                        json=experiment_data,
                        headers=get_headers(),
                        timeout=30
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    created_experiments.append({
                        "id": result["id"],
                        "name": result["name"],
                        "epistemic": epis_val,
                        "aleatoric": alea_val,
                    })
            
            # Show success message
            st.success(f"✅ Created {len(created_experiments)} experiment(s)!")
            
            # Show details
            with st.expander("📋 Created Experiments"):
                for exp in created_experiments:
                    st.text(f"  • {exp['name']} (ID: {exp['id']})")
            
            # Clear, actionable message with visual indicator
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #4CAF5022 0%, #4CAF5011 100%);
                border-left: 4px solid #4CAF50;
                padding: 16px;
                border-radius: 4px;
                margin: 16px 0;
            ">
                <div style="font-size: 1.1em; font-weight: 600; color: #4CAF50; margin-bottom: 8px;">
                    ⬇️ Next Step: Start Your Experiments
                </div>
                <div style="color: #666;">
                    The <strong>"▶️ Queued Experiments"</strong> section is right below this form.
                    <br>Look for the green expandable sections with start buttons.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to create experiment(s): {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Response: {e.response.text}")


def render_completed_experiments_with_visualizations(
    completed: List[Dict],
    api_base_url: str
) -> None:
    """
    Render completed experiments section with automatic visualization detection.
    
    Detects and renders:
    - 2D grid heatmaps (with 1D slices)
    - 1D sweep line plots (epistemic or aleatoric)
    
    Args:
        completed: List of completed experiments
        api_base_url: API base URL for additional data fetching
    """
    from collections import defaultdict
    from ..visualization.sweeps.heatmap_visualization import (
        render_2d_grid_heatmaps,
        render_1d_sweep_plot,
        detect_sweep_type,
        render_multi_signal_heatmaps,
        render_enhanced_1d_sweep_plot,
    )

    if not completed:
        return

    st.markdown("---")
    st.markdown("### ✅ Completed Experiments")
    st.success(f"**{len(completed)} experiment(s)** finished successfully!")

    # Group completed experiments by timestamp
    completed_batches = defaultdict(list)
    for exp in completed:
        name = exp.get('name', '')
        parts = name.split('_')
        if len(parts) >= 3:
            timestamp = f"{parts[1]}_{parts[2]}"
        else:
            timestamp = "unknown"
        completed_batches[timestamp].append(exp)

    # Classify each batch using the proper variation-based detector.  Substring
    # sniffing on ``_e`` / ``_a`` previously misclassified 1D aleatoric sweeps
    # (whose names contain ``_exp_`` AND ``_aleatoric_``) as 2D grids.
    batch_kinds: Dict[str, Tuple[str, Optional[str]]] = {
        timestamp: detect_sweep_type(experiments)
        for timestamp, experiments in completed_batches.items()
    }
    grid_batches = {
        timestamp: experiments
        for timestamp, experiments in completed_batches.items()
        if batch_kinds[timestamp][0] == "2d_grid" and len(experiments) >= 4
    }
    
    # Show results table
    with st.expander(f"📊 View Results Table ({len(completed)})", expanded=True):
        results_data = []
        for exp in completed[:20]:
            results_data.append({
                'Name': exp['name'],
                'Epistemic AUROC': f"{exp.get('epistemic_auroc', 0):.3f}" if exp.get('epistemic_auroc') else 'N/A',
                'Aleatoric AUROC': f"{exp.get('aleatoric_auroc', 0):.3f}" if exp.get('aleatoric_auroc') else 'N/A',
                'Accuracy': f"{exp.get('accuracy', 0):.3f}" if exp.get('accuracy') else 'N/A',
            })
        
        if results_data:
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        if len(completed) > 20:
            st.caption(f"... and {len(completed) - 20} more")
    
    # ========== 2D GRID VISUALIZATIONS ==========
    if grid_batches:
        st.markdown("---")
        st.markdown("### 🔥 2D Grid Analysis")
        st.info(f"**{len(grid_batches)} 2D grid batch(es)** detected")
        
        for timestamp, experiments in grid_batches.items():
            kind, _ = batch_kinds.get(timestamp, ("unknown", None))
            if kind == "2d_grid":
                type_badge = "🔥 2D Grid"
            elif kind == "epistemic":
                type_badge = "📈 1D Epistemic"
            elif kind in ("aleatoric_custom", "aleatoric_cifar10n"):
                type_badge = "📈 1D Aleatoric"
            else:
                type_badge = "📊 Batch"
            type_desc = f"{len(experiments)} data points"
            
            with st.expander(f"{type_badge} | Batch {timestamp} | {type_desc}", expanded=True):
                # Create tabs for different views
                tab_2d, tab_multi, tab_1d_epis, tab_1d_alea = st.tabs([
                    "🔥 2D Heatmaps (Aggregated)",
                    "🔬 All 7 Signals",
                    "📈 1D Epistemic",
                    "📈 1D Aleatoric"
                ])
                
                with tab_2d:
                    render_2d_grid_heatmaps(
                        experiments=experiments,
                        epistemic_param="under_train_per_class",
                        aleatoric_param="aleatoric_noise_percentage",
                        key_prefix=timestamp
                    )
                
                with tab_multi:
                    render_multi_signal_heatmaps(
                        experiments=experiments,
                        epistemic_param="under_train_per_class",
                        aleatoric_param="aleatoric_noise_percentage",
                        key_prefix=timestamp
                    )
                
                with tab_1d_epis:
                    st.caption("Epistemic slice: All 7 signals vs training data size")
                    render_enhanced_1d_sweep_plot(
                        experiments=experiments,
                        sweep_param="under_train_per_class",
                        sweep_type="epistemic",
                        key_prefix=timestamp
                    )
                
                with tab_1d_alea:
                    st.caption("Aleatoric slice: All 7 signals vs label noise")
                    render_enhanced_1d_sweep_plot(
                        experiments=experiments,
                        sweep_param="aleatoric_noise_percentage",
                        sweep_type="aleatoric",
                        key_prefix=timestamp
                    )
    
    # ========== 1D SWEEP VISUALIZATIONS ==========
    one_d_batches = {}
    for timestamp, experiments in completed_batches.items():
        if timestamp in grid_batches or len(experiments) < 2:
            continue
        sweep_type, sweep_param = batch_kinds.get(timestamp, ("unknown", None))
        if sweep_type in ("epistemic", "aleatoric_custom", "aleatoric_cifar10n"):
            one_d_batches[timestamp] = {
                "experiments": experiments,
                "sweep_type": sweep_type,
                "sweep_param": sweep_param,
            }
    
    if one_d_batches:
        st.markdown("---")
        st.markdown("### 📈 1D Sweep Analysis")
        st.info(f"**{len(one_d_batches)} 1D sweep batch(es)** detected")
        
        for timestamp, batch_info in one_d_batches.items():
            experiments = batch_info['experiments']
            sweep_type = batch_info['sweep_type']
            sweep_param = batch_info['sweep_param']
            
            # Create friendly label
            if sweep_type == "epistemic":
                type_label = "📈 Epistemic Sweep"
            elif sweep_type == "aleatoric_cifar10n":
                type_label = "📈 Aleatoric (CIFAR-10N)"
            else:
                type_label = "📈 Aleatoric (Custom)"
            
            with st.expander(f"{type_label}: Batch {timestamp} ({len(experiments)} experiments)", expanded=True):
                render_type = "epistemic" if sweep_type == "epistemic" else "aleatoric"
                render_1d_sweep_plot(
                    experiments=experiments,
                    sweep_param=sweep_param,
                    sweep_type=render_type
                )


def render_experiment_execution_panel(
    api_base_url: str,
    get_headers: Callable
) -> None:
    """
    Render comprehensive execution panel showing all experiment statuses.
    
    Shows:
    - Overall status dashboard (queued, running, completed, failed counts)
    - Queued experiments with start buttons
    - Running experiments with progress indicators
    - Completed experiments with results preview
    - Failed experiments with error info
    
    Args:
        api_base_url: API base URL
        get_headers: Function to get headers
    """
    import requests
    from collections import defaultdict
    
    st.markdown("---")
    
    try:
        # Fetch all experiments
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers(),
            timeout=30
        )
        response.raise_for_status()
        all_experiments = response.json()
        
        # Categorize experiments by status
        queued = [exp for exp in all_experiments if exp.get('status') == 'queued']
        running = [exp for exp in all_experiments if exp.get('status') == 'running']
        completed = [exp for exp in all_experiments if exp.get('status') == 'completed']
        failed = [exp for exp in all_experiments if exp.get('status') == 'failed']
        
        # ========== STATUS DASHBOARD ==========
        st.markdown("## 📊 Experiment Status Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "⏸️ Queued",
                len(queued),
                delta="Ready to start" if queued else None
            )
        
        with col2:
            st.metric(
                "▶️ Running",
                len(running),
                delta="In progress" if running else None
            )
        
        with col3:
            st.metric(
                "✅ Completed",
                len(completed),
                delta="View results" if completed else None
            )
        
        with col4:
            st.metric(
                "❌ Failed",
                len(failed),
                delta="Check errors" if failed else None,
                delta_color="inverse"
            )
        
        # ========== QUEUED EXPERIMENTS ==========
        if queued:
            st.markdown("---")
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #4CAF5022 0%, #4CAF5011 100%);
                border-left: 4px solid #4CAF50;
                padding: 12px 16px;
                border-radius: 4px;
                margin: 16px 0;
            ">
                <h3 style="margin: 0; color: #4CAF50;">
                    ▶️ Queued Experiments - Start Here!
                </h3>
                <p style="margin: 8px 0 0 0; color: #666; font-size: 0.9em;">
                    Click the start button below to begin training your experiments
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Group by timestamp (e.g., "20260522_185329")
            batches = defaultdict(list)
            for exp in queued:
                name = exp.get('name', '')
                parts = name.split('_')
                if len(parts) >= 3:
                    timestamp = f"{parts[1]}_{parts[2]}"  # e.g., "20260522_185329"
                else:
                    timestamp = name.split('_')[0] if '_' in name else name
                batches[timestamp].append(exp)
            
            st.info(f"**{len(queued)} queued experiment(s)** in {len(batches)} batch(es)")
            
            # Show each batch with start button
            for timestamp, experiments in batches.items():
                with st.expander(f"📦 Batch {timestamp} ({len(experiments)} experiments)", expanded=True):
                    # Show experiment names
                    for exp in experiments[:5]:
                        st.caption(f"  • {exp['name']}")
                    if len(experiments) > 5:
                        st.caption(f"  ... and {len(experiments) - 5} more")
                    
                    # Start button
                    if st.button(
                        f"🚀 Start Batch {timestamp} ({len(experiments)} experiments)",
                        key=f"start_batch_{timestamp}",
                        type="primary",
                        use_container_width=True
                    ):
                        with st.spinner(f"Starting {len(experiments)} experiments..."):
                            try:
                                started_count = 0
                                for exp in experiments:
                                    start_response = requests.post(
                                        f"{api_base_url}/api/v1/experiments/no-auth/{exp['id']}/start",
                                        headers=get_headers(),
                                        timeout=30
                                    )
                                    start_response.raise_for_status()
                                    started_count += 1
                                
                                st.success(f"✅ Started {started_count} experiments in batch {timestamp}!")
                                st.rerun()
                                return  # CRITICAL: Stop execution after rerun
                                
                            except requests.exceptions.RequestException as e:
                                st.error(f"❌ Failed to start batch: {str(e)}")
        
        # ========== RUNNING EXPERIMENTS ==========
        if running:
            st.markdown("---")
            st.markdown("### ▶️ Running Experiments")
            
            # Group running experiments by timestamp
            running_batches = defaultdict(list)
            for exp in running:
                name = exp.get('name', '')
                # Extract timestamp (e.g., "grid_20260522_185329" -> "20260522_185329")
                parts = name.split('_')
                if len(parts) >= 3:
                    timestamp = f"{parts[1]}_{parts[2]}"  # e.g., "20260522_185329"
                else:
                    timestamp = "unknown"
                running_batches[timestamp].append(exp)
            
            st.info(f"**{len(running)} experiment(s)** currently training in {len(running_batches)} batch(es)")
            
            # Show each batch with progress
            for timestamp, experiments in running_batches.items():
                with st.expander(f"🔄 Batch {timestamp} ({len(experiments)} running)", expanded=True):
                    # Show first few experiment names
                    for exp in experiments[:5]:
                        st.caption(f"  • {exp['name']}")
                    if len(experiments) > 5:
                        st.caption(f"  ... and {len(experiments) - 5} more")
                    
                    # Progress indicator
                    st.progress(0.5, text=f"Training {len(experiments)} experiments...")
            
            # Auto-refresh button
            if st.button("🔄 Refresh Status", key="refresh_running", use_container_width=True):
                st.rerun()
        
        # ========== COMPLETED EXPERIMENTS ==========
        # Use modular function for completed experiments with visualizations
        if completed:
            render_completed_experiments_with_visualizations(completed, api_base_url)
        
        # ========== FAILED EXPERIMENTS ==========
        if failed:
            st.markdown("---")
            st.markdown("### ❌ Failed Experiments")
            st.error(f"**{len(failed)} experiment(s)** encountered errors")
            
            with st.expander(f"🔍 View Errors ({len(failed)})", expanded=False):
                for exp in failed[:10]:
                    st.text(f"❌ {exp['name']}")
                    if exp.get('error'):
                        st.caption(f"Error: {exp['error']}")
                
                if len(failed) > 10:
                    st.caption(f"... and {len(failed) - 10} more")
        
        # ========== EMPTY STATE ==========
        if not any([queued, running, completed, failed]):
            st.info("📝 No experiments yet. Create some using the form above!")
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch experiments: {str(e)}")


# Made with Bob