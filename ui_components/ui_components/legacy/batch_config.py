"""
Batch Experiment Configuration UI Components

This module contains UI components for configuring batch experiments
with parameter sweeps.
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Callable, Any

from ..config.experiment_config import render_epistemic_config, render_aleatoric_config


def render_batch_sweep_config() -> Tuple[str, str, float, float, float, List[float]]:
    """
    Render controls for configuring a single-parameter numeric sweep.
    
    IMPROVED UX: Simplified approach that clearly separates sweep configuration
    from base configuration, following best practices for parameter sweeps.

    Returns:
        Tuple of (parameter, value_type, start, end, step, generated_values)
    """
    st.markdown("### 🔁 Parameter Sweep Configuration")
    st.info("""
    💡 **Best Practice**: Sweep one parameter at a time while keeping all others constant.
    This allows you to isolate the effect of each parameter on model performance.
    """)

    parameter_options = {
        "under_train_per_class": "Epistemic: under-supported samples/class",
        "regular_train_per_class": "Epistemic: regular samples/class",
        "eval_per_group": "Evaluation: samples/group",
        "aleatoric_noise_percentage": "Aleatoric: noise percentage (0-100%)",
        "hidden_dim": "Model: hidden dimension",
        "dropout": "Model: dropout rate",
        "epochs": "Training: epochs",
        "learning_rate": "Training: learning rate",
        "weight_decay": "Training: weight decay",
        "train_batch_size": "Training: batch size",
        "mc_passes": "Evaluation: MC dropout passes",
    }

    parameter = st.selectbox(
        "Parameter to Sweep",
        options=list(parameter_options.keys()),
        format_func=lambda key: f"{key} — {parameter_options[key]}",
        help="This parameter will vary across experiments. All other parameters will use base configuration values.",
    )

    integer_parameters = {
        "under_train_per_class",
        "regular_train_per_class",
        "eval_per_group",
        "hidden_dim",
        "epochs",
        "train_batch_size",
        "mc_passes",
    }
    value_type = "int" if parameter in integer_parameters else "float"

    # Smart defaults based on parameter type
    sweep_defaults = {
        "under_train_per_class": (5, 50, 5),
        "regular_train_per_class": (100, 500, 100),
        "eval_per_group": (100, 500, 100),
        "aleatoric_noise_percentage": (0.0, 100.0, 10.0),
        "hidden_dim": (64, 512, 64),
        "dropout": (0.1, 0.5, 0.1),
        "epochs": (5, 20, 5),
        "learning_rate": (0.0005, 0.005, 0.0005),
        "weight_decay": (0.0001, 0.001, 0.0001),
        "train_batch_size": (64, 512, 64),
        "mc_passes": (10, 50, 10),
    }
    
    default_start, default_end, default_step = sweep_defaults.get(parameter, (1, 10, 1))

    col1, col2, col3 = st.columns(3)
    if value_type == "int":
        with col1:
            start = float(st.number_input("Start Value", min_value=1, value=int(default_start), step=1,
                                         help="First value in the sweep range"))
        with col2:
            end = float(st.number_input("End Value", min_value=1, value=int(default_end), step=1,
                                       help="Last value in the sweep range (inclusive)"))
        with col3:
            step = float(st.number_input("Step Size", min_value=1, value=int(default_step), step=1,
                                        help="Increment between values"))
    else:
        with col1:
            start = float(st.number_input("Start Value", min_value=0.0, value=default_start, format="%.4f",
                                         help="First value in the sweep range"))
        with col2:
            end = float(st.number_input("End Value", min_value=0.0, value=default_end, format="%.4f",
                                       help="Last value in the sweep range (inclusive)"))
        with col3:
            step = float(st.number_input("Step Size", min_value=0.0001, value=default_step, format="%.4f",
                                        help="Increment between values"))

    generated_values = _generate_sweep_preview(start, end, step, value_type)

    if generated_values:
        st.success(f"✅ **{len(generated_values)} experiments** will be created with values: {', '.join(str(v) for v in generated_values)}")
        if len(generated_values) > 20:
            st.warning("⚠️ This batch has more than 20 runs and may take a long time to complete.")
        if len(generated_values) > 100:
            st.error("❌ Sweep exceeds the V1 limit of 100 generated runs. Please reduce the range or increase the step size.")
    else:
        st.error("❌ Sweep produced no values. Check start/end/step inputs.")

    return parameter, value_type, start, end, step, generated_values


def render_batch_base_config(
    swept_parameter: str,
    noise_type: str,
    stats: Optional[Dict],
    dataset_name: str,
    class_names: List[str],
    fetch_dataset_stats: Callable[[str, str], Optional[Dict]]
) -> Dict[str, Any]:
    """
    Render base configuration section for batch experiments.
    Only shows parameters that are NOT being swept.
    
    Args:
        swept_parameter: The parameter being swept (will be hidden from base config)
        noise_type: Selected noise type
        stats: Dataset statistics
        dataset_name: Name of the dataset
        class_names: List of class names
        fetch_dataset_stats: Function to fetch dataset stats
    
    Returns:
        Dictionary with all base configuration values
    """
    st.markdown("### ⚙️ Base Configuration")
    st.info("""
    💡 **Base Configuration**: These values will be used for all experiments in the batch.
    Only the swept parameter will vary across experiments.
    """)
    
    config = {}
    
    # Use expander for advanced configuration
    with st.expander("🔧 Configure Base Parameters", expanded=True):
        
        # Epistemic Configuration (if not being swept)
        if swept_parameter not in ["under_train_per_class", "regular_train_per_class"]:
            st.markdown("#### 📊 Epistemic Uncertainty Configuration")
            
            if swept_parameter == "under_train_per_class":
                # Only need under_supported selection, not the samples/class
                config["under_supported"] = st.selectbox(
                    "Under-supported Classes",
                    ["0,1", "random:2", "random:3"],
                    help="Classes with limited training data (samples/class will be swept)",
                    key="batch_base_under_supported"
                )
                config["under_train_per_class"] = None  # Will use sweep values
                config["regular_train_per_class"] = st.number_input(
                    "Regular samples/class",
                    min_value=50,
                    max_value=500,
                    value=300,
                    step=50,
                    help="Training samples for well-supported classes",
                    key="batch_base_regular_train"
                )
            elif swept_parameter == "regular_train_per_class":
                # Get under_supported and under_train, but not regular_train
                config["under_supported"], config["under_train_per_class"], _, _ = render_epistemic_config(
                    class_names,
                    default_under_train=50,
                    default_regular_train=300
                )
                config["regular_train_per_class"] = None  # Will use sweep values
            else:
                # Get all epistemic config
                config["under_supported"], config["under_train_per_class"], config["regular_train_per_class"], _ = render_epistemic_config(
                    class_names,
                    default_under_train=50,
                    default_regular_train=300
                )
        else:
            # Both epistemic parameters are being swept - just get under_supported
            config["under_supported"] = st.selectbox(
                "Under-supported Classes",
                ["0,1", "random:2", "random:3"],
                help="Classes with limited training data",
                key="batch_base_under_supported_only"
            )
            config["under_train_per_class"] = None
            config["regular_train_per_class"] = None
        
        # Aleatoric Configuration
        st.markdown("#### 🎲 Aleatoric Uncertainty Configuration")
        
        # If sweeping aleatoric_noise_percentage, force random flipping mode and disable UI
        if swept_parameter == "aleatoric_noise_percentage":
            st.info("🔄 **Sweeping aleatoric noise**: Using random label flipping mode (controlled noise)")
            st.caption("⚠️ Noise source is locked to 'random flipping' when sweeping aleatoric_noise_percentage")
            config["noise_source"] = "Add random label flipping (0-100%, sweepable)"
            config["custom_noise_rate"] = None  # Will be filled by sweep
        elif swept_parameter not in ["aleatoric_noise_percentage"]:
            config["noise_source"], config["custom_noise_rate"] = render_aleatoric_config(
                stats,
                noise_type,
                class_names,
                fetch_dataset_stats,
                dataset_name,
                key_prefix="batch_"
            )
        else:
            # Shouldn't reach here, but set defaults
            config["noise_source"] = "Add random label flipping (0-100%, sweepable)"
            config["custom_noise_rate"] = None
        
        st.markdown("---")
        
        # Model Configuration (if not being swept)
        if swept_parameter not in ["hidden_dim", "dropout"]:
            st.markdown("#### 🧠 Model Architecture")
            
            # Architecture selector
            config["architecture"] = st.selectbox(
                "Architecture",
                options=["resnet18_mcdropout", "cnn_mcdropout", "dinov2_mlp"],
                format_func=lambda x: {
                    "dinov2_mlp": "🎯 DINOv2 + MLP (Feature Space)",
                    "cnn_mcdropout": "🔷 CNN with MC Dropout (End-to-End)",
                    "resnet18_mcdropout": "🔶 ResNet18 with MC Dropout (End-to-End)"
                }[x],
                key="batch_base_architecture",
                help="ResNet18: High-capacity end-to-end | CNN: Lightweight end-to-end | DINOv2: Fast, uses pretrained features"
            )
            
            # Auto-set training mode
            config["training_mode"] = "feature_space" if config["architecture"] == "dinov2_mlp" else "end_to_end"
            
            # Architecture-specific parameters
            if config["architecture"] == "dinov2_mlp":
                col1, col2, col3 = st.columns(3)
                with col1:
                    config["dinov2_model"] = st.selectbox(
                        "DINOv2 Size",
                        ["small", "base", "large"],
                        index=0,
                        key="batch_base_dinov2"
                    )
                with col2:
                    if swept_parameter != "hidden_dim":
                        config["hidden_dim"] = st.number_input(
                            "Hidden Dim",
                            min_value=64,
                            max_value=1024,
                            value=256,
                            step=64,
                            key="batch_base_hidden_dim"
                        )
                    else:
                        config["hidden_dim"] = None
                with col3:
                    if swept_parameter != "dropout":
                        config["dropout"] = st.number_input(
                            "Dropout",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.2,
                            step=0.1,
                            key="batch_base_dropout"
                        )
                    else:
                        config["dropout"] = None
                
                config["num_conv_layers"] = 3
                config["conv_channels"] = [32, 64, 64]
                
            elif config["architecture"] == "cnn_mcdropout":
                col1, col2 = st.columns(2)
                with col1:
                    config["num_conv_layers"] = st.number_input(
                        "Conv Layers",
                        min_value=2,
                        max_value=5,
                        value=3,
                        key="batch_base_conv_layers"
                    )
                with col2:
                    conv_channels_str = st.text_input(
                        "Conv Channels",
                        "32,64,64",
                        key="batch_base_conv_channels"
                    )
                    config["conv_channels"] = [int(x.strip()) for x in conv_channels_str.split(",")]
                
                col3, col4 = st.columns(2)
                with col3:
                    if swept_parameter != "hidden_dim":
                        config["hidden_dim"] = st.number_input(
                            "Hidden Dim",
                            min_value=64,
                            max_value=512,
                            value=128,
                            step=64,
                            key="batch_base_hidden_dim"
                        )
                    else:
                        config["hidden_dim"] = None
                with col4:
                    if swept_parameter != "dropout":
                        config["dropout"] = st.number_input(
                            "Dropout",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.5,
                            step=0.1,
                            key="batch_base_dropout"
                        )
                    else:
                        config["dropout"] = None
                
                config["dinov2_model"] = "small"
                
            else:  # resnet18_mcdropout
                col1, col2 = st.columns(2)
                with col1:
                    if swept_parameter != "hidden_dim":
                        config["hidden_dim"] = st.number_input(
                            "Hidden Dim",
                            min_value=128,
                            max_value=1024,
                            value=512,
                            step=64,
                            key="batch_base_hidden_dim"
                        )
                    else:
                        config["hidden_dim"] = None
                with col2:
                    if swept_parameter != "dropout":
                        config["dropout"] = st.number_input(
                            "Dropout",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.3,
                            step=0.1,
                            key="batch_base_dropout"
                        )
                    else:
                        config["dropout"] = None
                
                config["dinov2_model"] = "small"
                config["num_conv_layers"] = 3
                config["conv_channels"] = [32, 64, 64]
        else:
            # Set defaults when model params are being swept
            config["architecture"] = "dinov2_mlp"
            config["training_mode"] = "feature_space"
            config["dinov2_model"] = "small"
            config["num_conv_layers"] = 3
            config["conv_channels"] = [32, 64, 64]
        
        st.markdown("---")
        
        # Training Configuration (if not being swept)
        if swept_parameter not in ["epochs", "learning_rate", "weight_decay", "train_batch_size"]:
            st.markdown("#### 🎯 Training Hyperparameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                if swept_parameter != "epochs":
                    config["epochs"] = st.number_input(
                        "Training Epochs",
                        min_value=1,
                        max_value=100,
                        value=12,
                        key="batch_base_epochs"
                    )
                else:
                    config["epochs"] = None  # Will use sweep values
            with col2:
                if swept_parameter != "learning_rate":
                    config["learning_rate"] = st.number_input(
                        "Learning Rate",
                        min_value=0.0001,
                        max_value=0.1,
                        value=0.001,
                        format="%.4f",
                        key="batch_base_lr"
                    )
                else:
                    config["learning_rate"] = None  # Will use sweep values
            with col3:
                if swept_parameter != "weight_decay":
                    config["weight_decay"] = st.number_input(
                        "Weight Decay",
                        min_value=0.0,
                        max_value=0.01,
                        value=0.0001,
                        format="%.5f",
                        key="batch_base_wd"
                    )
                else:
                    config["weight_decay"] = None  # Will use sweep values
            
            if swept_parameter != "train_batch_size":
                config["train_batch_size"] = st.number_input(
                    "Batch Size",
                    min_value=16,
                    max_value=512,
                    value=256,
                    step=16,
                    key="batch_base_batch_size"
                )
            else:
                config["train_batch_size"] = None  # Will use sweep values
        
        st.markdown("---")
        
        # Evaluation Configuration (if not being swept)
        if swept_parameter not in ["mc_passes", "eval_per_group"]:
            st.markdown("#### 📊 Evaluation Configuration")
            col1, col2 = st.columns(2)
            with col1:
                if swept_parameter != "mc_passes":
                    config["mc_passes"] = st.number_input(
                        "MC Dropout Passes",
                        min_value=5,
                        max_value=100,
                        value=20,
                        help="Number of forward passes for uncertainty estimation",
                        key="batch_base_mc_passes"
                    )
                else:
                    config["mc_passes"] = None  # Will use sweep values
            with col2:
                if swept_parameter != "eval_per_group":
                    config["eval_per_group"] = st.number_input(
                        "Evaluation samples per group",
                        min_value=100,
                        max_value=2000,
                        value=100,
                        step=100,
                        help="Samples for each evaluation group (clean, noisy, under-supported)",
                        key="batch_base_eval_per_group"
                    )
                else:
                    config["eval_per_group"] = None  # Will use sweep values
    
    return config


def _generate_sweep_preview(
    start: float,
    end: float,
    step: float,
    value_type: str,
) -> List[float]:
    """
    Generate preview values for the batch sweep UI.
    """
    if step <= 0 or end < start:
        return []

    values: List[float] = []
    current = start
    epsilon = abs(step) / 1000.0

    while current <= end + epsilon and len(values) <= 101:
        if value_type == "int":
            values.append(int(round(current)))
        else:
            values.append(round(current, 10))
        current += step

    return values

# Made with Bob
