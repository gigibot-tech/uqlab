"""
Model Selector UI Component

This module provides UI components for browsing, selecting, and loading
saved model checkpoints from completed experiments.
"""

import streamlit as st
import requests
import torch
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import pandas as pd


def render_model_selector(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
) -> Optional[Dict[str, Any]]:
    """
    Render model selector UI for loading saved checkpoints.
    
    Args:
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
    
    Returns:
        Dictionary with selected model info and checkpoint, or None
    """
    st.markdown("### 🎯 Load Saved Model")
    st.markdown("Select a completed experiment to load its trained model checkpoint.")
    
    try:
        # Fetch completed experiments
        response = requests.get(
            f"{api_base_url}/api/v1/experiments/no-auth",
            headers=get_headers_func(),
            timeout=10
        )
        response.raise_for_status()
        experiments = response.json()
        
        # Filter for completed experiments
        completed_experiments = [
            exp for exp in experiments
            if exp.get('status') == 'completed'
        ]
        
        if not completed_experiments:
            st.info("📭 No completed experiments with saved models yet. Run an experiment first!")
            return None
        
        # Sort by completion time (most recent first)
        completed_experiments.sort(
            key=lambda x: x.get('completed_at', ''),
            reverse=True
        )
        
        # Create selection table
        st.markdown(f"**Found {len(completed_experiments)} completed experiments**")
        
        # Build display dataframe
        display_data = []
        for exp in completed_experiments:
            # Extract key metrics (directly from experiment, not nested in results)
            best_epistemic = exp.get('epistemic_auroc', 0.0)
            best_aleatoric = exp.get('aleatoric_auroc', 0.0)
            
            # Try to load config from filesystem for display
            config_path = Path(f"/tmp/walaris_experiments/{exp['id']}/config.yaml")
            model_name = 'N/A'
            epochs = 'N/A'
            
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    model_name = config.get('model', {}).get('dinov2_model', 'N/A')
                    epochs = config.get('training', {}).get('epochs', 'N/A')
                except:
                    pass  # Use N/A if config can't be loaded
            
            display_data.append({
                'ID': exp['id'][:8],  # Short ID
                'Name': exp['name'],
                'Completed': exp.get('completed_at', 'N/A')[:19] if exp.get('completed_at') else 'N/A',
                'Epistemic AUROC': f"{best_epistemic:.3f}" if best_epistemic else 'N/A',
                'Aleatoric AUROC': f"{best_aleatoric:.3f}" if best_aleatoric else 'N/A',
                'Model': model_name,
                'Epochs': str(epochs),  # Convert to string to avoid Arrow serialization error
            })
        
        df = pd.DataFrame(display_data)
        
        # Display table with selection
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Selection dropdown
        experiment_options = {
            f"{exp['name']} ({exp['id'][:8]}) - Epis: {exp.get('epistemic_auroc', 0.0):.3f}": exp['id']
            for exp in completed_experiments
        }
        
        selected_name = st.selectbox(
            "Select experiment to load:",
            options=list(experiment_options.keys()),
            key="model_selector_dropdown"
        )
        
        if not selected_name:
            return None
        
        selected_id = experiment_options[selected_name]
        selected_exp = next(exp for exp in completed_experiments if exp['id'] == selected_id)
        
        # Display selected experiment details
        with st.expander("📊 Selected Experiment Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Configuration:**")
                
                # Load config from filesystem (not stored in database)
                config_path = Path(f"/tmp/walaris_experiments/{selected_id}/config.yaml")
                if config_path.exists():
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    st.json({
                        'model': config.get('model', {}),
                        'training': config.get('training', {}),
                        'data': {
                            'under_supported_classes': config.get('data', {}).get('under_supported_classes'),
                            'under_train_per_class': config.get('data', {}).get('under_train_per_class'),
                            'regular_train_per_class': config.get('data', {}).get('regular_train_per_class'),
                            'noise_type': config.get('data', {}).get('noise_type'),
                        }
                    })
                else:
                    st.warning(f"⚠️ Config file not found at {config_path}")
                    st.caption("The experiment may have been cleaned up.")
            
            with col2:
                st.markdown("**Performance:**")
                
                # Display AUROC scores directly from experiment
                perf_data = {
                    'Metric': ['Epistemic AUROC', 'Aleatoric AUROC'],
                    'Score': [
                        f"{selected_exp.get('epistemic_auroc', 0.0):.4f}",
                        f"{selected_exp.get('aleatoric_auroc', 0.0):.4f}"
                    ]
                }
                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
                
                # Show results path if available
                if selected_exp.get('results_path'):
                    st.caption(f"📁 Results: {selected_exp['results_path']}")
                
                # Load and display training data info from results.pt
                results_pt_path = Path(f"/tmp/walaris_experiments/{selected_id}/results/results.pt")
                if results_pt_path.exists():
                    try:
                        results_data = torch.load(results_pt_path, map_location='cpu', weights_only=False)
                        
                        st.markdown("---")
                        st.markdown("**Training Data:**")
                        
                        train_indices = results_data.get('train_indices')
                        if train_indices is not None:
                            st.text(f"Training samples: {len(train_indices)}")
                            st.text(f"Dataset indices: {train_indices[:10].tolist()}... (showing first 10)")
                            
                            # Show data split info
                            train_labels = results_data.get('train_labels')
                            train_is_noisy = results_data.get('train_is_noisy')
                            if train_labels is not None and train_is_noisy is not None:
                                clean_count = (~train_is_noisy).sum().item()
                                noisy_count = train_is_noisy.sum().item()
                                st.text(f"Clean samples: {clean_count}")
                                st.text(f"Noisy samples: {noisy_count}")
                        
                        eval_indices = results_data.get('eval_indices')
                        if eval_indices is not None:
                            st.text(f"Evaluation samples: {len(eval_indices)}")
                            
                    except Exception as e:
                        st.caption(f"⚠️ Could not load training data info: {str(e)}")
        
        # Load checkpoint button
        if st.button("🔽 Load Model Checkpoint", type="primary", use_container_width=True):
            with st.spinner("Loading model checkpoint..."):
                checkpoint_path = Path(f"/tmp/walaris_experiments/{selected_id}/results/checkpoint.pt")
                
                if not checkpoint_path.exists():
                    st.error(f"❌ Checkpoint not found at {checkpoint_path}")
                    st.info("💡 The experiment may have been cleaned up. Try running a new experiment.")
                    return None
                
                try:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    
                    st.success("✅ Model checkpoint loaded successfully!")
                    
                    # Display checkpoint info
                    st.markdown("**Checkpoint Contents:**")
                    st.json({
                        'epoch': checkpoint.get('epoch'),
                        'config': checkpoint.get('config'),
                        'model_type': str(type(checkpoint.get('model'))),
                        'state_dict_keys': len(checkpoint.get('model_state_dict', {}).keys()),
                    })
                    
                    # Store in session state for use in other components
                    st.session_state['loaded_model'] = checkpoint['model']
                    st.session_state['loaded_model_config'] = checkpoint.get('config')
                    st.session_state['loaded_model_experiment'] = selected_exp
                    
                    return {
                        'experiment': selected_exp,
                        'checkpoint': checkpoint,
                        'checkpoint_path': str(checkpoint_path),
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error loading checkpoint: {str(e)}")
                    return None
        
        # Show if model is already loaded
        if 'loaded_model' in st.session_state:
            loaded_exp = st.session_state.get('loaded_model_experiment', {})
            st.success(f"✅ Model loaded: {loaded_exp.get('name', 'Unknown')} ({loaded_exp.get('id', 'N/A')[:8]})")
            
            if st.button("🗑️ Clear Loaded Model"):
                del st.session_state['loaded_model']
                del st.session_state['loaded_model_config']
                del st.session_state['loaded_model_experiment']
                st.rerun()
        
        # Add data overlap analysis section
        st.markdown("---")
        if selected_name:
            from ..visualization.analysis.data_overlap_analysis import render_data_overlap_analysis
            render_data_overlap_analysis(selected_exp, experiments)
        
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch experiments: {str(e)}")
        return None


def render_model_inference_panel(
    api_base_url: str,
    get_headers_func: Callable[[], Dict],
):
    """
    Render inference panel for loaded model.
    
    Args:
        api_base_url: Base URL for API requests
        get_headers_func: Function to get request headers
    """
    if 'loaded_model' not in st.session_state:
        st.info("💡 Load a model first using the Model Selector above")
        return
    
    st.markdown("### 🔮 Model Inference")
    
    model = st.session_state['loaded_model']
    config = st.session_state.get('loaded_model_config', {})
    experiment = st.session_state.get('loaded_model_experiment', {})
    
    st.markdown(f"**Loaded Model:** {experiment.get('name', 'Unknown')}")
    st.markdown(f"**Architecture:** {config.get('dinov2_model', 'N/A')} + MLP({config.get('hidden_dim', 'N/A')})")
    
    # Inference options
    inference_mode = st.radio(
        "Inference Mode:",
        options=["Single Sample", "Batch Inference", "Uncertainty Analysis"],
        horizontal=True
    )
    
    if inference_mode == "Single Sample":
        st.markdown("**Upload an image for classification:**")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            key="inference_image_upload"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", width=200)
            
            if st.button("🎯 Predict", type="primary"):
                st.info("🚧 Single sample inference coming soon!")
                st.markdown("**Next steps:**")
                st.markdown("- Extract DINOv2 features from uploaded image")
                st.markdown("- Run model.forward() on features")
                st.markdown("- Display prediction + uncertainty scores")
    
    elif inference_mode == "Batch Inference":
        st.markdown("**Run inference on evaluation dataset:**")
        
        if st.button("🚀 Run Batch Inference", type="primary"):
            st.info("🚧 Batch inference coming soon!")
            st.markdown("**Next steps:**")
            st.markdown("- Load evaluation dataset")
            st.markdown("- Extract features for all samples")
            st.markdown("- Run model predictions")
            st.markdown("- Display results table with uncertainties")
    
    else:  # Uncertainty Analysis
        st.markdown("**Analyze uncertainty signals:**")
        
        # Load results if available
        exp_id = experiment.get('id')
        results_path = Path(f"/tmp/walaris_experiments/{exp_id}/results/results.pt")
        
        if results_path.exists():
            try:
                results = torch.load(results_path, map_location='cpu', weights_only=False)
                
                st.markdown("**Available Uncertainty Signals:**")
                signal_table = results.get('signal_table', {})
                
                # Display signal statistics
                signal_stats = []
                for signal_name, values in signal_table.items():
                    signal_stats.append({
                        'Signal': signal_name,
                        'Mean': f"{values.mean():.4f}",
                        'Std': f"{values.std():.4f}",
                        'Min': f"{values.min():.4f}",
                        'Max': f"{values.max():.4f}",
                    })
                
                st.dataframe(pd.DataFrame(signal_stats), use_container_width=True, hide_index=True)
                
                # AUROC results
                st.markdown("**AUROC Performance:**")
                auroc_rows = results.get('auroc_rows', [])
                if auroc_rows:
                    auroc_df = pd.DataFrame([
                        {
                            'Signal': name,
                            'Aleatoric AUROC': f"{alea:.3f}",
                            'Epistemic AUROC': f"{epis:.3f}",
                        }
                        for name, alea, epis in auroc_rows
                    ])
                    st.dataframe(auroc_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error loading results: {str(e)}")
        else:
            st.warning("⚠️ Results file not found. The experiment may have been cleaned up.")


# Made with Bob