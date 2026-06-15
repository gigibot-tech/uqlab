"""
Streamlit UI components for watsonx.ai cloud inference integration.

This module provides UI components to enable cloud-based inference using
watsonx.ai deployed models, allowing users to switch between local and
cloud inference modes.
"""

import streamlit as st
import torch
from typing import Optional, Dict, Any
from pathlib import Path


def render_cloud_mode_toggle() -> tuple[bool, Optional[Dict[str, str]]]:
    """
    Render cloud mode toggle and configuration.
    
    Returns:
        Tuple of (cloud_enabled, cloud_config)
        - cloud_enabled: Whether cloud mode is enabled
        - cloud_config: Dictionary with watsonx.ai credentials if enabled
    """
    st.markdown("### ☁️ Inference Mode")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        cloud_enabled = st.toggle(
            "Cloud Mode",
            value=False,
            help="Use watsonx.ai for inference instead of local computation"
        )
    
    with col2:
        if cloud_enabled:
            st.info("🌐 Using watsonx.ai for inference (faster, server-side)")
        else:
            st.info("💻 Using local inference (slower, client-side)")
    
    cloud_config = None
    
    if cloud_enabled:
        st.markdown("#### watsonx.ai Configuration")
        
        with st.expander("🔑 Cloud Credentials", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    help="Your watsonx.ai API key"
                )
                space_id = st.text_input(
                    "Space ID",
                    help="watsonx.ai deployment space ID"
                )
            
            with col_b:
                deployment_id = st.text_input(
                    "Deployment ID",
                    help="Model deployment ID"
                )
                region = st.selectbox(
                    "Region",
                    ["us-south", "eu-de", "eu-gb", "jp-tok"],
                    help="watsonx.ai region"
                )
            
            if api_key and space_id and deployment_id:
                cloud_config = {
                    "api_key": api_key,
                    "space_id": space_id,
                    "deployment_id": deployment_id,
                    "region": region
                }
                st.success("✅ Cloud credentials configured")
            else:
                st.warning("⚠️ Please provide all cloud credentials")
    
    return cloud_enabled, cloud_config


def render_cloud_inference_status(
    cloud_enabled: bool,
    num_samples: int,
    estimated_time_local: float,
    estimated_time_cloud: float
) -> None:
    """
    Display inference time comparison between local and cloud modes.
    
    Args:
        cloud_enabled: Whether cloud mode is enabled
        num_samples: Number of samples to process
        estimated_time_local: Estimated local inference time (seconds)
        estimated_time_cloud: Estimated cloud inference time (seconds)
    """
    st.markdown("#### ⚡ Inference Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Samples",
            f"{num_samples:,}",
            help="Number of samples to process"
        )
    
    with col2:
        if cloud_enabled:
            st.metric(
                "Cloud Time",
                f"{estimated_time_cloud:.1f}s",
                delta=f"-{estimated_time_local - estimated_time_cloud:.1f}s",
                delta_color="normal",
                help="Estimated inference time using watsonx.ai"
            )
        else:
            st.metric(
                "Local Time",
                f"{estimated_time_local:.1f}s",
                help="Estimated inference time using local GPU/CPU"
            )
    
    with col3:
        speedup = estimated_time_local / estimated_time_cloud if estimated_time_cloud > 0 else 1
        st.metric(
            "Speedup",
            f"{speedup:.1f}x",
            help="Cloud vs local inference speedup"
        )
    
    # Visual comparison
    if not cloud_enabled:
        st.info(
            f"💡 **Tip**: Enable Cloud Mode to reduce inference time from "
            f"{estimated_time_local:.1f}s to {estimated_time_cloud:.1f}s "
            f"({speedup:.1f}x faster)"
        )


def render_cloud_signals_info(cloud_enabled: bool, has_dualxda: bool) -> None:
    """
    Display information about uncertainty signals available in cloud mode.
    
    Args:
        cloud_enabled: Whether cloud mode is enabled
        has_dualxda: Whether DualXDA attribution is available
    """
    st.markdown("#### 📊 Uncertainty Signals")
    
    if cloud_enabled:
        if has_dualxda:
            st.success(
                "✅ **Full Signal Coverage** (11 signals)\n\n"
                "Cloud deployment includes DualXDA attribution:\n"
                "- 4 Predictive signals (MSP, entropy, mutual info, inverse logit)\n"
                "- 7 Attribution signals (mass, coherence, dominance, etc.)"
            )
        else:
            st.info(
                "ℹ️ **Standard Signal Coverage** (7 signals)\n\n"
                "Cloud deployment without DualXDA:\n"
                "- 4 Predictive signals (MSP, entropy, mutual info, inverse logit)\n"
                "- 3 Approximated attribution signals"
            )
    else:
        st.info(
            "💻 **Local Inference**\n\n"
            "All signals computed locally:\n"
            "- MC Dropout for epistemic uncertainty\n"
            "- DualXDA attribution (if available)\n"
            "- Full signal computation"
        )


def create_cloud_inference_client(cloud_config: Dict[str, str]):
    """
    Create watsonx.ai scoring client from configuration.
    
    Args:
        cloud_config: Dictionary with watsonx.ai credentials
            - api_key: IBM Cloud API key
            - space_id: Deployment space ID
            - deployment_id: Model deployment ID
            - region: watsonx.ai region (us-south, eu-de, etc.)
        
    Returns:
        WatsonxScoringClient instance
    """
    from uqlab.classification.watsonx_scoring import WatsonxScoringClient
    
    # Build scoring URL from region and deployment ID
    region = cloud_config["region"]
    deployment_id = cloud_config["deployment_id"]
    
    # watsonx.ai scoring endpoint format
    scoring_url = (
        f"https://{region}.ml.cloud.ibm.com/ml/v4/deployments/"
        f"{deployment_id}/predictions"
    )
    
    return WatsonxScoringClient(
        api_key=cloud_config["api_key"],
        scoring_url=scoring_url,
        space_id=cloud_config["space_id"]
    )


def run_cloud_inference(
    client,
    embeddings: torch.Tensor,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Run inference using watsonx.ai cloud deployment.
    
    Args:
        client: WatsonxScoringClient instance
        embeddings: Input embeddings [N, feature_dim]
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with predictions and uncertainty signals
    """
    import numpy as np
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Run batch inference
    results = client.score_batch(
        embeddings_np,
        batch_size=32,
        progress_callback=progress_callback
    )
    
    return results


def render_cloud_deployment_guide() -> None:
    """
    Display guide for deploying models to watsonx.ai.
    """
    with st.expander("📚 How to Deploy to watsonx.ai", expanded=False):
        st.markdown("""
        ### Deployment Steps
        
        1. **Export Deployment Package**
           ```python
           from uqlab.classification.watsonx_export import export_all_for_watsonx
           
           export_all_for_watsonx(
               model=model,
               train_embeddings=train_embeddings,
               eval_embeddings=eval_embeddings,
               # ... other parameters
               output_base_dir="./watsonx_deployment"
           )
           ```
        
        2. **Upload to watsonx.ai**
           - Log in to IBM Cloud
           - Navigate to watsonx.ai
           - Create deployment space
           - Upload model checkpoint and data files
        
        3. **Deploy Model**
           - Create deployment from uploaded model
           - Configure resources (CPU/GPU)
           - Note deployment ID
        
        4. **Get Credentials**
           - API Key: From IBM Cloud IAM
           - Space ID: From deployment space settings
           - Deployment ID: From deployment details
        
        5. **Use in Streamlit**
           - Enable Cloud Mode toggle
           - Enter credentials
           - Run inference!
        
        ### Benefits
        
        ✅ **Faster**: Server-side computation (1-2s vs 10-35s)  
        ✅ **Scalable**: Handle large batches efficiently  
        ✅ **Consistent**: Same signals as local inference  
        ✅ **Monitored**: Built-in governance and logging  
        
        ### Documentation
        
        - See `WATSONX_DEPLOYMENT_GUIDE.md` for detailed instructions
        - See `WATSONX_DUALXDA_INTEGRATION.md` for DualXDA setup
        - See `watsonx_dualxda_example.py` for code examples
        """)


def estimate_inference_time(
    num_samples: int,
    mc_passes: int,
    has_attribution: bool,
    cloud_mode: bool
) -> float:
    """
    Estimate inference time based on configuration.
    
    Args:
        num_samples: Number of samples
        mc_passes: Number of MC dropout passes
        has_attribution: Whether attribution signals are computed
        cloud_mode: Whether using cloud inference
        
    Returns:
        Estimated time in seconds
    """
    if cloud_mode:
        # Cloud inference: ~0.05s per sample (batched)
        base_time = num_samples * 0.05
        if has_attribution:
            base_time *= 1.2  # 20% overhead for attribution
        return base_time
    else:
        # Local inference: ~0.1s per sample per MC pass
        base_time = num_samples * mc_passes * 0.1
        if has_attribution:
            base_time += num_samples * 2.0  # 2s per sample for DualXDA
        return base_time


# Made with Bob