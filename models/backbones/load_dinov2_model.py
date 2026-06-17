"""
DINOv2 Model Loading and Inspection Utilities

Comprehensive utilities for loading, inspecting, and analyzing DINOv2 models
for academic research with detailed architecture inspection capabilities.
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from typing import Optional, Union, List, Dict, Tuple, Any
from pathlib import Path
import numpy as np
from PIL import Image
import json
from collections import OrderedDict
import time
from datetime import datetime


# Model configuration
AVAILABLE_MODELS = {
    'small': 'facebook/dinov2-small',
    'base': 'facebook/dinov2-base',
    'large': 'facebook/dinov2-large',
    'giant': 'facebook/dinov2-giant',
    'small-reg': 'facebook/dinov2-small-reg',
    'base-reg': 'facebook/dinov2-base-reg',
    'large-reg': 'facebook/dinov2-large-reg',
    'giant-reg': 'facebook/dinov2-giant-reg',
}

FEATURE_DIMS = {
    'small': 384,
    'base': 768,
    'large': 1024,
    'giant': 1536,
}


# ============================================================================
# Core Loading Functions
# ============================================================================

def load_model(model_name: str = 'large', device: str = 'cpu') -> nn.Module:
    """
    Load pre-trained DINOv2 model.
    
    Args:
        model_name: Model variant ('small', 'base', 'large', 'giant', with optional '-reg')
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Loaded DINOv2 model
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_id = AVAILABLE_MODELS[model_name]
    model = AutoModel.from_pretrained(model_id)
    return model.to(device)


def load_processor(model_name: str = 'large') -> AutoImageProcessor:
    """Load image processor for DINOv2 model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_id = AVAILABLE_MODELS[model_name]
    return AutoImageProcessor.from_pretrained(model_id)


def load_model_and_processor(
    model_name: str = 'large',
    device: str = 'cpu',
    verbose: bool = False
) -> Tuple[nn.Module, AutoImageProcessor]:
    """
    Load both model and processor.
    
    Args:
        model_name: Model variant
        device: Target device
        verbose: Print loading information
        
    Returns:
        (model, processor) tuple
    """
    if verbose:
        print(f"Loading DINOv2 model: {model_name}")
        print(f"Device: {device}")
    
    model = load_model(model_name, device)
    processor = load_processor(model_name)
    
    if verbose:
        base_name = model_name.replace('-reg', '')
        print(f"Feature dimension: {FEATURE_DIMS[base_name]}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, processor


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str = 'large',
    device: str = 'cpu',
    strict: bool = True
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Base model variant
        device: Target device
        strict: Strict state dict loading
        
    Returns:
        Model with loaded weights
    """
    model = load_model(model_name, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    return model


# ============================================================================
# Architecture Inspection
# ============================================================================

def print_model_architecture(model: nn.Module, max_depth: int = 3) -> None:
    """
    Print model architecture hierarchy.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth to display
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    
    def print_module(module, name, depth=0):
        if depth > max_depth:
            return
        
        indent = "  " * depth
        params = sum(p.numel() for p in module.parameters())
        print(f"{indent}{name}: {type(module).__name__} [{params:,} params]")
        
        if depth < max_depth:
            for child_name, child in module.named_children():
                print_module(child, child_name, depth + 1)
    
    print_module(model, "model", 0)
    print()


def print_state_dict_keys(model: nn.Module, show_shapes: bool = True) -> None:
    """
    Print all state dict keys with shapes.
    
    Args:
        model: PyTorch model
        show_shapes: Display tensor shapes
    """
    print("=" * 80)
    print("STATE DICT KEYS")
    print("=" * 80)
    
    state_dict = model.state_dict()
    for key, tensor in state_dict.items():
        if show_shapes:
            print(f"{key}: {list(tensor.shape)}")
        else:
            print(key)
    
    print(f"\nTotal keys: {len(state_dict)}")
    print()


def print_layer_details(model: nn.Module, layer_type: type = nn.Linear) -> None:
    """
    Print details for specific layer type.
    
    Args:
        model: PyTorch model
        layer_type: Layer type to inspect (e.g., nn.Linear, nn.LayerNorm)
    """
    print("=" * 80)
    print(f"{layer_type.__name__} LAYERS")
    print("=" * 80)
    
    layers = [(name, module) for name, module in model.named_modules() 
              if isinstance(module, layer_type)]
    
    for name, module in layers:
        print(f"\n{name}:")
        if isinstance(module, nn.Linear):
            print(f"  in_features: {module.in_features}")
            print(f"  out_features: {module.out_features}")
            print(f"  bias: {module.bias is not None}")
        elif isinstance(module, nn.LayerNorm):
            print(f"  normalized_shape: {module.normalized_shape}")
            print(f"  eps: {module.eps}")
        elif isinstance(module, nn.Dropout):
            print(f"  p: {module.p}")
    
    print(f"\nTotal {layer_type.__name__} layers: {len(layers)}")
    print()


def get_attention_info(model: nn.Module) -> Dict[str, Any]:
    """
    Extract attention mechanism information.
    
    Args:
        model: DINOv2 model
        
    Returns:
        Dictionary with attention details
    """
    config = model.config
    # Calculate intermediate_size safely - DINOv2 uses mlp_ratio instead
    intermediate_size = getattr(config, 'intermediate_size', None)
    if intermediate_size is None and hasattr(config, 'mlp_ratio'):
        intermediate_size = int(config.hidden_size * config.mlp_ratio)
    
    return {
        'num_attention_heads': config.num_attention_heads,
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'head_dim': config.hidden_size // config.num_attention_heads,
        'intermediate_size': intermediate_size,
    }


def get_embedding_info(model: nn.Module) -> Dict[str, Any]:
    """
    Extract embedding dimension information.
    
    Args:
        model: DINOv2 model
        
    Returns:
        Dictionary with embedding details
    """
    config = model.config
    num_patches = (config.image_size // config.patch_size) ** 2
    
    return {
        'hidden_size': config.hidden_size,
        'patch_size': config.patch_size,
        'image_size': config.image_size,
        'num_channels': config.num_channels,
        'num_patches': num_patches,
        'sequence_length': num_patches + 1,  # +1 for [CLS] token
    }


def analyze_model(model: nn.Module) -> Dict[str, Any]:
    """
    Comprehensive model analysis.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count layer types
    layer_counts = {}
    for module in model.modules():
        layer_type = type(module).__name__
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'memory_mb': total_params * 4 / 1024 / 1024,  # FP32
        'layer_counts': layer_counts,
        'num_modules': len(list(model.modules())),
    }


def print_parameter_statistics(model: nn.Module) -> None:
    """
    Print parameter statistics per layer.
    
    Args:
        model: PyTorch model
    """
    print("=" * 80)
    print("PARAMETER STATISTICS")
    print("=" * 80)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\n{name}:")
            print(f"  Shape: {list(param.shape)}")
            print(f"  Size: {param.numel():,}")
            print(f"  Mean: {param.data.mean().item():.6f}")
            print(f"  Std: {param.data.std().item():.6f}")
            print(f"  Min: {param.data.min().item():.6f}")
            print(f"  Max: {param.data.max().item():.6f}")
    print()


def export_architecture_summary(model: nn.Module, output_path: Union[str, Path]) -> None:
    """
    Export architecture summary to JSON.
    
    Args:
        model: PyTorch model
        output_path: Path to save JSON file
    """
    analysis = analyze_model(model)
    attention_info = get_attention_info(model)
    embedding_info = get_embedding_info(model)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_type': type(model).__name__,
        'analysis': analysis,
        'attention': attention_info,
        'embeddings': embedding_info,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Architecture summary saved to: {output_path}")


def compare_model_variants(model_names: List[str]) -> None:
    """
    Compare different DINOv2 model variants.
    
    Args:
        model_names: List of model names to compare
    """
    print("=" * 80)
    print("MODEL VARIANT COMPARISON")
    print("=" * 80)
    
    for model_name in model_names:
        base_name = model_name.replace('-reg', '')
        model_id = AVAILABLE_MODELS[model_name]
        feature_dim = FEATURE_DIMS[base_name]
        
        print(f"\n{model_name}:")
        print(f"  Model ID: {model_id}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Has registers: {'Yes' if '-reg' in model_name else 'No'}")
    print()


# ============================================================================
# Image Preprocessing
# ============================================================================

def preprocess_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    processor: AutoImageProcessor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Preprocess image for DINOv2.
    
    Args:
        image: Input image (PIL, numpy, or path)
        processor: Image processor
        device: Target device
        
    Returns:
        Preprocessed tensor [1, C, H, W]
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(device)


def preprocess_batch(
    images: List[Union[Image.Image, np.ndarray, str, Path]],
    processor: AutoImageProcessor,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Preprocess batch of images.
    
    Args:
        images: List of images
        processor: Image processor
        device: Target device
        
    Returns:
        Preprocessed tensor [B, C, H, W]
    """
    loaded_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        loaded_images.append(img)
    
    inputs = processor(images=loaded_images, return_tensors="pt")
    return inputs['pixel_values'].to(device)


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    use_cls_token: bool = True
) -> torch.Tensor:
    """
    Extract features from images.
    
    Args:
        model: DINOv2 model
        images: Input images [B, C, H, W]
        use_cls_token: Use [CLS] token (True) or mean pooling (False)
        
    Returns:
        Features [B, D]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=images)
        hidden_states = outputs.last_hidden_state  # [B, N, D]
        
        if use_cls_token:
            features = hidden_states[:, 0]  # [CLS] token
        else:
            features = hidden_states.mean(dim=1)  # Mean pooling
    
    return features


def get_intermediate_activations(
    model: nn.Module,
    images: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate layer activations.
    
    Args:
        model: DINOv2 model
        images: Input images [B, C, H, W]
        layer_names: Specific layers to extract (None = all)
        
    Returns:
        Dictionary mapping layer names to activations
    """
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook
    
    # Register hooks
    if layer_names is None:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hooks.append(module.register_forward_hook(get_activation(name)))
    else:
        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(pixel_values=images)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations


# ============================================================================
# Uncertainty Quantification
# ============================================================================

def predict_with_uncertainty(
    model: nn.Module,
    classifier: nn.Module,
    images: torch.Tensor,
    n_passes: int = 10
) -> Dict[str, torch.Tensor]:
    """
    MC Dropout prediction with uncertainty estimation.
    
    Args:
        model: DINOv2 backbone
        classifier: Classification head with dropout
        images: Input images [B, C, H, W]
        n_passes: Number of MC forward passes
        
    Returns:
        Dictionary with predictions and uncertainty metrics
    """
    model.eval()
    
    # Enable dropout in classifier
    for module in classifier.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    
    # Extract features once
    with torch.no_grad():
        features = extract_features(model, images)
    
    # MC forward passes
    all_probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = classifier(features)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
    
    all_probs = torch.stack(all_probs)  # [T, B, C]
    
    # Calculate statistics
    mean_probs = all_probs.mean(dim=0)
    std_probs = all_probs.std(dim=0)
    predictions = torch.argmax(mean_probs, dim=-1)
    
    # Entropy metrics
    eps = 1e-10
    predictive_entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)
    entropies = -(all_probs * torch.log(all_probs + eps)).sum(dim=-1)
    expected_entropy = entropies.mean(dim=0)
    mutual_information = predictive_entropy - expected_entropy
    
    return {
        'predictions': predictions,
        'mean_probs': mean_probs,
        'std_probs': std_probs,
        'predictive_entropy': predictive_entropy,
        'mutual_information': mutual_information,
        'all_probs': all_probs,
    }


# ============================================================================
# Utilities
# ============================================================================

def save_model(
    model: nn.Module,
    save_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        metadata: Additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to: {save_path}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model info
    """
    return {
        'model_type': type(model).__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(next(model.parameters()).device),
        'dtype': str(next(model.parameters()).dtype),
    }


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    print("DINOv2 Model Loading Utilities - Example Usage")
    print("=" * 80)
    
    # Load model
    print("\n1. Loading model...")
    model, processor = load_model_and_processor('base', device='cpu', verbose=True)
    
    # Print architecture
    print("\n2. Model architecture:")
    print_model_architecture(model, max_depth=2)
    
    # Analyze model
    print("\n3. Model analysis:")
    analysis = analyze_model(model)
    for key, value in analysis.items():
        if key != 'layer_counts':
            print(f"  {key}: {value}")
    
    # Attention info
    print("\n4. Attention mechanism:")
    attn_info = get_attention_info(model)
    for key, value in attn_info.items():
        print(f"  {key}: {value}")
    
    # Embedding info
    print("\n5. Embedding dimensions:")
    emb_info = get_embedding_info(model)
    for key, value in emb_info.items():
        print(f"  {key}: {value}")
    
    # Compare variants
    print("\n6. Model variants:")
    compare_model_variants(['small', 'base', 'large'])
    
    print("\n✅ All examples completed!")

# Made with Bob
