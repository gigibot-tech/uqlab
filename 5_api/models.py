"""
Model Serving API Endpoints

Consolidated model serving endpoints for loading models and running inference.
Handles model management, caching, and batch inference.
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict

import torch
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from app.api.deps import CurrentUser, SessionDep
from app.tables import UncertaintyExperiment, JobStatus

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Global Model Cache
# ============================================================================

class ModelCache:
    """Simple in-memory model cache."""
    
    def __init__(self, max_size: int = 5):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, model_id: str) -> Optional[Any]:
        """Get model from cache."""
        if model_id in self.cache:
            self.access_times[model_id] = datetime.utcnow()
            return self.cache[model_id]
        return None
    
    def put(self, model_id: str, model: Any) -> None:
        """Add model to cache with LRU eviction."""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            oldest_id = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_id]
            del self.access_times[oldest_id]
            logger.info(f"Evicted model {oldest_id} from cache")
        
        self.cache[model_id] = model
        self.access_times[model_id] = datetime.utcnow()
        logger.info(f"Cached model {model_id}")
    
    def clear(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Model cache cleared")


# Global cache instance
_model_cache = ModelCache(max_size=5)


# ============================================================================
# Request/Response Models
# ============================================================================

class ModelLoadRequest(BaseModel):
    """Request to load a model."""
    experiment_id: uuid.UUID = Field(description="Experiment ID to load model from")
    checkpoint: str | None = Field(
        None,
        description="Checkpoint name (e.g., 'best', 'last', 'epoch_10')"
    )


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    experiment_id: uuid.UUID
    experiment_name: str
    checkpoint: str
    architecture: str
    num_parameters: int
    input_shape: List[int]
    num_classes: int
    loaded_at: datetime
    device: str


class InferenceRequest(BaseModel):
    """Request for model inference."""
    model_id: str = Field(description="Loaded model ID")
    images: List[List[List[List[float]]]] = Field(
        description="Batch of images as 4D array [batch, channels, height, width]"
    )
    return_uncertainty: bool = Field(
        default=True,
        description="Whether to return uncertainty estimates"
    )
    mc_samples: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of MC dropout samples for uncertainty"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_id: int
    class_name: str | None
    confidence: float
    probabilities: List[float]
    aleatoric_uncertainty: float | None = None
    epistemic_uncertainty: float | None = None


class InferenceResponse(BaseModel):
    """Inference response."""
    model_id: str
    predictions: List[PredictionResult]
    inference_time_ms: float
    batch_size: int


class ModelListResponse(BaseModel):
    """Response for listing loaded models."""
    models: List[ModelInfo]
    total: int


# ============================================================================
# Helper Functions
# ============================================================================

def load_model_from_checkpoint(
    experiment_id: uuid.UUID,
    checkpoint: str,
    session: Session
) -> tuple[Any, Dict[str, Any]]:
    """Load model from experiment checkpoint."""
    # Get experiment
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    if experiment.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment not completed (status: {experiment.status})"
        )
    
    # Find checkpoint file
    if not experiment.results_path:
        raise HTTPException(
            status_code=400,
            detail="Experiment has no results path"
        )
    
    results_dir = Path(experiment.results_path)
    if not results_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Results directory not found: {results_dir}"
        )
    
    # Determine checkpoint path
    checkpoint_name = checkpoint or "best"
    checkpoint_path = results_dir / "checkpoints" / f"{checkpoint_name}.pt"
    
    if not checkpoint_path.exists():
        # Try alternative naming
        checkpoint_path = results_dir / f"{checkpoint_name}_checkpoint.pt"
    
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint not found: {checkpoint_name}"
        )
    
    # Load checkpoint
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model and metadata
        if isinstance(checkpoint_data, dict):
            model_state = checkpoint_data.get("model_state_dict", checkpoint_data)
            metadata = {
                "epoch": checkpoint_data.get("epoch", 0),
                "architecture": checkpoint_data.get("architecture", "unknown"),
                "num_classes": checkpoint_data.get("num_classes", 10),
            }
        else:
            model_state = checkpoint_data
            metadata = {"architecture": "unknown", "num_classes": 10}
        
        # TODO: Reconstruct model architecture from config
        # For now, return state dict and metadata
        return model_state, metadata
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load checkpoint: {str(e)}"
        )


def count_parameters(model_state: Dict[str, torch.Tensor]) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))


def run_inference(
    model_state: Dict[str, torch.Tensor],
    images: np.ndarray,
    mc_samples: int = 10,
    return_uncertainty: bool = True
) -> List[Dict[str, Any]]:
    """Run inference on batch of images."""
    # TODO: Implement actual inference
    # This is a placeholder that returns mock predictions
    
    batch_size = len(images)
    num_classes = 10  # Default for CIFAR-10
    
    results = []
    for i in range(batch_size):
        # Mock prediction
        probs = np.random.dirichlet(np.ones(num_classes))
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        
        result = {
            "class_id": class_id,
            "class_name": f"class_{class_id}",
            "confidence": confidence,
            "probabilities": probs.tolist(),
        }
        
        if return_uncertainty:
            # Mock uncertainty estimates
            result["aleatoric_uncertainty"] = float(np.random.uniform(0.1, 0.5))
            result["epistemic_uncertainty"] = float(np.random.uniform(0.1, 0.5))
        
        results.append(result)
    
    return results


# ============================================================================
# API Endpoints - No Auth (Development)
# ============================================================================

@router.post("/no-auth/load", response_model=ModelInfo)
async def load_model_no_auth(
    request: ModelLoadRequest,
    session: SessionDep,
) -> Any:
    """Load a model from experiment checkpoint (no authentication for local testing)."""
    # Check if already loaded
    model_id = f"{request.experiment_id}_{request.checkpoint or 'best'}"
    
    cached_model = _model_cache.get(model_id)
    if cached_model:
        logger.info(f"Model {model_id} already loaded (from cache)")
        return cached_model["info"]
    
    # Load model
    model_state, metadata = load_model_from_checkpoint(
        request.experiment_id,
        request.checkpoint or "best",
        session
    )
    
    # Get experiment info
    experiment = session.get(UncertaintyExperiment, request.experiment_id)
    
    # Create model info
    info = ModelInfo(
        model_id=model_id,
        experiment_id=request.experiment_id,
        experiment_name=experiment.name,
        checkpoint=request.checkpoint or "best",
        architecture=metadata.get("architecture", "unknown"),
        num_parameters=count_parameters(model_state),
        input_shape=[3, 32, 32],  # Default CIFAR-10 shape
        num_classes=metadata.get("num_classes", 10),
        loaded_at=datetime.utcnow(),
        device="cpu",
    )
    
    # Cache model
    _model_cache.put(model_id, {
        "state": model_state,
        "metadata": metadata,
        "info": info,
    })
    
    return info


@router.post("/no-auth/inference", response_model=InferenceResponse)
async def run_inference_no_auth(
    request: InferenceRequest,
    session: SessionDep,
) -> Any:
    """Run inference with loaded model (no authentication for local testing)."""
    # Get model from cache
    cached_model = _model_cache.get(request.model_id)
    if not cached_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {request.model_id}. Load it first with /models/load"
        )
    
    # Convert images to numpy array
    try:
        images = np.array(request.images, dtype=np.float32)
        if images.ndim != 4:
            raise ValueError(f"Expected 4D array, got {images.ndim}D")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {str(e)}"
        )
    
    # Run inference
    start_time = datetime.utcnow()
    
    predictions = run_inference(
        cached_model["state"],
        images,
        mc_samples=request.mc_samples,
        return_uncertainty=request.return_uncertainty
    )
    
    end_time = datetime.utcnow()
    inference_time_ms = (end_time - start_time).total_seconds() * 1000
    
    return InferenceResponse(
        model_id=request.model_id,
        predictions=[PredictionResult(**pred) for pred in predictions],
        inference_time_ms=inference_time_ms,
        batch_size=len(images),
    )


@router.get("/no-auth/{model_id}/info", response_model=ModelInfo)
async def get_model_info_no_auth(
    model_id: str,
    session: SessionDep,
) -> Any:
    """Get information about a loaded model (no authentication for local testing)."""
    cached_model = _model_cache.get(model_id)
    if not cached_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    return cached_model["info"]


@router.get("/no-auth", response_model=ModelListResponse)
async def list_models_no_auth(
    session: SessionDep,
) -> Any:
    """List all loaded models (no authentication for local testing)."""
    models = [
        cached_model["info"]
        for cached_model in _model_cache.cache.values()
    ]
    
    return ModelListResponse(
        models=models,
        total=len(models),
    )


@router.delete("/no-auth/{model_id}")
async def unload_model_no_auth(
    model_id: str,
    session: SessionDep,
) -> dict:
    """Unload a model from cache (no authentication for local testing)."""
    if model_id not in _model_cache.cache:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    del _model_cache.cache[model_id]
    del _model_cache.access_times[model_id]
    
    return {
        "message": "Model unloaded successfully",
        "model_id": model_id,
    }


@router.post("/no-auth/clear-cache")
async def clear_cache_no_auth(session: SessionDep) -> dict:
    """Clear all cached models (no authentication for local testing)."""
    count = len(_model_cache.cache)
    _model_cache.clear()
    
    return {
        "message": "Model cache cleared",
        "models_cleared": count,
    }


# ============================================================================
# API Endpoints - Authenticated
# ============================================================================

@router.post("/load", response_model=ModelInfo)
async def load_model(
    request: ModelLoadRequest,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Load a model from experiment checkpoint."""
    # Verify user owns experiment
    experiment = session.get(UncertaintyExperiment, request.experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Check if already loaded
    model_id = f"{request.experiment_id}_{request.checkpoint or 'best'}"
    
    cached_model = _model_cache.get(model_id)
    if cached_model:
        return cached_model["info"]
    
    # Load model
    model_state, metadata = load_model_from_checkpoint(
        request.experiment_id,
        request.checkpoint or "best",
        session
    )
    
    # Create model info
    info = ModelInfo(
        model_id=model_id,
        experiment_id=request.experiment_id,
        experiment_name=experiment.name,
        checkpoint=request.checkpoint or "best",
        architecture=metadata.get("architecture", "unknown"),
        num_parameters=count_parameters(model_state),
        input_shape=[3, 32, 32],
        num_classes=metadata.get("num_classes", 10),
        loaded_at=datetime.utcnow(),
        device="cpu",
    )
    
    # Cache model
    _model_cache.put(model_id, {
        "state": model_state,
        "metadata": metadata,
        "info": info,
    })
    
    return info


@router.post("/inference", response_model=InferenceResponse)
async def run_inference_auth(
    request: InferenceRequest,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Run inference with loaded model."""
    # Get model from cache
    cached_model = _model_cache.get(request.model_id)
    if not cached_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {request.model_id}"
        )
    
    # Verify user owns the experiment
    experiment_id = cached_model["info"].experiment_id
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Convert images to numpy array
    try:
        images = np.array(request.images, dtype=np.float32)
        if images.ndim != 4:
            raise ValueError(f"Expected 4D array, got {images.ndim}D")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format: {str(e)}"
        )
    
    # Run inference
    start_time = datetime.utcnow()
    
    predictions = run_inference(
        cached_model["state"],
        images,
        mc_samples=request.mc_samples,
        return_uncertainty=request.return_uncertainty
    )
    
    end_time = datetime.utcnow()
    inference_time_ms = (end_time - start_time).total_seconds() * 1000
    
    return InferenceResponse(
        model_id=request.model_id,
        predictions=[PredictionResult(**pred) for pred in predictions],
        inference_time_ms=inference_time_ms,
        batch_size=len(images),
    )


@router.get("/{model_id}/info", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Get information about a loaded model."""
    cached_model = _model_cache.get(model_id)
    if not cached_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    # Verify user owns the experiment
    experiment_id = cached_model["info"].experiment_id
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return cached_model["info"]


@router.get("", response_model=ModelListResponse)
async def list_models(
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """List all loaded models for current user."""
    # Filter models by user ownership
    user_models = []
    for cached_model in _model_cache.cache.values():
        experiment_id = cached_model["info"].experiment_id
        experiment = session.get(UncertaintyExperiment, experiment_id)
        if experiment and experiment.created_by_id == current_user.id:
            user_models.append(cached_model["info"])
    
    return ModelListResponse(
        models=user_models,
        total=len(user_models),
    )


@router.delete("/{model_id}")
async def unload_model(
    model_id: str,
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, str]:
    """Unload a model from cache."""
    cached_model = _model_cache.get(model_id)
    if not cached_model:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    # Verify user owns the experiment
    experiment_id = cached_model["info"].experiment_id
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    del _model_cache.cache[model_id]
    del _model_cache.access_times[model_id]
    
    return {"message": "Model unloaded successfully"}


# Made with Bob