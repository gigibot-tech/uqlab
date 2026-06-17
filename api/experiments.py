"""
Experiment Management API Endpoints

Consolidated experiment endpoints with reduced boilerplate.
Handles single experiment creation, execution, and monitoring.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, select, desc

from app.api.deps import CurrentUser, SessionDep
from app.core.security import get_password_hash
from app.domain.models import TrainingConfig, TrainingPresetName
from app.repositories.experiment_repository import ExperimentRepository
from app.tables import JobStatus, UncertaintyExperiment, User

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class ExperimentConfig(TrainingConfig):
    """Experiment configuration matching the simplified flat API."""
    pass


class ExperimentCreate(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(min_length=1, max_length=255)
    preset: TrainingPresetName | None = None
    config: ExperimentConfig | None = None


class ExperimentResponse(BaseModel):
    """Experiment response."""
    id: uuid.UUID
    name: str
    status: JobStatus
    progress: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    aleatoric_auroc: float | None
    epistemic_auroc: float | None
    results_path: str | None
    best_signals_json: str | None = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_test_user(session: Session) -> User:
    """Get or create a test user for local development."""
    test_email = "test@example.com"
    user = session.exec(select(User).where(User.email == test_email)).first()
    if not user:
        user = User(
            email=test_email,
            hashed_password=get_password_hash("test_password"),
            is_active=True,
            is_superuser=False,
            full_name="Test User",
        )
        session.add(user)
        session.commit()
        session.refresh(user)
    return user


def validate_experiment_request(experiment: ExperimentCreate) -> TrainingConfig:
    """Validate and extract training config from request."""
    if experiment.preset is not None:
        return TrainingConfig.preset_config(experiment.preset)
    elif experiment.config is not None:
        return TrainingConfig(**experiment.config.model_dump())
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either `preset` or `config` when creating an experiment.",
        )


def create_experiment_record(
    name: str,
    config: TrainingConfig,
    user: User,
    session: Session
) -> UncertaintyExperiment:
    """Create experiment database record."""
    config_yaml = yaml.dump(config.to_yaml_dict(), default_flow_style=False)
    
    db_experiment = UncertaintyExperiment(
        name=name,
        config_yaml=config_yaml,
        created_by_id=user.id,
    )
    session.add(db_experiment)
    session.commit()
    session.refresh(db_experiment)
    return db_experiment


# ============================================================================
# API Endpoints - No Auth (Development)
# ============================================================================

@router.post("/no-auth", response_model=ExperimentResponse)
async def create_experiment_no_auth(
    experiment: ExperimentCreate,
    session: SessionDep,
) -> Any:
    """Create experiment (no authentication required for local testing)."""
    user = get_or_create_test_user(session)
    config = validate_experiment_request(experiment)
    db_experiment = create_experiment_record(experiment.name, config, user, session)
    return db_experiment


@router.get("/no-auth", response_model=list[ExperimentResponse])
async def list_experiments_no_auth(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List all experiments (no authentication required for local testing)."""
    statement = (
        select(UncertaintyExperiment)
        .offset(skip)
        .limit(limit)
        .order_by(desc(UncertaintyExperiment.created_at))
    )
    experiments = session.exec(statement).all()
    return experiments


@router.post("/no-auth/{experiment_id}/start")
async def start_experiment_no_auth(experiment_id: str, session: SessionDep) -> dict:
    """
    Start ML training for experiment.
    
    Note: Actual training orchestration is handled by 7_orchestration layer.
    This endpoint validates the request and delegates to the orchestrator.
    """
    # Validate experiment_id format
    try:
        experiment_uuid = uuid.UUID(experiment_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid experiment ID format: {experiment_id}. Must be a valid UUID."
        )
    
    # Check if experiment exists
    db_experiment = session.get(UncertaintyExperiment, experiment_uuid)
    if not db_experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment not found: {experiment_id}"
        )
    
    # Import orchestrator (lazy import to avoid circular dependencies)
    from app.services.training_orchestrator import TrainingOrchestrator
    from app.services.executors.direct_executor import DirectExecutor
    from pathlib import Path
    from app.core.config import settings
    
    ML_SCRIPT = Path(settings.DTAG_ROOT) / "run_fast_uncertainty_classification.py"
    
    if not ML_SCRIPT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"ML script not found at {ML_SCRIPT}"
        )
    
    try:
        # Create orchestrator and start training
        executor = DirectExecutor(ML_SCRIPT)
        repository = ExperimentRepository(session)
        orchestrator = TrainingOrchestrator(executor, repository)
        
        await orchestrator.start_training(experiment_uuid)
        
        return {
            "id": experiment_id,
            "status": "running",
            "message": "ML training started successfully"
        }
    except Exception as e:
        logger.error(f"Failed to start training for {experiment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.delete("/no-auth/{experiment_id}")
async def delete_experiment_no_auth(experiment_id: str, session: SessionDep) -> dict:
    """Delete an experiment (no authentication required for development)."""
    try:
        experiment_uuid = uuid.UUID(experiment_id)
        repository = ExperimentRepository(session)
        
        deleted = repository.delete(experiment_uuid)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        logger.info(f"Experiment deleted: {experiment_id}")
        return {"message": "Experiment deleted successfully", "id": experiment_id}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid experiment ID format")
    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete experiment: {str(e)}")


# ============================================================================
# API Endpoints - Authenticated
# ============================================================================

@router.post("", response_model=ExperimentResponse)
async def create_experiment(
    experiment: ExperimentCreate,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Create a new experiment configuration."""
    config = validate_experiment_request(experiment)
    db_experiment = create_experiment_record(experiment.name, config, current_user, session)
    return db_experiment


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List all experiments for the current user."""
    statement = (
        select(UncertaintyExperiment)
        .where(UncertaintyExperiment.created_by_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(desc(UncertaintyExperiment.created_at))
    )
    experiments = session.exec(statement).all()
    return experiments


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Get experiment details."""
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    return experiment


@router.post("/{experiment_id}/start")
async def start_experiment(
    experiment_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Start ML training execution."""
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    if experiment.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Experiment already running")

    # Import orchestrator
    from app.services.training_orchestrator import TrainingOrchestrator
    from app.services.executors.direct_executor import DirectExecutor
    from pathlib import Path
    from app.core.config import settings
    
    ML_SCRIPT = Path(settings.DTAG_ROOT) / "run_fast_uncertainty_classification.py"
    
    if not ML_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"ML script not found at {ML_SCRIPT}")

    try:
        executor = DirectExecutor(ML_SCRIPT)
        repository = ExperimentRepository(session)
        orchestrator = TrainingOrchestrator(executor, repository)
        
        await orchestrator.start_training(experiment_id)
        
        return {
            "id": str(experiment_id),
            "status": "running",
            "message": "ML training started successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.delete("/{experiment_id}")
async def delete_experiment(
    experiment_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, str]:
    """Delete an experiment."""
    experiment = session.get(UncertaintyExperiment, experiment_id)
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    session.delete(experiment)
    session.commit()
    return {"message": "Experiment deleted"}

# Made with Bob
