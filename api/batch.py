"""
Batch Experiment Management API Endpoints

Consolidated batch experiment endpoints for managing multiple experiments.
Handles batch creation, execution, monitoring, and results aggregation.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlmodel import Session, select, desc, and_

from app.api.deps import CurrentUser, SessionDep
from app.domain.models import TrainingConfig, TrainingPresetName
from app.repositories.experiment_repository import ExperimentRepository
from app.tables import JobStatus, UncertaintyExperiment, BatchExperiment, User

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================

class BatchExperimentConfig(BaseModel):
    """Configuration for a single experiment in a batch."""
    name: str = Field(min_length=1, max_length=255)
    preset: TrainingPresetName | None = None
    config: TrainingConfig | None = None


class BatchCreate(BaseModel):
    """Request to create a batch of experiments."""
    name: str = Field(min_length=1, max_length=255, description="Batch name")
    description: str | None = Field(None, max_length=1000, description="Batch description")
    experiments: List[BatchExperimentConfig] = Field(
        min_length=1,
        max_length=100,
        description="List of experiments to run"
    )
    run_parallel: bool = Field(
        default=False,
        description="Whether to run experiments in parallel"
    )
    max_parallel: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of parallel experiments"
    )


class BatchExperimentSummary(BaseModel):
    """Summary of a single experiment in a batch."""
    id: uuid.UUID
    name: str
    status: JobStatus
    progress: float
    aleatoric_auroc: float | None
    epistemic_auroc: float | None
    error_message: str | None


class BatchResponse(BaseModel):
    """Batch experiment response."""
    id: uuid.UUID
    name: str
    description: str | None
    status: JobStatus
    total_experiments: int
    completed_experiments: int
    failed_experiments: int
    progress: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    experiments: List[BatchExperimentSummary] = []


class BatchListResponse(BaseModel):
    """Simplified batch list response."""
    id: uuid.UUID
    name: str
    description: str | None
    status: JobStatus
    total_experiments: int
    completed_experiments: int
    progress: float
    created_at: datetime


class BatchUpdateRequest(BaseModel):
    """Request to update batch metadata."""
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = Field(None, max_length=1000)


# ============================================================================
# Helper Functions
# ============================================================================

def get_or_create_test_user(session: Session) -> User:
    """Get or create a test user for local development."""
    from app.core.security import get_password_hash
    
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


def validate_batch_experiment(exp: BatchExperimentConfig) -> TrainingConfig:
    """Validate and extract training config from batch experiment."""
    if exp.preset is not None:
        return TrainingConfig.preset_config(exp.preset)
    elif exp.config is not None:
        return TrainingConfig(**exp.config.model_dump())
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{exp.name}' must have either preset or config"
        )


def create_batch_record(
    name: str,
    description: str | None,
    user: User,
    session: Session
) -> BatchExperiment:
    """Create batch experiment database record."""
    db_batch = BatchExperiment(
        name=name,
        description=description,
        created_by_id=user.id,
        status=JobStatus.PENDING,
        total_experiments=0,
        completed_experiments=0,
        failed_experiments=0,
    )
    session.add(db_batch)
    session.commit()
    session.refresh(db_batch)
    return db_batch


def create_experiment_in_batch(
    name: str,
    config: TrainingConfig,
    batch_id: uuid.UUID,
    user: User,
    session: Session
) -> UncertaintyExperiment:
    """Create an experiment as part of a batch."""
    config_yaml = yaml.dump(config.to_yaml_dict(), default_flow_style=False)
    
    db_experiment = UncertaintyExperiment(
        name=name,
        config_yaml=config_yaml,
        created_by_id=user.id,
        batch_id=batch_id,
    )
    session.add(db_experiment)
    return db_experiment


def calculate_batch_progress(experiments: List[UncertaintyExperiment]) -> tuple[int, int, float]:
    """Calculate batch progress statistics."""
    if not experiments:
        return 0, 0, 0.0
    
    completed = sum(1 for exp in experiments if exp.status == JobStatus.COMPLETED)
    failed = sum(1 for exp in experiments if exp.status == JobStatus.FAILED)
    
    # Calculate weighted progress
    total_progress = sum(exp.progress for exp in experiments)
    avg_progress = total_progress / len(experiments) if experiments else 0.0
    
    return completed, failed, avg_progress


def get_batch_experiments(batch_id: uuid.UUID, session: Session) -> List[UncertaintyExperiment]:
    """Get all experiments in a batch."""
    statement = (
        select(UncertaintyExperiment)
        .where(UncertaintyExperiment.batch_id == batch_id)
        .order_by(UncertaintyExperiment.created_at)
    )
    return list(session.exec(statement).all())


async def start_batch_execution(
    batch_id: uuid.UUID,
    run_parallel: bool,
    max_parallel: int,
    session: Session
):
    """Start execution of all experiments in a batch."""
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
    
    # Get all experiments in batch
    experiments = get_batch_experiments(batch_id, session)
    
    if not experiments:
        raise HTTPException(
            status_code=400,
            detail="Batch has no experiments"
        )
    
    # Update batch status
    batch = session.get(BatchExperiment, batch_id)
    if batch:
        batch.status = JobStatus.RUNNING
        batch.started_at = datetime.utcnow()
        session.commit()
    
    # Create orchestrator
    executor = DirectExecutor(ML_SCRIPT)
    repository = ExperimentRepository(session)
    orchestrator = TrainingOrchestrator(executor, repository)
    
    if run_parallel:
        # Start experiments in parallel (up to max_parallel)
        import asyncio
        
        async def run_experiment(exp_id: uuid.UUID):
            try:
                await orchestrator.start_training(exp_id)
            except Exception as e:
                logger.error(f"Failed to start experiment {exp_id}: {e}")
        
        # Process in batches of max_parallel
        for i in range(0, len(experiments), max_parallel):
            batch_slice = experiments[i:i + max_parallel]
            tasks = [run_experiment(exp.id) for exp in batch_slice]
            await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Run experiments sequentially
        for exp in experiments:
            try:
                await orchestrator.start_training(exp.id)
            except Exception as e:
                logger.error(f"Failed to start experiment {exp.id}: {e}")


# ============================================================================
# API Endpoints - No Auth (Development)
# ============================================================================

@router.post("/no-auth", response_model=BatchResponse)
async def create_batch_no_auth(
    batch: BatchCreate,
    session: SessionDep,
) -> Any:
    """Create batch of experiments (no authentication for local testing)."""
    user = get_or_create_test_user(session)
    
    # Validate all experiment configs
    configs = []
    for exp in batch.experiments:
        config = validate_batch_experiment(exp)
        configs.append((exp.name, config))
    
    # Create batch record
    db_batch = create_batch_record(batch.name, batch.description, user, session)
    
    # Create all experiments
    for name, config in configs:
        create_experiment_in_batch(name, config, db_batch.id, user, session)
    
    # Update batch totals
    db_batch.total_experiments = len(configs)
    session.commit()
    session.refresh(db_batch)
    
    # Get experiments for response
    experiments = get_batch_experiments(db_batch.id, session)
    
    return BatchResponse(
        id=db_batch.id,
        name=db_batch.name,
        description=db_batch.description,
        status=db_batch.status,
        total_experiments=db_batch.total_experiments,
        completed_experiments=0,
        failed_experiments=0,
        progress=0.0,
        created_at=db_batch.created_at,
        started_at=None,
        completed_at=None,
        experiments=[
            BatchExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                progress=exp.progress,
                aleatoric_auroc=exp.aleatoric_auroc,
                epistemic_auroc=exp.epistemic_auroc,
                error_message=exp.error_message,
            )
            for exp in experiments
        ]
    )


@router.get("/no-auth", response_model=List[BatchListResponse])
async def list_batches_no_auth(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List all batches (no authentication for local testing)."""
    statement = (
        select(BatchExperiment)
        .offset(skip)
        .limit(limit)
        .order_by(desc(BatchExperiment.created_at))
    )
    batches = session.exec(statement).all()
    
    return [
        BatchListResponse(
            id=batch.id,
            name=batch.name,
            description=batch.description,
            status=batch.status,
            total_experiments=batch.total_experiments,
            completed_experiments=batch.completed_experiments,
            progress=batch.completed_experiments / batch.total_experiments * 100
                if batch.total_experiments > 0 else 0.0,
            created_at=batch.created_at,
        )
        for batch in batches
    ]


@router.get("/no-auth/{batch_id}", response_model=BatchResponse)
async def get_batch_no_auth(
    batch_id: str,
    session: SessionDep,
) -> Any:
    """Get batch details (no authentication for local testing)."""
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID format")
    
    batch = session.get(BatchExperiment, batch_uuid)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Get all experiments
    experiments = get_batch_experiments(batch_uuid, session)
    
    # Calculate current progress
    completed, failed, progress = calculate_batch_progress(experiments)
    
    return BatchResponse(
        id=batch.id,
        name=batch.name,
        description=batch.description,
        status=batch.status,
        total_experiments=batch.total_experiments,
        completed_experiments=completed,
        failed_experiments=failed,
        progress=progress,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
        experiments=[
            BatchExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                progress=exp.progress,
                aleatoric_auroc=exp.aleatoric_auroc,
                epistemic_auroc=exp.epistemic_auroc,
                error_message=exp.error_message,
            )
            for exp in experiments
        ]
    )


@router.post("/no-auth/{batch_id}/start")
async def start_batch_no_auth(
    batch_id: str,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    run_parallel: bool = False,
    max_parallel: int = 4,
) -> dict:
    """Start batch execution (no authentication for local testing)."""
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID format")
    
    batch = session.get(BatchExperiment, batch_uuid)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    if batch.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Batch already running")
    
    # Start batch execution in background
    background_tasks.add_task(
        start_batch_execution,
        batch_uuid,
        run_parallel,
        max_parallel,
        session
    )
    
    return {
        "id": batch_id,
        "status": "running",
        "message": f"Batch execution started ({'parallel' if run_parallel else 'sequential'})"
    }


@router.delete("/no-auth/{batch_id}")
async def delete_batch_no_auth(
    batch_id: str,
    session: SessionDep,
) -> dict:
    """Delete a batch and all its experiments (no authentication for local testing)."""
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch ID format")
    
    batch = session.get(BatchExperiment, batch_uuid)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Delete all experiments in batch
    experiments = get_batch_experiments(batch_uuid, session)
    for exp in experiments:
        session.delete(exp)
    
    # Delete batch
    session.delete(batch)
    session.commit()
    
    return {
        "message": "Batch and all experiments deleted successfully",
        "id": batch_id,
        "deleted_experiments": len(experiments)
    }


# ============================================================================
# API Endpoints - Authenticated
# ============================================================================

@router.post("", response_model=BatchResponse)
async def create_batch(
    batch: BatchCreate,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Create a batch of experiments."""
    # Validate all experiment configs
    configs = []
    for exp in batch.experiments:
        config = validate_batch_experiment(exp)
        configs.append((exp.name, config))
    
    # Create batch record
    db_batch = create_batch_record(batch.name, batch.description, current_user, session)
    
    # Create all experiments
    for name, config in configs:
        create_experiment_in_batch(name, config, db_batch.id, current_user, session)
    
    # Update batch totals
    db_batch.total_experiments = len(configs)
    session.commit()
    session.refresh(db_batch)
    
    # Get experiments for response
    experiments = get_batch_experiments(db_batch.id, session)
    
    return BatchResponse(
        id=db_batch.id,
        name=db_batch.name,
        description=db_batch.description,
        status=db_batch.status,
        total_experiments=db_batch.total_experiments,
        completed_experiments=0,
        failed_experiments=0,
        progress=0.0,
        created_at=db_batch.created_at,
        started_at=None,
        completed_at=None,
        experiments=[
            BatchExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                progress=exp.progress,
                aleatoric_auroc=exp.aleatoric_auroc,
                epistemic_auroc=exp.epistemic_auroc,
                error_message=exp.error_message,
            )
            for exp in experiments
        ]
    )


@router.get("", response_model=List[BatchListResponse])
async def list_batches(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """List all batches for the current user."""
    statement = (
        select(BatchExperiment)
        .where(BatchExperiment.created_by_id == current_user.id)
        .offset(skip)
        .limit(limit)
        .order_by(desc(BatchExperiment.created_at))
    )
    batches = session.exec(statement).all()
    
    return [
        BatchListResponse(
            id=batch.id,
            name=batch.name,
            description=batch.description,
            status=batch.status,
            total_experiments=batch.total_experiments,
            completed_experiments=batch.completed_experiments,
            progress=batch.completed_experiments / batch.total_experiments * 100
                if batch.total_experiments > 0 else 0.0,
            created_at=batch.created_at,
        )
        for batch in batches
    ]


@router.get("/{batch_id}", response_model=BatchResponse)
async def get_batch(
    batch_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Get batch details."""
    batch = session.get(BatchExperiment, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get all experiments
    experiments = get_batch_experiments(batch_id, session)
    
    # Calculate current progress
    completed, failed, progress = calculate_batch_progress(experiments)
    
    return BatchResponse(
        id=batch.id,
        name=batch.name,
        description=batch.description,
        status=batch.status,
        total_experiments=batch.total_experiments,
        completed_experiments=completed,
        failed_experiments=failed,
        progress=progress,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
        experiments=[
            BatchExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                progress=exp.progress,
                aleatoric_auroc=exp.aleatoric_auroc,
                epistemic_auroc=exp.epistemic_auroc,
                error_message=exp.error_message,
            )
            for exp in experiments
        ]
    )


@router.put("/{batch_id}", response_model=BatchResponse)
async def update_batch(
    batch_id: uuid.UUID,
    update: BatchUpdateRequest,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Update batch metadata."""
    batch = session.get(BatchExperiment, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    if update.name is not None:
        batch.name = update.name
    if update.description is not None:
        batch.description = update.description
    
    session.commit()
    session.refresh(batch)
    
    # Get experiments for response
    experiments = get_batch_experiments(batch_id, session)
    completed, failed, progress = calculate_batch_progress(experiments)
    
    return BatchResponse(
        id=batch.id,
        name=batch.name,
        description=batch.description,
        status=batch.status,
        total_experiments=batch.total_experiments,
        completed_experiments=completed,
        failed_experiments=failed,
        progress=progress,
        created_at=batch.created_at,
        started_at=batch.started_at,
        completed_at=batch.completed_at,
        experiments=[
            BatchExperimentSummary(
                id=exp.id,
                name=exp.name,
                status=exp.status,
                progress=exp.progress,
                aleatoric_auroc=exp.aleatoric_auroc,
                epistemic_auroc=exp.epistemic_auroc,
                error_message=exp.error_message,
            )
            for exp in experiments
        ]
    )


@router.delete("/{batch_id}")
async def delete_batch(
    batch_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Delete a batch and all its experiments."""
    batch = session.get(BatchExperiment, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    if batch.created_by_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Delete all experiments in batch
    experiments = get_batch_experiments(batch_id, session)
    for exp in experiments:
        session.delete(exp)
    
    # Delete batch
    session.delete(batch)
    session.commit()
    
    return {
        "message": "Batch and all experiments deleted successfully",
        "deleted_experiments": len(experiments)
    }


# Made with Bob