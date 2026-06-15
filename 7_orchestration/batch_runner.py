"""
Batch Runner

Orchestrates batch experiment execution with queue management,
parallel processing, and progress tracking.
"""

import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import threading

from .experiment_runner import (
    ExperimentRunner,
    ExperimentContext,
    ExperimentStatus,
    ResourceRequirements,
    create_experiment_context,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class BatchStatus(Enum):
    """Batch execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchContext:
    """Context for batch execution."""
    batch_id: str
    experiments: List[ExperimentContext] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    completed_count: int = 0
    failed_count: int = 0
    progress: float = 0.0


# ============================================================================
# Batch Runner
# ============================================================================

class BatchRunner:
    """
    Orchestrates batch experiment execution.
    
    Features:
    - Queue management
    - Parallel execution
    - Progress tracking
    - Error handling
    - Resource-aware scheduling
    """
    
    def __init__(
        self,
        script_path: Path,
        max_parallel: int = 4,
        requirements: Optional[ResourceRequirements] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialize batch runner.
        
        Args:
            script_path: Path to training script
            max_parallel: Maximum parallel experiments
            requirements: Resource requirements per experiment
            progress_callback: Optional callback for progress updates
        """
        self.script_path = script_path
        self.max_parallel = max_parallel
        self.requirements = requirements or ResourceRequirements()
        self.progress_callback = progress_callback
        
        self.experiment_runner = ExperimentRunner(
            script_path,
            requirements,
        )
        
        self.queue: Queue[ExperimentContext] = Queue()
        self.running: Dict[str, ExperimentContext] = {}
        self.completed: List[ExperimentContext] = []
        self.failed: List[ExperimentContext] = []
        
        self._stop_event = threading.Event()
    
    def run(self, batch_context: BatchContext) -> BatchContext:
        """
        Run batch of experiments.
        
        Args:
            batch_context: Batch context with experiments
        
        Returns:
            Updated batch context with results
        """
        logger.info(f"Starting batch {batch_context.batch_id} with {len(batch_context.experiments)} experiments")
        
        # Initialize queue
        for exp_context in batch_context.experiments:
            self.queue.put(exp_context)
        
        # Update status
        batch_context.status = BatchStatus.RUNNING
        batch_context.start_time = datetime.utcnow()
        
        try:
            # Run experiments
            self._run_batch(batch_context)
            
            # Update final status
            if batch_context.failed_count == 0:
                batch_context.status = BatchStatus.COMPLETED
            else:
                batch_context.status = BatchStatus.FAILED
            
            logger.info(
                f"Batch {batch_context.batch_id} completed: "
                f"{batch_context.completed_count} succeeded, "
                f"{batch_context.failed_count} failed"
            )
        
        except Exception as e:
            batch_context.status = BatchStatus.FAILED
            logger.error(f"Batch {batch_context.batch_id} failed: {e}", exc_info=True)
        
        finally:
            batch_context.end_time = datetime.utcnow()
        
        return batch_context
    
    def _run_batch(self, batch_context: BatchContext):
        """Run batch with parallel execution."""
        total_experiments = len(batch_context.experiments)
        
        while not self.queue.empty() or self.running:
            # Start new experiments if slots available
            while len(self.running) < self.max_parallel and not self.queue.empty():
                try:
                    exp_context = self.queue.get_nowait()
                    self._start_experiment(exp_context, batch_context)
                except Empty:
                    break
            
            # Check running experiments
            self._check_running_experiments(batch_context)
            
            # Update progress
            batch_context.progress = (
                batch_context.completed_count + batch_context.failed_count
            ) / total_experiments
            
            if self.progress_callback:
                self.progress_callback(batch_context.batch_id, batch_context.progress)
            
            # Check stop event
            if self._stop_event.is_set():
                logger.info(f"Batch {batch_context.batch_id} cancelled")
                batch_context.status = BatchStatus.CANCELLED
                break
            
            # Sleep briefly
            import time
            time.sleep(1)
    
    def _start_experiment(self, exp_context: ExperimentContext, batch_context: BatchContext):
        """Start experiment in background thread."""
        def run_experiment():
            try:
                result = self.experiment_runner.run(exp_context)
                
                # Update batch context
                if result.status == ExperimentStatus.COMPLETED:
                    batch_context.completed_count += 1
                    self.completed.append(result)
                else:
                    batch_context.failed_count += 1
                    self.failed.append(result)
                
                # Remove from running
                if result.experiment_id in self.running:
                    del self.running[result.experiment_id]
            
            except Exception as e:
                logger.error(f"Experiment {exp_context.experiment_id} failed: {e}")
                batch_context.failed_count += 1
                self.failed.append(exp_context)
                
                if exp_context.experiment_id in self.running:
                    del self.running[exp_context.experiment_id]
        
        # Add to running
        self.running[exp_context.experiment_id] = exp_context
        
        # Start thread
        thread = threading.Thread(target=run_experiment, daemon=True)
        thread.start()
        
        logger.info(f"Started experiment {exp_context.experiment_id}")
    
    def _check_running_experiments(self, batch_context: BatchContext):
        """Check status of running experiments."""
        # In a real implementation, this would check process status
        # For now, we rely on the thread completion
        pass
    
    def cancel(self):
        """Cancel batch execution."""
        self._stop_event.set()
        logger.info("Batch cancellation requested")


# ============================================================================
# Async Batch Runner
# ============================================================================

class AsyncBatchRunner:
    """
    Async version of batch runner for better concurrency.
    """
    
    def __init__(
        self,
        script_path: Path,
        max_parallel: int = 4,
        requirements: Optional[ResourceRequirements] = None,
    ):
        self.script_path = script_path
        self.max_parallel = max_parallel
        self.requirements = requirements or ResourceRequirements()
        
        self.experiment_runner = ExperimentRunner(script_path, requirements)
    
    async def run(self, batch_context: BatchContext) -> BatchContext:
        """Run batch asynchronously."""
        logger.info(f"Starting async batch {batch_context.batch_id}")
        
        batch_context.status = BatchStatus.RUNNING
        batch_context.start_time = datetime.utcnow()
        
        try:
            # Create tasks
            tasks = [
                self._run_experiment_async(exp_context)
                for exp_context in batch_context.experiments
            ]
            
            # Run with concurrency limit
            semaphore = asyncio.Semaphore(self.max_parallel)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in tasks],
                return_exceptions=True
            )
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    batch_context.failed_count += 1
                elif result.status == ExperimentStatus.COMPLETED:
                    batch_context.completed_count += 1
                else:
                    batch_context.failed_count += 1
            
            # Update status
            if batch_context.failed_count == 0:
                batch_context.status = BatchStatus.COMPLETED
            else:
                batch_context.status = BatchStatus.FAILED
            
            batch_context.progress = 1.0
        
        except Exception as e:
            batch_context.status = BatchStatus.FAILED
            logger.error(f"Async batch {batch_context.batch_id} failed: {e}")
        
        finally:
            batch_context.end_time = datetime.utcnow()
        
        return batch_context
    
    async def _run_experiment_async(self, exp_context: ExperimentContext) -> ExperimentContext:
        """Run single experiment asynchronously."""
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.experiment_runner.run,
            exp_context
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def create_batch_context(
    batch_id: str,
    experiment_configs: List[Dict[str, Any]],
    output_base_dir: Path,
) -> BatchContext:
    """
    Create batch context from experiment configurations.
    
    Args:
        batch_id: Unique batch ID
        experiment_configs: List of experiment configurations
        output_base_dir: Base output directory
    
    Returns:
        Batch context
    """
    experiments = []
    
    for idx, config in enumerate(experiment_configs):
        exp_id = f"{batch_id}_exp_{idx}"
        exp_output_dir = output_base_dir / exp_id
        
        exp_context = create_experiment_context(
            experiment_id=exp_id,
            config=config,
            output_dir=exp_output_dir,
        )
        
        experiments.append(exp_context)
    
    return BatchContext(
        batch_id=batch_id,
        experiments=experiments,
    )


def run_batch(
    batch_id: str,
    experiment_configs: List[Dict[str, Any]],
    script_path: Path,
    output_base_dir: Path,
    max_parallel: int = 4,
    requirements: Optional[ResourceRequirements] = None,
) -> BatchContext:
    """
    Run batch of experiments with default settings.
    
    Args:
        batch_id: Unique batch ID
        experiment_configs: List of experiment configurations
        script_path: Path to training script
        output_base_dir: Base output directory
        max_parallel: Maximum parallel experiments
        requirements: Optional resource requirements
    
    Returns:
        Batch context with results
    """
    batch_context = create_batch_context(batch_id, experiment_configs, output_base_dir)
    runner = BatchRunner(script_path, max_parallel, requirements)
    return runner.run(batch_context)


# Made with Bob