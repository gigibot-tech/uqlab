"""
Experiment Runner

Orchestrates single experiment execution with resource management,
error handling, and progress tracking.
"""

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

import psutil
import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================

class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResourceRequirements:
    """Resource requirements for experiment."""
    gpu_memory_gb: float = 8.0
    cpu_cores: int = 4
    ram_gb: float = 16.0
    disk_gb: float = 10.0


@dataclass
class ExperimentContext:
    """Context for experiment execution."""
    experiment_id: str
    config: Dict[str, Any]
    output_dir: Path
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0


# ============================================================================
# Resource Management
# ============================================================================

class ResourceManager:
    """Manages system resources for experiments."""
    
    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available."""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_gpu_memory_available() -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        available = (total - allocated) / (1024 ** 3)  # Convert to GB
        
        return available
    
    @staticmethod
    def get_cpu_available() -> int:
        """Get number of available CPU cores."""
        return psutil.cpu_count(logical=False) or 1
    
    @staticmethod
    def get_ram_available() -> float:
        """Get available RAM in GB."""
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    
    @staticmethod
    def get_disk_available(path: Path) -> float:
        """Get available disk space in GB."""
        usage = psutil.disk_usage(str(path))
        return usage.free / (1024 ** 3)
    
    @classmethod
    def check_resources(cls, requirements: ResourceRequirements, output_dir: Path) -> tuple[bool, str]:
        """
        Check if required resources are available.
        
        Returns:
            (available, message) tuple
        """
        issues = []
        
        # Check GPU
        if requirements.gpu_memory_gb > 0:
            if not cls.check_gpu_available():
                issues.append("GPU not available")
            else:
                available_gpu = cls.get_gpu_memory_available()
                if available_gpu < requirements.gpu_memory_gb:
                    issues.append(
                        f"Insufficient GPU memory: {available_gpu:.1f}GB available, "
                        f"{requirements.gpu_memory_gb:.1f}GB required"
                    )
        
        # Check RAM
        available_ram = cls.get_ram_available()
        if available_ram < requirements.ram_gb:
            issues.append(
                f"Insufficient RAM: {available_ram:.1f}GB available, "
                f"{requirements.ram_gb:.1f}GB required"
            )
        
        # Check disk
        available_disk = cls.get_disk_available(output_dir)
        if available_disk < requirements.disk_gb:
            issues.append(
                f"Insufficient disk space: {available_disk:.1f}GB available, "
                f"{requirements.disk_gb:.1f}GB required"
            )
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "All resources available"


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """
    Orchestrates experiment execution.
    
    Features:
    - Resource checking
    - Process management
    - Progress tracking
    - Error handling
    - Cleanup
    """
    
    def __init__(
        self,
        script_path: Path,
        requirements: Optional[ResourceRequirements] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        """
        Initialize experiment runner.
        
        Args:
            script_path: Path to training script
            requirements: Resource requirements
            progress_callback: Optional callback for progress updates
        """
        self.script_path = script_path
        self.requirements = requirements or ResourceRequirements()
        self.progress_callback = progress_callback
        self.resource_manager = ResourceManager()
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
    
    def run(self, context: ExperimentContext) -> ExperimentContext:
        """
        Run experiment.
        
        Args:
            context: Experiment context
        
        Returns:
            Updated context with results
        """
        logger.info(f"Starting experiment {context.experiment_id}")
        
        # Check resources
        available, message = self.resource_manager.check_resources(
            self.requirements,
            context.output_dir
        )
        
        if not available:
            context.status = ExperimentStatus.FAILED
            context.error_message = f"Resource check failed: {message}"
            logger.error(context.error_message)
            return context
        
        # Create output directory
        context.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update status
        context.status = ExperimentStatus.RUNNING
        context.start_time = datetime.utcnow()
        
        try:
            # Build command
            cmd = self._build_command(context)
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.script_path.parent),
            )
            
            # Monitor progress
            self._monitor_process(process, context)
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                context.status = ExperimentStatus.COMPLETED
                context.progress = 1.0
                logger.info(f"Experiment {context.experiment_id} completed successfully")
            else:
                context.status = ExperimentStatus.FAILED
                context.error_message = f"Process failed with code {process.returncode}: {stderr[:500]}"
                logger.error(context.error_message)
        
        except Exception as e:
            context.status = ExperimentStatus.FAILED
            context.error_message = f"Execution error: {str(e)}"
            logger.error(f"Experiment {context.experiment_id} failed: {e}", exc_info=True)
        
        finally:
            context.end_time = datetime.utcnow()
            self._cleanup(context)
        
        return context
    
    def _build_command(self, context: ExperimentContext) -> list[str]:
        """Build command to execute."""
        cmd = [
            "python",
            str(self.script_path),
            "--experiment-id", context.experiment_id,
            "--output-dir", str(context.output_dir),
        ]
        
        # Add config parameters
        for key, value in context.config.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
    def _monitor_process(self, process: subprocess.Popen, context: ExperimentContext):
        """Monitor process and update progress."""
        # Simple progress monitoring
        # In production, this would parse log files or use IPC
        
        start_time = time.time()
        max_duration = 3600  # 1 hour timeout
        
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            # Estimate progress (simple linear)
            estimated_progress = min(0.9, elapsed / max_duration)
            context.progress = estimated_progress
            
            if self.progress_callback:
                self.progress_callback(context.progress)
            
            # Check timeout
            if elapsed > max_duration:
                logger.warning(f"Experiment {context.experiment_id} timeout, terminating")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                break
            
            time.sleep(5)  # Check every 5 seconds
    
    def _cleanup(self, context: ExperimentContext):
        """Cleanup after experiment."""
        # Clear GPU cache if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Cleanup completed for experiment {context.experiment_id}")
    
    def cancel(self, context: ExperimentContext):
        """Cancel running experiment."""
        context.status = ExperimentStatus.CANCELLED
        logger.info(f"Experiment {context.experiment_id} cancelled")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_experiment_context(
    experiment_id: str,
    config: Dict[str, Any],
    output_dir: Path,
) -> ExperimentContext:
    """Create experiment context."""
    return ExperimentContext(
        experiment_id=experiment_id,
        config=config,
        output_dir=output_dir,
    )


def run_experiment(
    experiment_id: str,
    config: Dict[str, Any],
    script_path: Path,
    output_dir: Path,
    requirements: Optional[ResourceRequirements] = None,
) -> ExperimentContext:
    """
    Run experiment with default settings.
    
    Args:
        experiment_id: Unique experiment ID
        config: Experiment configuration
        script_path: Path to training script
        output_dir: Output directory
        requirements: Optional resource requirements
    
    Returns:
        Experiment context with results
    """
    context = create_experiment_context(experiment_id, config, output_dir)
    runner = ExperimentRunner(script_path, requirements)
    return runner.run(context)


# Made with Bob