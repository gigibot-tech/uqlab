"""
Shared Utilities - Common helper functions and utilities.

This module provides utility functions used across all layers:
- Logging setup and helpers
- File I/O operations
- Reproducibility utilities
- Common transformations
- Error handling decorators
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import pickle
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import numpy as np
import torch
import yaml

from ..types import PathLike

# Type variable for generic functions
T = TypeVar('T')


# ============================================================================
# Logging Utilities
# ============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[PathLike] = None,
    log_to_console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        log_to_console: Whether to log to console
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger("walaris")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "walaris") -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)


# ============================================================================
# Reproducibility Utilities
# ============================================================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: Optional[str] = None, gpu_id: int = 0) -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', or None for auto)
        gpu_id: GPU ID to use
    
    Returns:
        torch.device instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_json(data: Dict, path: PathLike, indent: int = 2) -> None:
    """Save dictionary to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: PathLike) -> Dict:
    """Load dictionary from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict, path: PathLike) -> None:
    """Save dictionary to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path: PathLike) -> Dict:
    """Load dictionary from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, path: PathLike) -> None:
    """Save object to pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: PathLike) -> Any:
    """Load object from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(path: PathLike) -> Path:
    """Ensure directory exists, create if not."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(path: PathLike, algorithm: str = "md5") -> str:
    """
    Compute hash of file contents.
    
    Args:
        path: File path
        algorithm: Hash algorithm (md5, sha256, etc.)
    
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def get_string_hash(text: str, algorithm: str = "md5") -> str:
    """Compute hash of string."""
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(text.encode('utf-8'))
    return hash_obj.hexdigest()


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or get_logger()
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.logger.info(f"{self.name} took {elapsed:.2f} seconds")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


# ============================================================================
# Error Handling Decorators
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            logger = get_logger()
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise
                    
                    msg = (
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    logger.warning(msg)
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            raise RuntimeError(f"{func.__name__} failed after {max_attempts} attempts")
        return wrapper
    return decorator


def safe_execute(default_return: Any = None, log_errors: bool = True) -> Callable:
    """
    Decorator to safely execute function and return default on error.
    
    Args:
        default_return: Value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Union[T, Any]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = get_logger()
                    logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator


# ============================================================================
# Data Utilities
# ============================================================================

def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_tensor(array: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array to tensor."""
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move batch dictionary to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ============================================================================
# String Utilities
# ============================================================================

def format_number(num: float, precision: int = 2) -> str:
    """Format number with appropriate precision."""
    if abs(num) < 0.01:
        return f"{num:.2e}"
    return f"{num:.{precision}f}"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_range(value: float, min_val: float, max_val: float, name: str = "value") -> None:
    """Validate that value is within range."""
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """Validate that value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


# Made with Bob