"""
API Layer - MLOps Stage 5

This module contains consolidated API endpoints organized by resource type.
Follows the refactored MLOps structure for better maintainability.

Modules:
- experiments: Single experiment management endpoints
- batch: Batch experiment management endpoints  
- datasets: Dataset statistics and exploration endpoints
- models: Model serving and inference endpoints
"""

from . import experiments, batch, datasets, models

__all__ = ["experiments", "batch", "datasets", "models"]

# Made with Bob
