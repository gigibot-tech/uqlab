# UQLab ML Core Constitution

## Core Principles

### I. Modularity-First Architecture
Every component must be independently usable, testable, and maintainable. Modules are organized by ML pipeline stage (data, models, training, evaluation) with clear boundaries and minimal coupling. Each module exposes a well-defined public API through `__init__.py` exports.

**Rules:**
- Single Responsibility: Each module does one thing well
- Clear Interfaces: Public APIs documented with type hints and docstrings
- Dependency Injection: Avoid hardcoded dependencies, use factory patterns
- Layered Architecture: Data → Models → Training → Evaluation (no circular dependencies)

### II. Code Reuse Mandate (NON-NEGOTIABLE)
Before creating ANY new function, class, or module, you MUST:
1. Search existing codebase for similar functionality
2. Check if existing utilities can be extended
3. Verify no duplicate logic exists
4. Document why reuse wasn't possible (if creating new code)

**Reuse Checklist:**
- [ ] Searched for similar function names
- [ ] Checked shared utilities (`shared/`, `utils/`)
- [ ] Reviewed related modules for extensibility
- [ ] Documented reuse decision in commit message

### III. DRY Principle (Don't Repeat Yourself)
Duplicate code is a critical defect. Extract common patterns into:
- **Shared utilities** (`shared/utils.py`) for cross-cutting concerns
- **Base classes** for common behavior patterns
- **Factory functions** for object creation logic
- **Configuration objects** for shared parameters

**Violation Examples:**
- ❌ Copy-pasting functions between modules
- ❌ Duplicate validation logic
- ❌ Repeated data transformation code
- ❌ Multiple implementations of same algorithm

### IV. Type Safety & Documentation
All public APIs must have:
- Type hints for all parameters and return values
- Docstrings following NumPy/Google style
- Usage examples in docstrings
- Validation of input types at runtime (where critical)

**Example:**
```python
def train_model(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Train a PyTorch model.
    
    Args:
        model: PyTorch model to train
        data_loader: Training data loader
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    
    Returns:
        Dictionary containing training metrics
    
    Example:
        >>> model = ResNet18()
        >>> loader = DataLoader(dataset)
        >>> metrics = train_model(model, loader, epochs=5)
    """
```

### V. Testing Standards
- **Unit Tests**: 80% minimum coverage on business logic
- **Integration Tests**: For cross-module interactions
- **Fixtures**: Reusable test data in `tests/fixtures/`
- **Mocking**: External dependencies (APIs, file I/O)

**Test Organization:**
```
tests/
├── unit/           # Module-level tests
├── integration/    # Cross-module tests
├── fixtures/       # Shared test data
└── conftest.py     # Pytest configuration
```

### VI. Dependency Management
- **Minimize Dependencies**: Only add if absolutely necessary
- **Pin Versions**: Use exact versions in `requirements.txt`
- **Optional Dependencies**: Use extras for non-core features
- **Compatibility**: Support Python 3.11+

**Dependency Tiers:**
1. **Core**: PyTorch, NumPy, Pandas (required)
2. **Optional**: Visualization, cloud integrations (extras)
3. **Development**: Testing, linting, docs (dev requirements)

### VII. Performance & Scalability
- **Profile Before Optimizing**: Use `cProfile`, `line_profiler`
- **Batch Processing**: Support batch operations where applicable
- **Memory Efficiency**: Use generators for large datasets
- **GPU Support**: Provide CPU fallback for all operations

## Code Organization Standards

### Module Structure
```
uqlab/
├── 1_data/              # Data loading & preprocessing
│   ├── loaders.py       # Dataset loaders
│   ├── transforms.py    # Data transformations
│   └── stats.py         # Dataset statistics
├── 2_models/            # Model architectures
│   ├── factory.py       # Model creation
│   ├── architectures.py # Model definitions
│   └── uncertainty.py   # Uncertainty quantification
├── 3_training/          # Training logic
│   ├── trainer.py       # Training loops
│   ├── callbacks.py     # Training callbacks
│   └── config.py        # Training configuration
├── 4_evaluation/        # Evaluation & metrics
│   ├── metrics.py       # Metric calculations
│   ├── signals.py       # Uncertainty signals
│   └── validators.py    # Result validation
├── shared/              # Cross-cutting utilities
│   ├── utils.py         # Common utilities
│   ├── types.py         # Shared type definitions
│   └── constants.py     # Project constants
└── ui_components/       # Streamlit UI components
    ├── config/          # Configuration UIs
    ├── selectors/       # Selection widgets
    ├── visualization/   # Plotting & charts
    └── orchestration/   # Workflow coordination
```

### Import Conventions
```python
# Standard library
import os
from pathlib import Path

# Third-party
import torch
import numpy as np

# Local - absolute imports from package root
from uqlab.data.loaders import CIFAR10NDataset
from uqlab.models.factory import build_model
from uqlab.shared.utils import setup_logging
```

## Quality Gates

### Pre-Commit Checks
1. **Linting**: `ruff check .` (no errors)
2. **Type Checking**: `mypy uqlab/` (no errors)
3. **Tests**: `pytest tests/` (all pass)
4. **Coverage**: `pytest --cov=uqlab` (≥80%)

### Code Review Requirements
- [ ] Modularity: Single responsibility maintained
- [ ] Reuse: Existing code checked, reused where possible
- [ ] DRY: No duplicate logic
- [ ] Types: All public APIs have type hints
- [ ] Docs: All public APIs have docstrings
- [ ] Tests: New code has tests
- [ ] Performance: No obvious inefficiencies

## Brownfield Development Guidelines

When enhancing existing code:
1. **Understand First**: Read existing implementation thoroughly
2. **Preserve Interfaces**: Maintain backward compatibility
3. **Refactor Safely**: Add tests before refactoring
4. **Document Changes**: Update docstrings and README
5. **Migration Path**: Provide deprecation warnings for breaking changes

## Error Handling

### Exception Hierarchy
```python
class UQLabError(Exception):
    """Base exception for UQLab"""

class DataError(UQLabError):
    """Data loading/processing errors"""

class ModelError(UQLabError):
    """Model creation/training errors"""

class EvaluationError(UQLabError):
    """Evaluation/metrics errors"""
```

### Error Messages
- **Actionable**: Tell user what to do
- **Contextual**: Include relevant values
- **Traceable**: Log full context for debugging

## Governance

### Constitution Authority
This constitution supersedes all other development practices. Any deviation requires:
1. Documented justification
2. Team review and approval
3. Constitution amendment (if pattern emerges)

### Amendment Process
1. Propose change with rationale
2. Review impact on existing code
3. Update constitution
4. Communicate to team
5. Update version and date

### Compliance
- All PRs must verify compliance with constitution
- Complexity must be justified
- Violations are blocking issues

**Version**: 1.0.0 | **Ratified**: 2026-06-15 | **Last Amended**: 2026-06-15
