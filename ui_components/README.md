# UQLab UI Components

**Modular Streamlit components for uncertainty quantification experiments**

## Overview

The `ui_components` package provides reusable, composable Streamlit components for building uncertainty quantification (UQ) experiment interfaces. Following Spec-Driven Development principles, each component has a clear specification, single responsibility, and well-defined interfaces.

## Architecture

```
ui_components/
├── config/              # Experiment configuration builders
│   ├── experiment_config.py    # Base experiment configuration
│   ├── batch_config.py         # Batch experiment configuration
│   └── dataset_config.py       # Dataset selection & configuration
├── selectors/           # UI selection components
│   ├── model_selector.py       # Model architecture selection
│   ├── training_selector.py    # Training parameter selection
│   └── evaluation_selector.py  # Evaluation configuration
├── results/             # Results display components
│   └── results.py              # Experiment results visualization
├── orchestration/       # Workflow orchestration
│   ├── unified_builder.py      # Unified experiment builder
│   └── validation_runner.py    # Validation experiment runner
├── visualization/       # Visualization components
│   ├── signals/                # Signal visualization
│   │   ├── signal_visualization.py
│   │   ├── signal_diagnostic_viz.py
│   │   └── per_sample_signals_viz.py
│   └── validation/             # Validation visualization
│       ├── hypothesis_validation.py
│       └── validation_visualization.py
├── legacy/              # Legacy components (deprecated)
└── utils.py             # Shared utilities
```

## Design Principles

### 1. Single Responsibility
Each component does ONE thing well:
- **Configuration builders** → Create experiment configs
- **Selectors** → Render UI for parameter selection
- **Results** → Display experiment outcomes
- **Orchestration** → Coordinate workflows
- **Visualization** → Render charts and plots

### 2. Composability
Components can be combined to build complex UIs:

```python
from uqlab.ui_components.config import build_base_experiment_config
from uqlab.ui_components.selectors import render_model_selector
from uqlab.ui_components.results import render_experiment_results

# Compose components
model_config = render_model_selector()
experiment_config = build_base_experiment_config(**model_config)
render_experiment_results(experiment_config)
```

### 3. Separation of Concerns
- **UI logic** → Streamlit rendering
- **Business logic** → Configuration building
- **Data access** → API calls (via callbacks)

### 4. Testability
All components accept dependencies via parameters (dependency injection):

```python
def render_dataset_selection(
    default_dataset: str,
    default_noise_type: str,
    fetch_stats_callback: Callable  # Injected dependency
) -> Tuple[str, str, Optional[dict]]:
    """Testable: mock fetch_stats_callback in tests"""
    pass
```

## Component Specifications

### Configuration Builders (`config/`)

#### `build_base_experiment_config()`
**Purpose:** Create base experiment configuration dictionary

**Specification:**
- **Input:** Individual experiment parameters (noise_type, model, training, etc.)
- **Output:** Structured configuration dictionary
- **Validation:** Ensures all required fields present
- **Side Effects:** None (pure function)

**Usage:**
```python
config = build_base_experiment_config(
    noise_type="worse_label",
    under_supported="random:2",
    under_train_per_class=50,
    regular_train_per_class=300,
    dinov2_model="small",
    epochs=12,
    # ... other parameters
)
```

#### `build_nested_experiment_config()`
**Purpose:** Create nested configuration with organized sections

**Specification:**
- **Input:** Same as base config
- **Output:** Nested dict with sections (data, model, training, evaluation)
- **Validation:** Type checking via Pydantic models
- **Side Effects:** None

### Selectors (`selectors/`)

#### `render_model_selector()`
**Purpose:** UI for selecting model architecture and parameters

**Specification:**
- **Input:** Optional defaults
- **Output:** Tuple of (model_name, hidden_dim, dropout, use_untrained)
- **UI Elements:** Radio buttons, sliders, checkboxes
- **Validation:** Parameter ranges enforced by UI

**Usage:**
```python
model, hidden_dim, dropout, untrained = render_model_selector()
```

#### `render_training_selector()`
**Purpose:** UI for training hyperparameters

**Specification:**
- **Input:** Optional defaults
- **Output:** Tuple of (epochs, lr, weight_decay, batch_size)
- **UI Elements:** Number inputs, sliders
- **Validation:** Positive values, reasonable ranges

#### `render_evaluation_selector()`
**Purpose:** UI for evaluation configuration

**Specification:**
- **Input:** Training config context
- **Output:** Tuple of (mc_passes, selected_signals, eval_per_group)
- **UI Elements:** Sliders, multiselect
- **Validation:** Signal names validated against available signals

### Results Display (`results/`)

#### `render_experiment_results()`
**Purpose:** Display experiment results with auto-refresh

**Specification:**
- **Input:** API base URL, auth headers, auto_refresh flag
- **Output:** Updated auto_refresh state
- **Side Effects:** Fetches data from API, updates UI
- **Error Handling:** Graceful degradation on API errors

**Usage:**
```python
auto_refresh = render_experiment_results(
    api_base_url="http://localhost:8000",
    get_headers_func=lambda: {"Authorization": "Bearer token"},
    auto_refresh=True
)
```

### Orchestration (`orchestration/`)

#### `render_unified_builder()`
**Purpose:** Unified interface for building experiments

**Specification:**
- **Input:** Dataset context, API callbacks
- **Output:** Experiment configuration
- **Workflow:** Dataset → Model → Training → Evaluation → Submit
- **Validation:** Progressive validation at each step

#### `render_validation_runner()`
**Purpose:** Run validation experiments with preset configurations

**Specification:**
- **Input:** Validation sweep type
- **Output:** Batch experiment configuration
- **Presets:** Label noise sweep, dataset size sweep, etc.

### Visualization (`visualization/`)

#### Signal Visualization (`visualization/signals/`)

**`render_signal_visualizations()`**
- **Purpose:** Display uncertainty signal distributions
- **Input:** Per-sample signal data
- **Output:** Plotly charts (histograms, scatter plots)
- **Features:** Interactive filtering, signal comparison

**`render_signal_diagnostic_viz()`**
- **Purpose:** Diagnostic plots for signal analysis
- **Input:** Validation results, sweep data
- **Output:** Multi-panel diagnostic dashboard
- **Features:** AUROC curves, signal rankings, sweep analysis

#### Validation Visualization (`visualization/validation/`)

**`render_hypothesis_validation_tab()`**
- **Purpose:** Hypothesis validation dashboard
- **Input:** Validation results directory
- **Output:** Interactive validation analysis
- **Features:** Method comparison, signal disentanglement, sweep analysis

## Usage Examples

### Example 1: Simple Experiment Form

```python
import streamlit as st
from uqlab.ui_components.config import build_base_experiment_config
from uqlab.ui_components.selectors import (
    render_model_selector,
    render_training_selector,
    render_evaluation_selector
)

st.title("Create Experiment")

with st.form("experiment_form"):
    # Model selection
    model, hidden_dim, dropout, untrained = render_model_selector()
    
    # Training parameters
    epochs, lr, weight_decay, batch_size = render_training_selector()
    
    # Evaluation configuration
    mc_passes, signals, eval_per_group = render_evaluation_selector()
    
    submitted = st.form_submit_button("Create Experiment")
    
    if submitted:
        config = build_base_experiment_config(
            dinov2_model=model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            train_batch_size=batch_size,
            mc_passes=mc_passes,
            eval_per_group=eval_per_group,
            use_untrained_resnet=untrained,
            # ... other parameters
        )
        st.json(config)
```

### Example 2: Results Dashboard

```python
import streamlit as st
from uqlab.ui_components.results import render_experiment_results

st.title("Experiment Results")

# Auto-refresh toggle
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Render results with auto-refresh
st.session_state.auto_refresh = render_experiment_results(
    api_base_url="http://localhost:8000",
    get_headers_func=lambda: {},
    auto_refresh=st.session_state.auto_refresh
)
```

### Example 3: Validation Dashboard

```python
import streamlit as st
from uqlab.ui_components.visualization.validation import (
    render_hypothesis_validation_tab
)

st.title("Validation Analysis")

render_hypothesis_validation_tab()
```

## Testing

### Unit Testing Components

```python
import pytest
from unittest.mock import Mock
from uqlab.ui_components.config import build_base_experiment_config

def test_build_base_experiment_config():
    """Test configuration builder"""
    config = build_base_experiment_config(
        noise_type="worse_label",
        under_supported="random:2",
        under_train_per_class=50,
        regular_train_per_class=300,
        dinov2_model="small",
        epochs=12,
        learning_rate=0.001,
        weight_decay=0.0001,
        train_batch_size=256,
        eval_per_group=100,
        mc_passes=20,
        use_untrained_resnet=False,
        aleatoric_noise_percentage=0.0,
    )
    
    assert config["noise_type"] == "worse_label"
    assert config["model"]["dinov2_model"] == "small"
    assert config["training"]["epochs"] == 12
```

### Integration Testing

```python
def test_experiment_workflow():
    """Test complete experiment creation workflow"""
    # Mock API
    mock_api = Mock()
    mock_api.create_experiment.return_value = {"id": "exp_123"}
    
    # Build config
    config = build_base_experiment_config(...)
    
    # Submit experiment
    result = mock_api.create_experiment(config)
    
    assert result["id"] == "exp_123"
```

## Best Practices

### 1. Component Design

✅ **DO:**
- Keep components focused on single responsibility
- Accept dependencies via parameters
- Return data, don't mutate state
- Use type hints for all parameters
- Document expected behavior

❌ **DON'T:**
- Mix UI rendering with business logic
- Access global state directly
- Make API calls without callbacks
- Assume specific Streamlit session state structure

### 2. Error Handling

```python
def render_component():
    """Component with proper error handling"""
    try:
        data = fetch_data()
        if not data:
            st.warning("No data available")
            return None
        return render_data(data)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None
```

### 3. State Management

```python
def render_stateful_component():
    """Component with explicit state management"""
    # Initialize state
    if 'component_state' not in st.session_state:
        st.session_state.component_state = default_state()
    
    # Render UI
    new_value = st.selectbox("Select", options)
    
    # Update state
    if new_value != st.session_state.component_state:
        st.session_state.component_state = new_value
    
    return st.session_state.component_state
```

### 4. Performance

```python
@st.cache_data
def expensive_computation(data):
    """Cache expensive computations"""
    return process_data(data)

def render_component():
    """Use cached data"""
    data = expensive_computation(input_data)
    st.plotly_chart(create_chart(data))
```

## Migration Guide

### From Legacy Components

Legacy components in `legacy/` are deprecated. Migrate to new structure:

**Old:**
```python
from uqlab.ui_components.legacy.batch_config import render_batch_config
```

**New:**
```python
from uqlab.ui_components.config import build_batch_experiment_config
from uqlab.ui_components.selectors import render_batch_sweep_config
```

### Breaking Changes

- `render_batch_config()` → Split into `build_batch_experiment_config()` + `render_batch_sweep_config()`
- `render_smart_experiment_selector()` → Not yet implemented (use stubs)
- Direct API calls → Use callback pattern

## Contributing

### Adding New Components

1. **Specify** the component:
   - What does it do?
   - What are inputs/outputs?
   - What are side effects?

2. **Implement** following principles:
   - Single responsibility
   - Dependency injection
   - Type hints
   - Docstrings

3. **Test** the component:
   - Unit tests for logic
   - Integration tests for workflows
   - Manual testing in Streamlit

4. **Document** in this README:
   - Add to architecture diagram
   - Add usage example
   - Update best practices if needed

### Code Review Checklist

- [ ] Component has single, clear responsibility
- [ ] Dependencies injected via parameters
- [ ] Type hints on all parameters and returns
- [ ] Docstring with specification
- [ ] Error handling implemented
- [ ] Tests added
- [ ] README updated

## Troubleshooting

### Import Errors

```python
# ❌ Wrong
from ui_components import render_model_selector

# ✅ Correct
from uqlab.ui_components.selectors import render_model_selector
```

### Missing Dependencies

```bash
# Install with UI dependencies
pip install "uqlab[ui]"
```

### Component Not Rendering

1. Check Streamlit version: `streamlit --version` (need >= 1.28.0)
2. Verify imports are correct
3. Check browser console for errors
4. Enable Streamlit debug mode: `streamlit run app.py --logger.level=debug`

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [UQLab Main README](../../README.md)
- [Packaging Guide](../../PACKAGING_GUIDE.md)
- [Spec-Kit Setup](../../SPEC_KIT_SETUP.md)

## License

MIT License - See LICENSE file in repository root