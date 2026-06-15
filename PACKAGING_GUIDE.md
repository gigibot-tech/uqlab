# UQLab Package - Installation & Distribution Guide

## Package Structure

The uqlab package is now properly configured for PyPI distribution with:
- ✅ `pyproject.toml` - Modern Python packaging configuration
- ✅ `MANIFEST.in` - Non-Python file inclusion rules
- ✅ `py.typed` - Type checking support marker
- ✅ Modular structure with clear separation of concerns

## Installation

### For Development (Editable Install)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with all dependencies
cd uqlab-streamlit/src/uqlab
pip install -e ".[all]"
```

### For Users (From PyPI - when published)

```bash
# Basic installation
pip install uqlab

# With UI components (Streamlit)
pip install uqlab[ui]

# With development tools
pip install uqlab[dev]

# Everything
pip install uqlab[all]
```

### From Git Repository

```bash
# Install directly from GitHub
pip install git+https://github.com/gigibot-tech/uqlab.git

# Or clone and install
git clone https://github.com/gigibot-tech/uqlab.git
cd uqlab
pip install -e ".[all]"
```

## Building Distribution Packages

### Prerequisites

```bash
pip install build twine
```

### Build Process

```bash
# Navigate to package root
cd uqlab-streamlit/src/uqlab

# Build source distribution and wheel
python -m build

# This creates:
# - dist/uqlab-0.1.0.tar.gz (source distribution)
# - dist/uqlab-0.1.0-py3-none-any.whl (wheel)
```

### Verify Build

```bash
# Check package contents
tar -tzf dist/uqlab-0.1.0.tar.gz

# Verify wheel
unzip -l dist/uqlab-0.1.0-py3-none-any.whl
```

### Test Installation

```bash
# Create fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install dist/uqlab-0.1.0-py3-none-any.whl

# Test import
python -c "import uqlab; print(uqlab.__version__)"
python -c "from uqlab.ui_components import *; print('UI components OK')"
```

## Publishing to PyPI

### Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ uqlab
```

### Production PyPI

```bash
# Upload to PyPI
python -m twine upload dist/*

# Verify
pip install uqlab
```

## Package Structure

```
uqlab/
├── pyproject.toml          # Package configuration
├── MANIFEST.in             # File inclusion rules
├── py.typed                # Type checking marker
├── README.md               # Package documentation
├── __init__.py             # Package entry point
├── 1_data/                 # Data loading modules
├── 2_models/               # Model implementations
├── 3_training/             # Training loops
├── 4_evaluation/           # Evaluation metrics
├── 5_api/                  # API interfaces
├── 7_orchestration/        # Workflow orchestration
├── ui_components/          # Streamlit UI components
│   ├── config/             # Configuration builders
│   ├── selectors/          # UI selectors
│   ├── results/            # Results display
│   ├── orchestration/      # Workflow UI
│   └── visualization/      # Visualization components
├── shared/                 # Shared utilities
└── notebook_support/       # Jupyter notebook helpers
```

## Dependencies

### Core Dependencies
- PyTorch >= 2.0.0
- NumPy >= 1.24.0, < 2.0.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.3.0
- Matplotlib >= 3.7.0
- Plotly >= 5.14.0

### Optional Dependencies
- **UI**: Streamlit >= 1.28.0
- **Dev**: pytest, black, mypy, pylint

## Version Management

Update version in `pyproject.toml`:

```toml
[project]
version = "0.1.0"  # Update this
```

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality
- **PATCH**: Backward-compatible bug fixes

## Troubleshooting

### Import Errors

```python
# Verify package is installed
pip list | grep uqlab

# Check import paths
python -c "import uqlab; print(uqlab.__file__)"
```

### Missing Dependencies

```bash
# Reinstall with all dependencies
pip install --force-reinstall "uqlab[all]"
```

### Build Errors

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info

# Rebuild
python -m build
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install build pytest
          pip install -e ".[all]"
      - name: Run tests
        run: pytest
      - name: Build package
        run: python -m build
```

## Next Steps

1. ✅ Package structure created
2. ⏳ Add LICENSE file
3. ⏳ Update README.md with usage examples
4. ⏳ Add comprehensive tests
5. ⏳ Set up CI/CD pipeline
6. ⏳ Publish to Test PyPI
7. ⏳ Publish to PyPI

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI](https://pypi.org/)
- [Test PyPI](https://test.pypi.org/)
- [Semantic Versioning](https://semver.org/)