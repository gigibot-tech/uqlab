# Streamlit Dashboard for UQ Classification

An interactive web-based dashboard for exploring training results, visualizing decision boundaries, and comparing model checkpoints.

## Features

### 🔍 Experiment Selection
- Browse all available experiments and training runs
- Select specific experiments from the sidebar
- View experiment metadata and hyperparameters
- Quick access to latest runs

### 📊 Training Metrics Visualization
- **Final Metrics Cards**: Display key performance indicators
- **Training Curves**: Interactive line charts for:
  - Loss curves (training and test)
  - Accuracy curves (training and test)
  - All metrics combined view
- **Tabbed Interface**: Easy navigation between different metric views

### 🎯 Decision Boundary Explorer
- **Epoch Slider**: Navigate through training progression
- **Side-by-side Visualization**: Compare training and test decision boundaries
- **Image Display**: High-quality decision boundary plots
- **Epoch Information**: Current epoch and total epochs

### 🔄 Checkpoint Comparison
- **Multi-select**: Choose multiple epochs to compare
- **Grid Layout**: Configurable images per row (1-4)
- **Evolution View**: See how decision boundaries evolve during training
- **Default Selection**: Automatically shows last 3 checkpoints

### 💾 Data Export
- **CSV Export**: Download training metrics as CSV
- **JSON Export**: Download complete experiment data
- **One-click Downloads**: Easy data extraction for further analysis

### ⚙️ Configuration
- **Directory Settings**: Customize experiments and visualizations paths
- **Refresh Button**: Reload data from disk
- **Responsive Layout**: Adapts to different screen sizes

## Installation

### Prerequisites

```bash
# Install required packages
pip install streamlit pandas pillow numpy
```

Or install from the project requirements:

```bash
pip install -r uq_classification/requirements_viz.txt
```

### Verify Installation

```bash
streamlit --version
```

## Usage

### Basic Usage

Run the Streamlit app from the project root:

```bash
streamlit run uq_classification/streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Custom Directories

If your experiments are in a different location:

1. Start the app normally
2. Use the sidebar "Directory Settings" expander
3. Update the paths:
   - **Experiments Directory**: Where JSON files are stored
   - **Visualizations Directory**: Where decision boundary images are stored
4. Click "Refresh Data" to reload

### Command Line Options

```bash
# Run on a different port
streamlit run uq_classification/streamlit_app.py --server.port 8502

# Run in headless mode (no browser)
streamlit run uq_classification/streamlit_app.py --server.headless true

# Enable file watching for development
streamlit run uq_classification/streamlit_app.py --server.fileWatcherType auto
```

## Workflow Integration

### 1. Train a Model

```bash
python uq_classification/train_with_checkpoints.py \
    --experiment-name my_experiment \
    --num-epochs 50 \
    --checkpoint-freq 5 \
    --viz-freq 5
```

This creates:
- `experiments/my_experiment/YYYYMMDD_HHMMSS.json` - Training data
- `visualizations/my_experiment/YYYYMMDD_HHMMSS/epoch_*.png` - Decision boundaries

### 2. Launch Dashboard

```bash
streamlit run uq_classification/streamlit_app.py
```

### 3. Explore Results

1. **Select Experiment**: Choose from sidebar
2. **View Metrics**: Check training curves and final performance
3. **Explore Boundaries**: Use slider to see evolution
4. **Compare Checkpoints**: Select multiple epochs for comparison
5. **Export Data**: Download metrics or full experiment data

## Dashboard Sections

### Sidebar

- **Experiment Selection**: Choose experiment and run
- **Configuration**: Set directory paths
- **Refresh**: Reload data from disk

### Main Area

1. **Experiment Metadata**
   - Run ID, experiment name, run name
   - Expandable hyperparameters table

2. **Final Metrics**
   - Key performance indicators in cards
   - Automatically formatted values

3. **Training Curves**
   - Loss tab: All loss metrics
   - Accuracy tab: All accuracy metrics
   - All Metrics tab: Combined view

4. **Decision Boundary Explorer**
   - Epoch slider for navigation
   - Full-width image display
   - File and epoch information

5. **Checkpoint Comparison**
   - Multi-select for epochs
   - Configurable grid layout
   - Side-by-side comparison

6. **Export Data**
   - CSV download for metrics
   - JSON download for full data

## Tips and Tricks

### Performance

- **Caching**: The app caches loaded data for fast navigation
- **Refresh**: Use the refresh button after new training runs
- **Large Datasets**: Images are loaded on-demand for efficiency

### Visualization

- **Zoom**: Click on images to view full-size
- **Grid Layout**: Adjust "Images per row" for optimal viewing
- **Epoch Selection**: Use slider for smooth navigation

### Data Analysis

- **Export Metrics**: Download CSV for analysis in Excel/Python
- **Compare Runs**: Switch between runs to compare performance
- **Track Progress**: Use checkpoint comparison to see training evolution

### Troubleshooting

**No experiments found:**
- Check that training has completed
- Verify experiments directory path
- Click "Refresh Data" button

**Images not loading:**
- Ensure visualizations were generated during training
- Check visualizations directory path
- Verify image files exist on disk

**Metrics not displaying:**
- Confirm experiment JSON contains metrics
- Check that training logged metrics properly
- Try refreshing the page

## Directory Structure

Expected directory layout:

```
project_root/
├── experiments/
│   └── my_experiment/
│       ├── 20260515_101530.json
│       └── 20260515_102045.json
├── visualizations/
│   └── my_experiment/
│       ├── 20260515_101530/
│       │   ├── epoch_5.png
│       │   ├── epoch_10.png
│       │   └── epoch_15.png
│       └── 20260515_102045/
│           └── ...
└── uq_classification/
    └── streamlit_app.py
```

## Customization

### Adding New Metrics

The app automatically displays all metrics in the JSON file. To add custom metrics:

1. Log them during training using `ExperimentTracker`
2. They will appear in the dashboard automatically
3. Loss/accuracy metrics get special tab treatment

### Custom Styling

Edit the CSS in the `st.markdown()` section at the top of `streamlit_app.py`:

```python
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #your-color;
    }
    /* Add your custom styles */
</style>
""", unsafe_allow_html=True)
```

### Adding New Features

The app is modular. To add new sections:

1. Create a new `render_*()` function
2. Call it in the `main()` function
3. Use Streamlit components for interactivity

## API Reference

### Key Functions

#### `load_experiments(experiments_dir: str) -> Dict[str, List[str]]`
Load all available experiments and their runs.

#### `load_experiment_data(experiment_name: str, run_id: str) -> Optional[Dict]`
Load experiment data from JSON file.

#### `get_checkpoint_images(experiment_name: str, run_id: str) -> List[Tuple[int, Path]]`
Get all checkpoint visualization images for an experiment.

#### `extract_metrics_dataframe(metrics: Dict) -> pd.DataFrame`
Convert metrics dictionary to pandas DataFrame.

### UI Components

- `render_header()`: Main dashboard header
- `render_experiment_selector()`: Experiment/run selection
- `render_experiment_metadata()`: Metadata display
- `render_metrics_summary()`: Final metrics cards
- `render_training_curves()`: Training curve plots
- `render_decision_boundary_explorer()`: Epoch slider and images
- `render_checkpoint_comparison()`: Multi-checkpoint grid
- `render_data_export()`: Export buttons

## Examples

### Example 1: Quick Start

```bash
# Train a model
python uq_classification/train_with_checkpoints.py --experiment-name demo

# Launch dashboard
streamlit run uq_classification/streamlit_app.py

# Select "demo" experiment from sidebar
```

### Example 2: Compare Multiple Runs

```bash
# Train multiple runs
python uq_classification/train_with_checkpoints.py --experiment-name comparison --learning-rate 0.001
python uq_classification/train_with_checkpoints.py --experiment-name comparison --learning-rate 0.01

# Launch dashboard and switch between runs to compare
streamlit run uq_classification/streamlit_app.py
```

### Example 3: Export and Analyze

1. Run training with checkpoints
2. Open dashboard
3. Navigate to desired experiment
4. Click "Download Metrics (CSV)"
5. Analyze in your favorite tool

## Advanced Usage

### Programmatic Access

You can also use the data loading functions programmatically:

```python
from uq_classification.streamlit_app import load_experiments, load_experiment_data

# Load all experiments
experiments = load_experiments("experiments")

# Load specific experiment
data = load_experiment_data("my_experiment", "20260515_101530")

# Access metrics
metrics = data['metrics']
params = data['params']
```

### Integration with Jupyter

```python
import pandas as pd
from uq_classification.streamlit_app import load_experiment_data, extract_metrics_dataframe

# Load experiment
data = load_experiment_data("my_experiment", "20260515_101530")

# Convert to DataFrame
df = extract_metrics_dataframe(data['metrics'])

# Analyze
df.plot()
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run uq_classification/streamlit_app.py --server.port 8502
   ```

2. **Module not found**
   ```bash
   pip install streamlit pandas pillow
   ```

3. **Images not displaying**
   - Check file permissions
   - Verify image paths
   - Ensure PNG format

4. **Slow performance**
   - Clear cache: Click "Refresh Data"
   - Reduce number of checkpoints
   - Use smaller images

## Contributing

To extend the dashboard:

1. Follow the existing code structure
2. Add docstrings to new functions
3. Use Streamlit best practices
4. Test with various experiment types
5. Update this README

## License

Part of the UQ Classification project.

---

**Made with ❤️ using Streamlit**

For issues or questions, please refer to the main project documentation.