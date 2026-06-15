"""
Signal Computation and Attribution (ML Core Layer)

This module computes uncertainty signals using ML algorithms:
- Entropy, mutual information, predictive uncertainty
- Pure Python/NumPy/PyTorch implementations
- No UI dependencies

Architecture:
- Layer: ML Core (computation)
- Used by: Evaluation pipeline, batch jobs
- Visualized by: uqlab.ui_components.visualization.signals (UI layer)

Related Modules:
- Visualization: uqlab.ui_components.visualization.signals (Streamlit/Plotly charts)
- Orchestration: uqlab_orchestrator.sweeps (batch execution)
"""
from .attribution import *
from .formulas import *
