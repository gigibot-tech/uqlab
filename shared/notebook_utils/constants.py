"""Constants for validation notebooks and reports."""

ARCHITECTURE_STYLES = {
    "DINOv2 + MLP": {"color": "#1f77b4", "marker": "o"},
    "CNN MC Dropout": {"color": "#ff7f0e", "marker": "s"},
    "ResNet18 MC Dropout": {"color": "#2ca02c", "marker": "^"},
}

SWEEP_TO_X = {
    "dataset_size": "dataset_size",
    "label_noise": "noise_percent",
}

# Mapping of CSV column prefixes to readable signal names
UNCERTAINTY_SIGNALS = {
    "msp_uncertainty": "MSP (Max Softmax Probability)",
    "predictive_entropy": "Predictive Entropy",
    "mutual_info": "Mutual Information",
    "inverse_coherence": "Inverse Coherence",
    "dominance": "Dominance",
    "inverse_mass": "Inverse Mass",
    "inverse_logit_magnitude": "Inverse Logit Magnitude",
}

# Made with Bob
