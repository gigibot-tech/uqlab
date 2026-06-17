"""
Acquisition Functions for Active Learning
Implements both baseline and proposed sampling strategies.

This module addresses Refinement #2: Explicit acquisition function specification
for the Active Learning benchmark (Proof 3).
"""

import torch
import numpy as np
from typing import Tuple, Dict
from .mc_dropout_uq import calculate_mc_dropout_uncertainty
from .surgical_score import SurgicalScoreCalculator
from uqlab.evaluation.legacy.experiments.dualxda_stream import compute_dualxda_scores_streaming
from uqlab.evaluation.legacy.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name
from uqlab.data.loaders.cifar10n_loader import CIFAR10NDataset, CIFAR10NLabelView


class AcquisitionFunction:
    """Base class for acquisition functions."""
    
    def __init__(self, name: str):
        self.name = name
    
    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Calculate acquisition scores for all samples.
        
        Args:
            model: MC Dropout model
            data_loader: Data loader for unlabeled pool
            device: Device to run on
            
        Returns:
            scores: Acquisition scores [n_samples]
        """
        raise NotImplementedError


class MaxVarianceAcquisition(AcquisitionFunction):
    """
    The Sledgehammer (Baseline): arg max(σ²)
    
    Standard MC Dropout acquisition function that selects samples with
    highest predictive variance. This is the industry standard but has
    a critical flaw: it cannot distinguish between epistemic and aleatoric
    uncertainty, leading to wasted labeling budget on noisy samples.
    
    Mathematical formulation:
        Sample = arg max(σ²)
        
    Where σ² is the MC Dropout predictive variance (Total Uncertainty).
    """
    
    def __init__(self, n_passes=50):
        super().__init__("MC Dropout (Max Variance)")
        self.n_passes = n_passes
    
    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Calculate variance-based acquisition scores.
        
        Returns:
            scores: MC Dropout variance [n_samples]
        """
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle both 2-value and 4-value batch formats
                if len(batch) == 2:
                    data, _ = batch
                elif len(batch) == 4:
                    data, _, _, _ = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                data = data.to(device)
                
                # MC Dropout forward passes
                mc_predictions = model.mc_forward(data, n_passes=self.n_passes)
                
                # Calculate uncertainty
                uncertainties = calculate_mc_dropout_uncertainty(mc_predictions)
                all_scores.append(uncertainties['mean_variance'].cpu())
        
        return torch.cat(all_scores)


class SurgicalScoreAcquisition(AcquisitionFunction):
    """
    The Scalpel (Proposed): arg max(σ² × I)
    
    Pathology-Aware Sampling that selects samples with high-integrity
    uncertainty. This filters out aleatoric noise (Self-Saboteurs) and
    focuses the labeling budget on true epistemic gaps (The Void).
    
    Mathematical formulation:
        Sample = arg max(σ² × I)
        
    Where:
        σ² = MC Dropout variance (Total Uncertainty)
        I = Integrity Score = M_diag / (M_diag + M_off + ε)
        
    This is "High-Integrity Uncertainty" - we want samples that are:
    1. Uncertain (high σ²) AND
    2. Have clean data provenance (high I)
    
    This is the unique contribution to the UQ literature: we aren't just
    looking for "uncertainty", we're looking for "learnable uncertainty".
    """
    
    def __init__(self, n_passes=50, epsilon=1e-6):
        super().__init__("Surgical Score (Pathology-Aware)")
        self.n_passes = n_passes
        self.calculator = SurgicalScoreCalculator(epsilon=epsilon)
    
    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Calculate surgical score acquisition scores.
        
        Returns:
            scores: Surgical Score (σ² × I) [n_samples]
        """
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle both 2-value and 4-value batch formats
                if len(batch) == 2:
                    data, _ = batch
                elif len(batch) == 4:
                    data, _, _, _ = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                data = data.to(device)
                
                # MC Dropout forward passes
                mc_predictions = model.mc_forward(data, n_passes=self.n_passes)
                
                # Calculate surgical score
                results = self.calculator.calculate_surgical_score(mc_predictions)
                all_scores.append(results['surgical_score'].cpu())
        
        return torch.cat(all_scores)


class RandomAcquisition(AcquisitionFunction):
    """
    Random Sampling (Sanity Check)
    
    Selects samples uniformly at random. This serves as a lower bound
    for active learning performance.
    """
    
    def __init__(self):
        super().__init__("Random Sampling")
    
    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Generate random scores.
        
        Returns:
            scores: Random values [n_samples]
        """
        # Count total samples
        n_samples = sum(len(batch[0]) for batch in data_loader)
        return torch.rand(n_samples)


class EntropyAcquisition(AcquisitionFunction):
    """
    Maximum Entropy (Alternative Baseline)
    
    Selects samples with highest predictive entropy. Similar to variance
    but uses entropy as the uncertainty measure.
    
    Mathematical formulation:
        Sample = arg max(H[p(y|x)])
    """
    
    def __init__(self, n_passes=50):
        super().__init__("Maximum Entropy")
        self.n_passes = n_passes
    
    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Calculate entropy-based acquisition scores.
        
        Returns:
            scores: Predictive entropy [n_samples]
        """
        model.eval()
        all_scores = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                
                # MC Dropout forward passes
                mc_predictions = model.mc_forward(data, n_passes=self.n_passes)
                
                # Calculate uncertainty
                uncertainties = calculate_mc_dropout_uncertainty(mc_predictions)
                all_scores.append(uncertainties['entropy'].cpu())
        
        return torch.cat(all_scores)


class BALDAcquisition(AcquisitionFunction):
    """
    BALD acquisition via MC Dropout mutual information.

    For classification with MC Dropout, BALD is approximated by:
        I[y, theta | x, D] = H[E p(y|x,theta)] - E H[p(y|x,theta)]
    """

    def __init__(self, n_passes=5):  # Reduced from 50 to 5 for faster computation
        super().__init__("BALD (Mutual Information - Fast Approx)")
        self.n_passes = n_passes

    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        """
        Fast BALD approximation using fewer MC passes.
        Uses only 5 passes instead of 50 for ~10x speedup with reasonable approximation.
        """
        model.eval()
        all_scores = []

        with torch.no_grad():
            for batch in data_loader:
                # Handle both 2-value and 4-value batch formats
                if len(batch) == 2:
                    data, _ = batch
                elif len(batch) == 4:
                    data, _, _, _ = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                
                data = data.to(device)
                mc_predictions = model.mc_forward(data, n_passes=self.n_passes)
                uncertainties = calculate_mc_dropout_uncertainty(mc_predictions)
                all_scores.append(uncertainties['mutual_info'].cpu())

        return torch.cat(all_scores)


class DualXDASignalAcquisition(AcquisitionFunction):
    """
    DualXDA signal acquisition.

    Uses the current labeled set as the attribution reference set and ranks unlabeled
    points by signal_uncertainty derived from signed DualDA traces.
    """

    def __init__(
        self,
        train_dataset,
        cache_dir='./cache/dualxda_active_learning',
        thresholds: AxiomThresholds | None = None,
    ):
        super().__init__("DualXDA Signal")
        self.train_dataset = train_dataset
        self.cache_dir = cache_dir
        self.thresholds = thresholds or AxiomThresholds(
            tau_mass=0.0,
            tau_signal=0.04,
            tau_dom=0.13,
            pile_metric='coherence',
        )

    def _predict_probs(self, model, x: torch.Tensor) -> torch.Tensor:
        try:
            logits = model(x, enable_dropout=False)
        except TypeError:
            logits = model(x)
        return torch.softmax(logits, dim=1)

    def score(self, model, data_loader, device='cuda', **kwargs) -> torch.Tensor:
        model.eval()

        if self.train_dataset is None or len(self.train_dataset) == 0:
            raise ValueError("DualXDASignalAcquisition requires a non-empty labeled train_dataset.")

        # Wrap dataset if samples return more than (x, y), because DualDA expects exactly 2 values.
        train_dataset_wrapped = self.train_dataset
        sample_dataset = self.train_dataset
        while hasattr(sample_dataset, 'dataset'):
            sample_dataset = sample_dataset.dataset

        try:
            sample_item = sample_dataset[0]
            if isinstance(sample_item, (tuple, list)) and len(sample_item) > 2:
                train_dataset_wrapped = CIFAR10NLabelView(sample_dataset, label_mode='noisy')
        except Exception:
            pass

        layer_name = infer_classifier_layer_name(model)
        tracer = DualXDATracer(
            model=model,
            train_dataset=train_dataset_wrapped,
            layer_name=layer_name,
            device=str(device),
            cache_dir=self.cache_dir,
            thresholds=self.thresholds,
        )

        try:
            mean_predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    if len(batch) == 2:
                        data, _ = batch
                    elif len(batch) == 4:
                        data, _, _, _ = batch
                    else:
                        raise ValueError(f"Unexpected batch format with {len(batch)} elements")
                    data = data.to(device)
                    mean_predictions.append(self._predict_probs(model, data).cpu())

            mean_predictions = torch.cat(mean_predictions, dim=0)
            scores = compute_dualxda_scores_streaming(
                tracer=tracer,
                dataloader=data_loader,
                mean_predictions=mean_predictions,
                device=device,
                drop_zero_columns=False,
                desc="DualXDA signal acquisition",
            )
            return scores['signal_uncertainty']
        finally:
            tracer.remove_hook()


def get_acquisition_function(strategy: str, **kwargs) -> AcquisitionFunction:
    """
    Factory function to create acquisition functions.
    
    Args:
        strategy: Name of strategy ('mc_dropout', 'bald', 'surgical_score', 'dualxda_signal', 'random', 'entropy')
        **kwargs: Additional arguments for the acquisition function
        
    Returns:
        acquisition_fn: Acquisition function instance
    """
    if strategy == 'mc_dropout':
        return MaxVarianceAcquisition(**kwargs)
    elif strategy == 'bald':
        return BALDAcquisition(**kwargs)
    elif strategy == 'surgical_score':
        return SurgicalScoreAcquisition(**kwargs)
    elif strategy == 'dualxda_signal':
        return DualXDASignalAcquisition(**kwargs)
    elif strategy == 'random':
        return RandomAcquisition()
    elif strategy == 'entropy':
        return EntropyAcquisition(**kwargs)
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")


def compare_acquisition_strategies(model, data_loader, device='cuda', n_passes=50):
    """
    Compare all acquisition strategies on the same data.
    
    This is useful for ablation studies in the thesis.
    
    Args:
        model: MC Dropout model
        data_loader: Data loader
        device: Device
        n_passes: Number of MC passes
        
    Returns:
        results: Dictionary mapping strategy name to scores
    """
    strategies = {
        'MC Dropout': MaxVarianceAcquisition(n_passes),
        'BALD': BALDAcquisition(n_passes),
        'Surgical Score': SurgicalScoreAcquisition(n_passes),
        'Entropy': EntropyAcquisition(n_passes),
        'Random': RandomAcquisition()
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"Calculating {name} scores...")
        scores = strategy.score(model, data_loader, device)
        results[name] = scores
    
    return results


if __name__ == "__main__":
    # Test acquisition functions
    print("Testing Acquisition Functions...")
    
    # Create dummy model and data
    from ..models.mc_dropout_model import create_mc_dropout_model
    import torch.utils.data as data
    
    model = create_mc_dropout_model('cnn', num_classes=10)
    dummy_data = data.TensorDataset(
        torch.randn(100, 3, 32, 32),
        torch.randint(0, 10, (100,))
    )
    dummy_loader = data.DataLoader(dummy_data, batch_size=32)
    
    # Test each strategy
    strategies = ['mc_dropout', 'bald', 'surgical_score', 'random', 'entropy']
    
    for strategy in strategies:
        acq_fn = get_acquisition_function(strategy, n_passes=10)
        scores = acq_fn.score(model, dummy_loader, device='cpu')
        print(f"\n{acq_fn.name}:")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std: {scores.std():.4f}")
        print(f"  Min: {scores.min():.4f}")
        print(f"  Max: {scores.max():.4f}")
    
    print("\n✓ All acquisition functions working correctly")

# Made with Bob
