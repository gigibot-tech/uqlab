"""
Modular Uncertainty Computation Suite

Provides a unified interface for computing multiple uncertainty metrics:
- MC Dropout Variance (σ²)
- Integrity Score (I)
- Surgical Score (σ² × I)
- DualXDA Signal Uncertainty
- DualXDA Triage Uncertainty
- DualXDA Spurious Score

Usage:
    suite = UncertaintySuite(model, train_dataset, device)
    results = suite.compute_all_uncertainties(test_loader, n_mc_passes=50)
"""

from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from uqlab.legacy_metrics.surgical_score import batch_surgical_score
from uqlab.legacy_experiments.dualxda_stream import compute_dualxda_scores_streaming
from uqlab.triage.dualxda_axioms import DualXDATracer, AxiomThresholds, infer_classifier_layer_name


class UncertaintySuite:
    """
    Unified interface for computing multiple uncertainty metrics.
    
    Efficiently computes:
    1. MC Dropout components (variance, integrity, surgical score)
    2. DualXDA components (signal, triage, spurious scores)
    
    All metrics are computed in a single pass where possible.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: torch.device,
        model_type: str = 'dinov2',
        enable_dualxda: bool = True,
        dualxda_thresholds: Optional[AxiomThresholds] = None
    ):
        """
        Initialize uncertainty suite.
        
        Args:
            model: PyTorch model with MC Dropout enabled
            train_dataset: Training dataset for DualXDA
            device: Device to run computations on
            model_type: 'dinov2' or 'resnet18_mcdropout' (determines layer name)
            enable_dualxda: Whether to compute DualXDA scores
            dualxda_thresholds: Custom thresholds for DualXDA axioms
        """
        self.model = model
        self.train_dataset = train_dataset
        self.device = device
        self.model_type = model_type
        self.enable_dualxda = enable_dualxda
        
        # Initialize DualXDA tracer if enabled
        self.dualxda_tracer = None
        if enable_dualxda:
            try:
                layer_name = infer_classifier_layer_name(model)
                self.dualxda_tracer = DualXDATracer(
                    model=model,
                    train_dataset=train_dataset,
                    layer_name=layer_name,
                    device=str(device),
                    cache_dir='./cache/dualxda',
                    thresholds=dualxda_thresholds or AxiomThresholds(
                        tau_mass=0.0,
                        tau_signal=0.04,
                        tau_dom=0.13,
                        pile_metric='coherence'
                    )
                )
                print(f"✓ DualXDA tracer initialized (layer: {layer_name})")
            except Exception as e:
                print(f"⚠️  DualXDA initialization failed: {e}")
                print("   Continuing with MC Dropout scores only...")
                self.enable_dualxda = False
    
    def compute_all_uncertainties(
        self,
        test_loader: DataLoader,
        n_mc_passes: int = 50,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all uncertainty metrics in an efficient manner.
        
        Args:
            test_loader: DataLoader for test set
            n_mc_passes: Number of MC Dropout forward passes
            show_progress: Whether to show tqdm progress bars
            
        Returns:
            Dictionary containing:
              - predictions: [N, C] softmax predictions
              - labels: [N] ground truth labels
              - mc_variance: [N] MC Dropout variance
              - integrity_score: [N] data quality score
              - surgical_score: [N] epistemic uncertainty
              - signal_uncertainty: [N] DualXDA signal uncertainty (if enabled)
              - triage_uncertainty: [N] DualXDA triage uncertainty (if enabled)
              - spurious_score: [N] DualXDA spurious score (if enabled)
              - mass: [N] DualXDA mass (if enabled)
              - signal: [N] DualXDA signal (if enabled)
              - coherence: [N] DualXDA coherence (if enabled)
              - dominance: [N] DualXDA dominance (if enabled)
        """
        # Step 1: Compute MC Dropout components (single pass)
        print("Computing MC Dropout predictions and uncertainty components...")
        mc_results = batch_surgical_score(
            self.model, test_loader, n_passes=n_mc_passes, device=self.device
        )
        
        results = {
            'predictions': mc_results['predictions'],
            'labels': mc_results['labels'],
            'mc_variance': mc_results['mc_variance'],
            'integrity_score': mc_results['integrity_score'],
            'surgical_score': mc_results['surgical_score'],
        }
        
        # Step 2: Compute DualXDA components if enabled
        if self.enable_dualxda and self.dualxda_tracer is not None:
            print("\nComputing DualXDA Uncertainty Scores...")
            try:
                dualxda_results = self._compute_dualxda_scores(
                    test_loader, results['predictions'], show_progress
                )
                results.update(dualxda_results)
                print("✓ DualXDA scores computed successfully")
            except Exception as e:
                print(f"⚠️  DualXDA computation failed: {e}")
                print("   Continuing with MC Dropout scores only...")
        
        return results
    
    def _compute_dualxda_scores(
        self,
        test_loader: DataLoader,
        predictions: torch.Tensor,
        show_progress: bool
    ) -> Dict[str, torch.Tensor]:
        """Compute DualXDA scalars without ever concatenating giant [N_test, N_train] tensors."""
        if self.dualxda_tracer is None:
            raise RuntimeError("DualXDA tracer is not initialized")

        # predictions is [N,C] mean probs aligned with loader order (shuffle=False).
        dualxda_scores = compute_dualxda_scores_streaming(
            tracer=self.dualxda_tracer,
            dataloader=test_loader,
            mean_predictions=predictions,
            device=self.device,
            drop_zero_columns=False,
            desc=("DualXDA attributions" if show_progress else "DualXDA attributions"),
        )

        return dualxda_scores

    def get_uncertainty_dict(self, results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract uncertainty scores suitable for AURC calculation.
        
        Args:
            results: Output from compute_all_uncertainties()
            
        Returns:
            Dictionary mapping metric names to uncertainty tensors:
              - 'mc_variance': MC Dropout variance
              - 'integrity': Inverted integrity score (low integrity = high uncertainty)
              - 'surgical': Surgical score (σ² × I)
              - 'signal_uncertainty': DualXDA signal uncertainty (if available)
              - 'triage_uncertainty': DualXDA triage uncertainty (if available)
              - 'spurious_score': DualXDA spurious score (if available)
        """
        uncertainties = {
            'mc_variance': results['mc_variance'],
            'integrity': 1.0 - results['integrity_score'],  # Invert
            'surgical': results['surgical_score'],
        }
        
        # Add DualXDA scores if available
        if 'signal_uncertainty' in results:
            uncertainties['signal_uncertainty'] = results['signal_uncertainty']
        if 'triage_uncertainty' in results:
            uncertainties['triage_uncertainty'] = results['triage_uncertainty']
        if 'spurious_score' in results:
            uncertainties['spurious_score'] = results['spurious_score']
        
        return uncertainties
    
    def get_display_names(self) -> Dict[str, str]:
        """
        Get human-readable display names for uncertainty metrics.
        
        Returns:
            Dictionary mapping metric keys to display names
        """
        names = {
            'mc_variance': 'MC Dropout Variance (σ²)',
            'integrity': 'Integrity Score (I⁻¹)',
            'surgical': 'Surgical Score (σ²×I) ⭐',
        }
        
        if self.enable_dualxda:
            names.update({
                'signal_uncertainty': 'Signal Uncertainty (DualXDA)',
                'triage_uncertainty': 'Triage Uncertainty (DualXDA)',
                'spurious_score': 'Spurious Score (DualXDA)'
            })
        
        return names
    
    def cleanup(self):
        """Clean up resources (remove hooks, etc.)."""
        if self.dualxda_tracer is not None:
            try:
                self.dualxda_tracer.remove_hook()
            except Exception:
                pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

# Made with Bob
