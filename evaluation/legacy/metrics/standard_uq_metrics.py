"""
Standard Uncertainty Quantification Metrics
Implements ECE, NLL, and AURC for comparison with SOTA papers.

These metrics are standard in the UQ literature and allow us to:
1. Compare Surgical Score with entropy-based ranking (ECE, AURC)
2. Show that DINOv2 provides more stable uncertainty estimates (NLL)
3. Bridge to CIFAR-10N noisy label research

References:
- ECE: Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017)
- AURC: Geifman & El-Yaniv "Selective Classification" (JMLR 2017)
- NLL: Standard probabilistic metric for model confidence
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import auc


class StandardUQMetrics:
    """
    Calculate standard uncertainty quantification metrics.
    
    Metrics:
    - ECE (Expected Calibration Error): Measures calibration quality
    - NLL (Negative Log-Likelihood): Measures prediction confidence
    - AURC (Area Under Risk-Coverage): Measures selective classification quality
    """
    
    def __init__(self, n_bins: int = 15):
        """
        Args:
            n_bins: Number of bins for ECE calculation
        """
        self.n_bins = n_bins
    
    def calculate_ece(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        ECE measures the difference between confidence and accuracy.
        Lower ECE = better calibrated model.
        
        Formula:
            ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
        
        Where B_m are bins of predictions grouped by confidence.
        
        Args:
            predictions: Predicted class probabilities [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            confidences: Optional confidence scores [batch_size]
                        If None, uses max prediction probability
        
        Returns:
            ECE score (lower is better)
        """
        # Get confidence scores (max probability if not provided)
        if confidences is None:
            confidences, predicted_classes = predictions.max(dim=1)
        else:
            predicted_classes = predictions.argmax(dim=1)
        
        # Check if predictions are correct
        correct = (predicted_classes == labels).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            # Find samples in this bin
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add weighted difference to ECE
                ece += prop_in_bin * torch.abs(accuracy_in_bin - avg_confidence_in_bin)
        
        # Handle case where ece is still a float (no bins had samples) or is a tensor
        if isinstance(ece, torch.Tensor):
            return ece.item()
        else:
            return float(ece)
    
    def calculate_nll(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Calculate Negative Log-Likelihood (NLL).
        
        NLL measures how well the model's predicted probabilities
        match the true labels. Lower NLL = better predictions.
        
        Formula:
            NLL = -1/n * Σ log(p(y_true | x))
        
        Args:
            predictions: Predicted class probabilities [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
        
        Returns:
            NLL score (lower is better)
        """
        # Get probability of true class
        true_class_probs = predictions[torch.arange(len(labels)), labels]
        
        # Calculate negative log-likelihood
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        nll = -torch.log(true_class_probs + epsilon).mean()
        
        return nll.item()
    
    def calculate_aurc(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        uncertainty_scores: torch.Tensor
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate Area Under Risk-Coverage Curve (AURC).
        
        AURC measures selective classification quality:
        - Rank samples by uncertainty (low to high)
        - Gradually reject most uncertain samples
        - Plot accuracy vs coverage (% of samples kept)
        - Lower AURC = better uncertainty ranking
        
        This is KEY for comparing Surgical Score vs Entropy!
        
        Formula:
            AURC = ∫ risk(c) dc  where c ∈ [0, 1]
            risk(c) = error rate when covering top c% of samples
        
        Args:
            predictions: Predicted class probabilities [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            uncertainty_scores: Uncertainty scores [batch_size]
                               (lower = more certain, will be kept)
        
        Returns:
            - aurc: Area under risk-coverage curve (lower is better)
            - coverages: Coverage values (x-axis for plotting)
            - risks: Risk values (y-axis for plotting)
        """
        # Get predicted classes
        predicted_classes = predictions.argmax(dim=1)
        
        # Calculate errors (1 = wrong, 0 = correct)
        errors = (predicted_classes != labels).float()
        
        # Sort by uncertainty (ascending = most certain first)
        sorted_indices = torch.argsort(uncertainty_scores)
        sorted_errors = errors[sorted_indices]
        
        # Calculate cumulative risk at different coverage levels
        n_samples = len(sorted_errors)
        coverages = []
        risks = []
        
        for i in range(1, n_samples + 1):
            coverage = i / n_samples
            # Risk = error rate on covered samples
            risk = sorted_errors[:i].mean().item()
            
            coverages.append(coverage)
            risks.append(risk)
        
        # Calculate AURC using trapezoidal rule
        aurc = auc(coverages, risks)
        
        return aurc, np.array(coverages), np.array(risks)

    def calculate_aurc_opt(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Calculate the *optimal* AURC achievable on this fixed set of predictions.

        This is the "perfect ranker" baseline used for Excess-AURC (E-AURC).
        It assumes we can sort samples by correctness (all correct first, then wrong),
        while keeping the model predictions themselves fixed.
        """
        predicted_classes = predictions.argmax(dim=1)
        errors = (predicted_classes != labels).float()  # 1=wrong, 0=correct

        # Perfect ranker: keep all correct first.
        sorted_errors = torch.sort(errors, descending=False).values
        n_samples = int(len(sorted_errors))
        if n_samples <= 0:
            return 0.0

        coverages = np.linspace(1.0 / n_samples, 1.0, n_samples)
        # cumulative mean error
        risks = torch.cumsum(sorted_errors, dim=0) / torch.arange(1, n_samples + 1, device=sorted_errors.device)
        risks = risks.detach().cpu().numpy()
        return float(auc(coverages, risks))

    def calculate_eaurc(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        uncertainty_scores: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """
        Excess AURC (E-AURC) = AURC - AURC_opt

        Returns:
          eaurc, aurc, aurc_opt
        """
        aurc, _cov, _risk = self.calculate_aurc(predictions, labels, uncertainty_scores)
        aurc_opt = self.calculate_aurc_opt(predictions, labels)
        return float(aurc - aurc_opt), float(aurc), float(aurc_opt)
    
    def calculate_selective_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        uncertainty_scores: torch.Tensor,
        rejection_rates: np.ndarray = np.linspace(0, 0.5, 11)
    ) -> Dict[str, np.ndarray]:
        """
        Calculate accuracy at different rejection rates.
        
        This creates the "Accuracy vs Rejection Rate" plot that shows
        how well uncertainty ranking improves accuracy.
        
        Args:
            predictions: Predicted class probabilities [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            uncertainty_scores: Uncertainty scores [batch_size]
            rejection_rates: Array of rejection rates to test (0 to 1)
        
        Returns:
            Dictionary with:
                - rejection_rates: Tested rejection rates
                - accuracies: Accuracy at each rejection rate
                - n_samples: Number of samples kept at each rate
        """
        predicted_classes = predictions.argmax(dim=1)
        correct = (predicted_classes == labels).float()
        
        # Sort by uncertainty (ascending)
        sorted_indices = torch.argsort(uncertainty_scores)
        sorted_correct = correct[sorted_indices]
        
        n_total = len(sorted_correct)
        accuracies = []
        n_samples_kept = []
        
        for rejection_rate in rejection_rates:
            # Keep (1 - rejection_rate) of samples
            n_keep = int(n_total * (1 - rejection_rate))
            
            if n_keep > 0:
                # Calculate accuracy on kept samples
                accuracy = sorted_correct[:n_keep].mean().item()
            else:
                accuracy = 0.0
            
            accuracies.append(accuracy)
            n_samples_kept.append(n_keep)
        
        return {
            'rejection_rates': rejection_rates,
            'accuracies': np.array(accuracies),
            'n_samples': np.array(n_samples_kept)
        }


def compare_uncertainty_rankings(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    surgical_scores: torch.Tensor,
    entropy_scores: torch.Tensor,
    n_bins: int = 15
) -> Dict[str, float]:
    """
    Compare Surgical Score vs Entropy-based ranking.
    
    This is the KEY comparison for Concept 1:
    "Show that Surgical Score outperforms Entropy-based ranking in AURC"
    
    Args:
        predictions: Predicted class probabilities [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        surgical_scores: Surgical Score uncertainty [batch_size]
        entropy_scores: Entropy-based uncertainty [batch_size]
        n_bins: Number of bins for ECE
    
    Returns:
        Dictionary with comparison metrics
    """
    metrics = StandardUQMetrics(n_bins=n_bins)
    
    # Calculate AURC for both methods
    aurc_surgical, _, _ = metrics.calculate_aurc(predictions, labels, surgical_scores)
    aurc_entropy, _, _ = metrics.calculate_aurc(predictions, labels, entropy_scores)
    
    # Calculate ECE (same for both, but included for completeness)
    ece = metrics.calculate_ece(predictions, labels)
    
    # Calculate NLL
    nll = metrics.calculate_nll(predictions, labels)
    
    return {
        'aurc_surgical': aurc_surgical,
        'aurc_entropy': aurc_entropy,
        'aurc_improvement': (aurc_entropy - aurc_surgical) / aurc_entropy * 100,  # % improvement
        'ece': ece,
        'nll': nll,
        'winner': 'Surgical Score' if aurc_surgical < aurc_entropy else 'Entropy'
    }


def batch_standard_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_passes: int = 50,
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """
    Calculate standard UQ metrics for entire dataset.
    
    Args:
        model: MC Dropout model
        dataloader: Data loader
        n_passes: Number of MC forward passes
        device: Device to run on
    
    Returns:
        Dictionary with all predictions and labels for metric calculation
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_mc_predictions = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            
            # MC Dropout forward passes
            mc_predictions = model.mc_forward(data, n_passes=n_passes)
            
            # Mean prediction
            mean_pred = mc_predictions.mean(dim=0)
            
            all_predictions.append(mean_pred.cpu())
            all_labels.append(target)
            all_mc_predictions.append(mc_predictions.cpu())
    
    return {
        'predictions': torch.cat(all_predictions),
        'labels': torch.cat(all_labels),
        'mc_predictions': torch.cat(all_mc_predictions, dim=1)  # Concatenate along batch dimension
    }


if __name__ == "__main__":
    # Test standard UQ metrics
    print("=" * 60)
    print("Testing Standard UQ Metrics")
    print("=" * 60)
    
    # Simulate predictions
    batch_size = 100
    num_classes = 10
    
    # Well-calibrated predictions
    predictions = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Simulate uncertainty scores
    surgical_scores = torch.rand(batch_size)  # Lower = more certain
    entropy_scores = torch.rand(batch_size)
    
    metrics = StandardUQMetrics(n_bins=10)
    
    # Test ECE
    ece = metrics.calculate_ece(predictions, labels)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print("  → Lower is better (well-calibrated model)")
    
    # Test NLL
    nll = metrics.calculate_nll(predictions, labels)
    print(f"\nNegative Log-Likelihood (NLL): {nll:.4f}")
    print("  → Lower is better (confident predictions)")
    
    # Test AURC
    aurc_surgical, coverages, risks = metrics.calculate_aurc(
        predictions, labels, surgical_scores
    )
    aurc_entropy, _, _ = metrics.calculate_aurc(
        predictions, labels, entropy_scores
    )
    
    print(f"\nArea Under Risk-Coverage (AURC):")
    print(f"  Surgical Score: {aurc_surgical:.4f}")
    print(f"  Entropy-based:  {aurc_entropy:.4f}")
    print(f"  → Lower is better (better uncertainty ranking)")
    
    if aurc_surgical < aurc_entropy:
        improvement = (aurc_entropy - aurc_surgical) / aurc_entropy * 100
        print(f"  ✓ Surgical Score wins by {improvement:.1f}%!")
    
    # Test selective accuracy
    selective_results = metrics.calculate_selective_accuracy(
        predictions, labels, surgical_scores
    )
    
    print(f"\nSelective Classification:")
    print(f"  Baseline accuracy: {selective_results['accuracies'][0]:.3f}")
    print(f"  At 20% rejection:  {selective_results['accuracies'][2]:.3f}")
    print(f"  At 50% rejection:  {selective_results['accuracies'][-1]:.3f}")
    print("  → Accuracy should increase as we reject uncertain samples")

# Made with Bob
