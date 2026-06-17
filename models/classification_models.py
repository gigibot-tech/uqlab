"""
Model architectures for uncertainty classification experiments.

Contains:
- EmbeddingDataset: Embedding-level dataset for DualXDA compatibility
- EmbeddingDropoutMLP: Small MLP classifier with MC Dropout support
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    Embedding-level dataset compatible with DualXDA.
    
    Stores pre-extracted DINOv2 embeddings (768-dim vectors) along with labels and metadata.
    The `targets` attribute contains the labels used during training
    (which may be noisy labels in CIFAR-10N experiments).
    
    Args:
        features: Pre-extracted DINOv2 embeddings [N, 768]
        labels: Training labels (may be noisy) [N]
        clean_labels: Ground truth clean labels [N]
        is_noisy: Boolean mask indicating noisy samples [N]
        original_indices: Original dataset indices [N]
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        clean_labels: torch.Tensor,
        is_noisy: torch.Tensor,
        original_indices: torch.Tensor,
    ):
        self.features = features.float()
        self.targets = labels.long()  # Used by DualXDA
        self.clean_labels = clean_labels.long()
        self.is_noisy = is_noisy.bool()
        self.original_indices = original_indices.long()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        """Return (embedding, label) tuple for training."""
        return self.features[index], self.targets[index]


class EmbeddingDropoutMLP(nn.Module):
    """
    Small embedding-space classifier with MC Dropout support.
    
    Architecture:
        Dropout -> Linear -> ReLU -> Dropout -> Linear
    
    The dropout layers enable Monte Carlo Dropout for uncertainty estimation.
    
    Args:
        input_dim: Dimension of input embeddings (768 for DINOv2)
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(
        self, 
        input_dim: int, 
        num_classes: int = 10, 
        hidden_dim: int = 256, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input embeddings [batch_size, 768]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        h = self.dropout1(x)
        h = self.proj(h)
        h = self.relu(h)
        h = self.dropout2(h)
        return self.fc(h)

    def enable_dropout(self) -> None:
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def mc_forward(
        self,
        x: torch.Tensor,
        n_passes: int = 20,
        *,
        sample_batch_size: int = 256,
    ) -> torch.Tensor:
        """
        Perform Monte Carlo Dropout forward passes (chunked for large eval sets).
        """
        from uqlab.mc_dropout_uq import mc_forward_efficient

        return mc_forward_efficient(
            self, x, n_passes, sample_batch_size=sample_batch_size
        )

# Made with Bob
