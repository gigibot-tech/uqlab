"""
Model architectures for uncertainty classification experiments.

Contains:
- EmbeddingDataset: Embedding-level dataset for DualXDA compatibility
- EmbeddingMLP: Small MLP classifier (legacy name EmbeddingDropoutMLP)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Embedding-level dataset compatible with DualXDA."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        clean_labels: torch.Tensor,
        is_noisy: torch.Tensor,
        original_indices: torch.Tensor,
    ):
        self.features = features.float()
        self.targets = labels.long()
        self.clean_labels = clean_labels.long()
        self.is_noisy = is_noisy.bool()
        self.original_indices = original_indices.long()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.targets[index]


class EmbeddingDropoutMLP(nn.Module):
    """Embedding-space classifier; optional dropout for MC predictive signals at eval."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout1(x)
        h = self.proj(h)
        h = self.relu(h)
        h = self.dropout2(h)
        return self.fc(h)

    def enable_dropout(self) -> None:
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
        from uqlab.models.mc_dropout import mc_forward_efficient

        return mc_forward_efficient(
            self, x, n_passes, sample_batch_size=sample_batch_size
        )


class EmbeddingMLP(EmbeddingDropoutMLP):
    """Canonical public name."""
