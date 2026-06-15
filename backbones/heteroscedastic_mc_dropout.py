"""
Heteroscedastic (aleatoric) + MC Dropout (epistemic) model.

This implements a practical version of Kendall & Gal-style aleatoric uncertainty
for classification by predicting a per-sample log-variance and marginalizing the
softmax likelihood via Monte Carlo sampling in logit space.

We keep the API close to MCDropoutResNet:
  - forward(x, enable_dropout=...): returns (logits, log_var)
  - forward_logits(x, enable_dropout=...): returns logits only
  - mc_forward(x, n_passes): returns stacked probabilities [T,B,K]
  - predict_aleatoric_var(x): returns predicted variance proxy [B]
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HeteroscedasticMCDropoutResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.3,
        pretrained: bool = False,
        log_var_init: float = -2.0,
    ):
        super().__init__()

        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout_rate)
        # Keep logits head name as "fc" so DualDA scripts can hook layer_name="fc".
        self.fc = nn.Linear(num_features, num_classes)
        self.fc_logvar = nn.Linear(num_features, 1)

        # Initialize log-variance head towards small variance.
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, log_var_init)

        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)

    def _maybe_dropout(self, features: torch.Tensor, enable_dropout: bool) -> torch.Tensor:
        if enable_dropout:
            self.dropout.train()
            return self.dropout(features)
        if self.training:
            return self.dropout(features)
        return features

    def forward(self, x: torch.Tensor, enable_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        features = self._maybe_dropout(features, enable_dropout=enable_dropout)
        logits = self.fc(features)
        log_var = self.fc_logvar(features).squeeze(1)  # [B]
        return logits, log_var

    def forward_logits(self, x: torch.Tensor, enable_dropout: bool = True) -> torch.Tensor:
        logits, _log_var = self.forward(x, enable_dropout=enable_dropout)
        return logits

    def mc_forward(self, x: torch.Tensor, n_passes: int = 50) -> torch.Tensor:
        self.eval()
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = self.forward_logits(x, enable_dropout=True)
                preds.append(F.softmax(logits, dim=1))
        return torch.stack(preds, dim=0)  # [T,B,K]

    @torch.no_grad()
    def predict_aleatoric_var(self, x: torch.Tensor, enable_dropout: bool = False) -> torch.Tensor:
        """
        Returns a non-negative scalar variance proxy per sample.
        """
        _logits, log_var = self.forward(x, enable_dropout=enable_dropout)
        return torch.exp(log_var).clamp_min(0.0)


def heteroscedastic_classification_nll(
    logits: torch.Tensor,
    log_var: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 10,
) -> torch.Tensor:
    """
    Approximate NLL for classification with logit noise:
      p(y|x) = E_{eps ~ N(0, I)}[ softmax(logits + sqrt(var)*eps )_y ]

    Loss:
      -log( 1/S sum_s softmax(noisy_logits_s)_y )
    computed stably via logsumexp.

    Args:
      logits: [B,K]
      log_var: [B] (scalar variance per sample)
      y: [B]
      n_samples: number of MC samples for marginalization
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,K], got {tuple(logits.shape)}")
    if log_var.dim() != 1:
        raise ValueError(f"log_var must be [B], got {tuple(log_var.shape)}")
    if y.dim() != 1:
        raise ValueError(f"y must be [B], got {tuple(y.shape)}")

    b, k = logits.shape
    s = int(n_samples)
    if s <= 1:
        # Fallback: standard CE on logits.
        return F.cross_entropy(logits, y)

    var = torch.exp(log_var).clamp_min(0.0)  # [B]
    std = torch.sqrt(var + 1e-12)  # [B]

    eps = torch.randn(s, b, k, device=logits.device, dtype=logits.dtype)
    noisy_logits = logits.unsqueeze(0) + eps * std.view(1, b, 1)

    # True class log-prob per sample
    log_probs = F.log_softmax(noisy_logits, dim=-1)  # [S,B,K]
    y_idx = y.view(1, b, 1).expand(s, b, 1)
    true_logp = log_probs.gather(-1, y_idx).squeeze(-1)  # [S,B]

    log_mean = torch.logsumexp(true_logp, dim=0) - math.log(s)  # [B]
    return (-log_mean).mean()

