"""Neural representation model for treatment-effect estimation.

Architecture (TARNet-style, deliberately small and legible)::

    X --[ encoder phi ]--> Z --+--> head_0 --> y_hat_0(x)
                               |
                               +--> head_1 --> y_hat_1(x)

    tau_hat(x) = y_hat_1(x) - y_hat_0(x)

Only the head matching a unit's observed treatment is trained on that unit's
factual outcome; the counterfactual head is exercised only at prediction time.
The shared encoder is what lets control units inform the treated head and vice
versa, which is the mechanism this project is testing.

Two boolean flags implement the ablations without changing anything else:

* ``use_representation=False`` -- the encoder becomes the identity, so heads
  read raw covariates. Isolates the contribution of the learned representation.
* ``treatment_specific_heads=False`` -- a single head consumes ``[phi(x), t]``,
  i.e. a neural S-learner. Isolates the contribution of separate heads.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float) -> nn.Sequential:
    """Build a ReLU MLP with dropout after each hidden layer."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class RepresentationNet(nn.Module):
    """Shared-representation network with treatment-specific outcome heads.

    Args:
        input_dim: Number of covariates.
        hidden_dims: Widths of the encoder's hidden layers.
        latent_dim: Dimension of the learned representation ``Z``.
        head_hidden_dims: Widths of each head's hidden layers.
        dropout: Dropout probability applied in encoder and heads.
        use_representation: If False, skip the encoder (ablation B).
        treatment_specific_heads: If False, use one head on ``[Z, t]`` (ablation C).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        latent_dim: int = 32,
        head_hidden_dims: Sequence[int] = (32,),
        dropout: float = 0.1,
        use_representation: bool = True,
        treatment_specific_heads: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must lie in [0, 1), got {dropout}")

        self.input_dim = input_dim
        self.use_representation = use_representation
        self.treatment_specific_heads = treatment_specific_heads

        if use_representation:
            self.encoder = _mlp(input_dim, hidden_dims, latent_dim, dropout)
            self.latent_dim = latent_dim
        else:
            self.encoder = nn.Identity()
            self.latent_dim = input_dim

        if treatment_specific_heads:
            self.head0 = _mlp(self.latent_dim, head_hidden_dims, 1, dropout)
            self.head1 = _mlp(self.latent_dim, head_hidden_dims, 1, dropout)
            self.shared_head = None
        else:
            self.head0 = self.head1 = None
            self.shared_head = _mlp(self.latent_dim + 1, head_hidden_dims, 1, dropout)

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """Map covariates to the latent representation ``Z``."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return predicted potential outcomes ``(y_hat_0, y_hat_1)``.

        Each has shape ``(n,)``.
        """
        if x.dim() != 2:
            raise ValueError(f"expected 2-D input (n, d), got shape {tuple(x.shape)}")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} covariates, got {x.shape[1]}"
            )
        z = self.represent(x)
        if self.treatment_specific_heads:
            y0 = self.head0(z).squeeze(-1)
            y1 = self.head1(z).squeeze(-1)
        else:
            zeros = torch.zeros(z.shape[0], 1, dtype=z.dtype, device=z.device)
            ones = torch.ones_like(zeros)
            y0 = self.shared_head(torch.cat([z, zeros], dim=1)).squeeze(-1)
            y1 = self.shared_head(torch.cat([z, ones], dim=1)).squeeze(-1)
        return y0, y1

    def predict_factual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the outcome under each unit's *observed* treatment."""
        y0, y1 = self.forward(x)
        return torch.where(t > 0.5, y1, y0)

    def predict_tau(self, x: torch.Tensor) -> torch.Tensor:
        """Predict individual treatment effects ``y_hat_1 - y_hat_0``."""
        y0, y1 = self.forward(x)
        return y1 - y0
