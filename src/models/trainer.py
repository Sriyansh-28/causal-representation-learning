"""Training loop for :class:`RepresentationNet`.

Only the factual outcome is observed, so the loss is the MSE between the
prediction of the observed-treatment head and the observed outcome. No
counterfactual information enters training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .representation import RepresentationNet


@dataclass
class TrainConfig:
    """Hyper-parameters for neural training.

    Attributes:
        hidden_dims: Encoder hidden layer widths.
        latent_dim: Representation dimension.
        head_hidden_dims: Head hidden layer widths.
        dropout: Dropout probability (part of "regularization" for ablation D).
        weight_decay: L2 penalty (part of "regularization" for ablation D).
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        epochs: Maximum training epochs.
        val_fraction: Fraction of training data held out for early stopping.
        early_stopping: Whether to restore the best-validation weights.
        patience: Epochs without validation improvement before stopping.
        use_representation: Ablation flag (see RepresentationNet).
        treatment_specific_heads: Ablation flag (see RepresentationNet).
        seed: Seed for initialisation and batch shuffling.
    """

    hidden_dims: Sequence[int] = (64, 64)
    latent_dim: int = 32
    head_hidden_dims: Sequence[int] = (32,)
    dropout: float = 0.1
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 200
    val_fraction: float = 0.2
    early_stopping: bool = True
    patience: int = 30
    use_representation: bool = True
    treatment_specific_heads: bool = True
    seed: int = 0
    history: Dict[str, List[float]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")


def train_model(
    x: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    cfg: TrainConfig,
) -> tuple[RepresentationNet, Dict[str, List[float]]]:
    """Fit a :class:`RepresentationNet` on factual data.

    Returns:
        The trained model (in eval mode) and the loss history.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()

    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(cfg.val_fraction * n)) if cfg.early_stopping else 0
    # Guard against a validation split that leaves an arm empty.
    if n_val > 0 and (n - n_val) < 8:
        n_val = 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    xt = torch.from_numpy(x[train_idx])
    tt = torch.from_numpy(t[train_idx])
    yt = torch.from_numpy(y[train_idx])
    loader = DataLoader(
        TensorDataset(xt, tt, yt),
        batch_size=min(cfg.batch_size, xt.shape[0]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        drop_last=False,
    )

    model = RepresentationNet(
        input_dim=x.shape[1],
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.latent_dim,
        head_hidden_dims=cfg.head_hidden_dims,
        dropout=cfg.dropout,
        use_representation=cfg.use_representation,
        treatment_specific_heads=cfg.treatment_specific_heads,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    has_val = n_val > 0
    if has_val:
        xv = torch.from_numpy(x[val_idx])
        tv = torch.from_numpy(t[val_idx])
        yv = torch.from_numpy(y[val_idx])

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0

    for _ in range(cfg.epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for xb, tb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model.predict_factual(xb, tb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]
        history["train_loss"].append(epoch_loss / max(n_seen, 1))

        if has_val:
            model.eval()
            with torch.no_grad():
                v = float(loss_fn(model.predict_factual(xv, tv), yv).item())
            history["val_loss"].append(v)
            if v < best_val - 1e-6:
                best_val = v
                best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
