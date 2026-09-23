"""Focal loss for class-imbalanced classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

REDUCTIONS = ("none", "mean", "sum")


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) with optional per-class alpha weights.

    gamma down-weights easy examples; alpha balances classes. With gamma=0 and
    alpha=None the loss equals standard cross entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D (batch, classes), got shape {tuple(logits.shape)}")
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D (batch,), got shape {tuple(targets.shape)}")
        if logits.size(0) != targets.size(0):
            raise ValueError(
                f"Batch size mismatch: logits {logits.size(0)} vs targets {targets.size(0)}"
            )
        if targets.dtype != torch.long:
            targets = targets.long()
        num_classes = logits.size(1)
        if targets.numel() > 0:
            if targets.min() < 0 or targets.max() >= num_classes:
                raise ValueError(
                    f"Target values must be in [0, {num_classes - 1}], "
                    f"got range [{targets.min().item()}, {targets.max().item()}]"
                )

        log_pt = F.log_softmax(logits, dim=1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(-100.0)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt

        alpha = self.alpha
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.as_tensor(alpha, dtype=loss.dtype, device=logits.device)
            if isinstance(alpha, torch.Tensor):
                if alpha.numel() != num_classes:
                    raise ValueError(
                        f"alpha has {alpha.numel()} entries but there are {num_classes} classes"
                    )
                alpha = alpha.to(device=logits.device, dtype=loss.dtype)
                loss = loss * alpha[targets]
            else:
                if num_classes != 2:
                    raise ValueError("Scalar alpha is only supported for binary classification")
                if not 0.0 <= float(alpha) <= 1.0:
                    raise ValueError(f"alpha must be in [0, 1], got {alpha}")
                weight = float(alpha) * targets + (1.0 - float(alpha)) * (1 - targets)
                loss = loss * weight.to(dtype=loss.dtype)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
