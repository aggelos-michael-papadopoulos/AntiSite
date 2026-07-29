"""AntiSite training loss.

Two binary cross-entropy terms, each applied only where the target is defined:

    L_aux    : BCE on the auxiliary head's logits vs labels   (all valid residues)
    L_fused  : BCE on the fused head's logits vs labels        (valid & surface_mask)

Weighted sum: L_aux + lambda_fused * L_fused.

The auxiliary term provides deep supervision to the sequence encoder; the fused
head is the prediction head at inference. There is no distillation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossConfig:
    lambda_fused: float = 1.0


def _bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean BCE over masked positions. Returns 0 scalar if mask is empty."""
    if mask.sum() == 0:
        return logits.sum() * 0.0  # keep grad graph
    per = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return per[mask].mean()


def antisite_loss(
    out: dict,
    batch: dict,
    cfg: LossConfig = LossConfig(),
) -> tuple[torch.Tensor, dict]:
    """Compute total loss and return (total, components_dict) for logging.

    ``out`` is AntiSite.forward()'s return; ``batch`` is collate_fn()'s return.
    Expected keys:
      out:   logits_aux [B,L], logits_fused [B,L] (optional)
      batch: labels [B,L], valid_mask [B,L], surface_mask [B,L]
    """
    valid  = batch["valid_mask"]
    s_mask = batch["surface_mask"] & valid
    labels = batch["labels"]

    L_aux = _bce(out["logits_aux"], labels, valid)

    L_fused = torch.tensor(0.0, device=L_aux.device)
    if "logits_fused" in out:
        L_fused = _bce(out["logits_fused"], labels, s_mask)

    total = L_aux + cfg.lambda_fused * L_fused
    return total, {
        "L_total": float(total.detach()),
        "L_aux":   float(L_aux.detach()),
        "L_fused": float(L_fused.detach()),
    }
