"""Naive baseline: concatenate 4 PLM embeddings + ParaSurf features → MLP.

Deliberately simple. No gating, no attention, no transformer, no chain-id.
This is the "you just concatenated" strawman the reviewer will throw at us.
"""
from __future__ import annotations

import torch
from torch import nn

from antisite.models.antisite import DEFAULT_PLMS, PLM_DIMS, SURFACE_FEAT_DIM


class NaiveConcat(nn.Module):
    """Concat(PLM₁, PLM₂, …, PLM_k, ParaSurf) → 3-layer MLP → logit."""

    def __init__(
        self,
        enabled_plms: list[str] | None = None,
        surface_feat_dim: int = SURFACE_FEAT_DIM,
        hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if enabled_plms is None:
            enabled_plms = DEFAULT_PLMS
        self.plm_names = list(enabled_plms)
        plm_total = sum(PLM_DIMS[n] for n in self.plm_names)
        self.in_dim = plm_total + surface_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        plm_embs: dict[str, torch.Tensor],
        chain_ids: torch.Tensor,           # accepted but unused
        valid_mask: torch.Tensor,          # accepted but unused
        surface_features: torch.Tensor,    # [B, L, 256]
    ) -> dict:
        ref_dtype = self.mlp[0].weight.dtype
        plm_cat = torch.cat(
            [plm_embs[n].to(ref_dtype) for n in self.plm_names], dim=-1
        )
        x = torch.cat([plm_cat, surface_features.to(ref_dtype)], dim=-1)
        logits = self.mlp(x).squeeze(-1)
        return {"logits_fused": logits, "logits_aux": logits}
