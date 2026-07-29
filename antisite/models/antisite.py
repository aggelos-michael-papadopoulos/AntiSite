"""AntiSite model: PLM fusion + chain-aware transformer + cross-modal fused head.

Architecture:

    frozen PLM embeddings  ──▶  per-PLM Linear (d_i ─▶ d_model)
      (ProtT5, ESM-2)               │
                                    ▼
                         learned position-wise gating
                         (softmax over the PLMs) + weighted sum
                                    │
                                    ▼
              + chain_id embedding + sinusoidal position
                                    │
                                    ▼
                  N × TransformerEncoderLayer (pre-norm)
                                    │
                                    ▼
                             z_seq [B, L, d_model]
                              │            │
                              ▼            ▼
                        aux head    cross-modal attention
                        (auxiliary   (z_seq queries surface_features)
                         deep-             │
                         supervision   concat(z_seq', surface_features)
                         BCE)              │
                                          ▼
                                     fused head
                                          │
                                          ▼
                                     logits_fused

The fused head is the prediction head in both inference modes: it receives real
ParaSurf ``surface_features`` in structure-aware mode and a zeroed placeholder in
sequence-only mode (see modality dropout). The ``aux head`` is an auxiliary
deep-supervision head used only during training; it does not affect inference.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


PLM_DIMS: dict[str, int] = {
    "ablang2":   480,
    "antiberty": 512,
    "esm2":      1280,
    "prot_t5":   1024,
    "igt5":      1024,
    "igbert":    1024,
}

# Default PLM stack — the two general protein language models selected by the
# PLM-stack ablation (see the Supplementary Information). Across three benchmarks
# and five seeds, ProtT5 + ESM-2 match or exceed larger stacks that also include
# antibody-specific models. Order matters — checkpoints store this list so the
# gate weights align at load time.
DEFAULT_PLMS: list[str] = ["prot_t5", "esm2"]

SURFACE_FEAT_DIM = 256


def _sinusoidal_positions(L: int, d: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal position encoding, shape [L, d]."""
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)
    angles = pos / torch.pow(10000.0, (2 * (i // 2)) / d)
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class PLMFusion(nn.Module):
    """Project each PLM to d_model, then fuse with a position-wise learned gate."""

    def __init__(self, d_model: int = 256, plm_dims: dict[str, int] = PLM_DIMS,
                 enabled_plms: list[str] | None = None):
        super().__init__()
        if enabled_plms is None:
            enabled_plms = DEFAULT_PLMS
        plm_dims = {k: plm_dims[k] for k in enabled_plms}
        self.plm_names = list(plm_dims.keys())
        self.projs = nn.ModuleDict({
            name: nn.Linear(dim, d_model) for name, dim in plm_dims.items()
        })
        # Gate: concat of all projected embeddings → scalar per PLM per position.
        self.gate = nn.Sequential(
            nn.Linear(d_model * len(plm_dims), d_model),
            nn.GELU(),
            nn.Linear(d_model, len(plm_dims)),
        )

    def forward(self, plm_embs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """plm_embs[name]: [B, L, d_name]. Returns (fused [B, L, d_model], gate_weights [B, L, P])."""
        # Upcast to match projection dtype (embeddings may be stored fp16 on disk).
        ref_dtype = next(self.projs[self.plm_names[0]].parameters()).dtype
        projected = [
            self.projs[name](plm_embs[name].to(ref_dtype)) for name in self.plm_names
        ]  # each [B,L,d]
        stacked   = torch.stack(projected, dim=2)                                   # [B,L,P,d]
        gate_in   = torch.cat(projected, dim=-1)                                    # [B,L,P*d]
        gate_w    = F.softmax(self.gate(gate_in), dim=-1)                           # [B,L,P]
        fused     = (stacked * gate_w.unsqueeze(-1)).sum(dim=2)                     # [B,L,d]
        return fused, gate_w


class CrossModalAttention(nn.Module):
    """One-way cross-attention: z_seq queries the surface features, with residual.

    In seq-only mode surface_features = zeros. This is *not* a literal no-op
    (the LayerNorm on the K/V stream maps zeros to its bias, not to zero), so the
    layer still applies a fixed transform to z_seq. But a constant zero input
    carries no information, so no structural signal enters; the seq-only path
    works because modality dropout trains the same weights to predict well with
    the surface-feature slot both empty and full.
    """

    def __init__(self, d: int = 256, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm_q":   nn.LayerNorm(d),
                "norm_kv":  nn.LayerNorm(d),
                "attn":     nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "norm_ff":  nn.LayerNorm(d),
                "ff":       nn.Sequential(nn.Linear(d, d * 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(d * 2, d)),
            })
            for _ in range(n_layers)
        ])

    def forward(self, z_seq: torch.Tensor, surface_features: torch.Tensor,
                valid_mask: torch.Tensor) -> torch.Tensor:
        kpm = ~valid_mask  # MultiheadAttention: True = ignore
        for layer in self.layers:
            q = layer["norm_q"](z_seq)
            kv = layer["norm_kv"](surface_features)
            attn_out, _ = layer["attn"](q, kv, kv, key_padding_mask=kpm, need_weights=False)
            z_seq = z_seq + attn_out
            z_seq = z_seq + layer["ff"](layer["norm_ff"](z_seq))
        return z_seq


class AntiSite(nn.Module):
    """Full AntiSite model: sequence encoder + cross-modal fused head.

    Also carries an auxiliary deep-supervision head (``aux_head``) used only during
    training; inference reads ``logits_fused``.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        surface_feat_dim: int = SURFACE_FEAT_DIM,
        enabled_plms: list[str] | None = None,
        cross_modal_layers: int = 0,    # 0 disables cross-attn
        cross_modal_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model

        self.fusion = PLMFusion(d_model=d_model, enabled_plms=enabled_plms)
        self.chain_emb = nn.Embedding(2, d_model)  # 0=heavy, 1=light

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

        # Auxiliary head: z → paratope logit (deep supervision during training only).
        self.aux_head = nn.Linear(d_model, 1)

        # Optional cross-modal attention BEFORE concat (z_seq queries surface features).
        self.cross_modal = (
            CrossModalAttention(d=d_model, n_heads=cross_modal_heads,
                                n_layers=cross_modal_layers, dropout=dropout)
            if cross_modal_layers > 0 else None
        )

        # Fused head: concat(z_seq, surface_features) → logit.
        self.fused_head = nn.Sequential(
            nn.Linear(d_model + surface_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(
        self,
        plm_embs: dict[str, torch.Tensor],
        chain_ids: torch.Tensor,      # [B, L] long
        valid_mask: torch.Tensor,     # [B, L] bool, True on real residues
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_seq [B, L, d], gate_weights [B, L, P])."""
        fused, gate_w = self.fusion(plm_embs)
        B, L, d = fused.shape

        pos = _sinusoidal_positions(L, d, fused.device).unsqueeze(0)  # [1, L, d]
        x = fused + self.chain_emb(chain_ids) + pos

        # TransformerEncoder expects src_key_padding_mask where True = ignore.
        z = self.encoder(x, src_key_padding_mask=~valid_mask)
        z = self.norm_out(z)
        return z, gate_w

    def forward(
        self,
        plm_embs: dict[str, torch.Tensor],
        chain_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        surface_features: torch.Tensor | None = None,  # [B, L, 256]
    ) -> dict:
        """Run the model. Returns a dict of outputs.

        If ``surface_features`` is given, the fused head is run (the prediction head
        in both modes: real features for structure-aware, zeros for sequence-only).
        ``logits_aux`` is an auxiliary training signal and is not used at inference.
        """
        z, gate_w = self.encode(plm_embs, chain_ids, valid_mask)

        out: dict = {
            "logits_aux":    self.aux_head(z).squeeze(-1),  # [B, L] (training only)
            "gate_weights":  gate_w,                        # [B, L, P]
        }
        if surface_features is not None:
            z_for_fuse = z
            if self.cross_modal is not None:
                z_for_fuse = self.cross_modal(z, surface_features, valid_mask)
            fused_in = torch.cat([z_for_fuse, surface_features], dim=-1)   # [B, L, d+256]
            out["logits_fused"] = self.fused_head(fused_in).squeeze(-1)
        return out
