"""PyTorch Dataset + collate for AntiSite training.

Reads per-antibody example bundles produced by ``antisite.data.build_dataset`` and
returns tensors aligned residue-by-residue. The collate function pads variable-length
examples to the batch maximum and emits a ``valid_mask`` that downstream losses must
apply so padded positions contribute nothing.

Each example covers one antibody Fv (heavy + light concatenated). Positions 0..L_H-1
are heavy residues; L_H..L are light. Both PLM tokenization and chain-aware position
embeddings use this ordering.

Usage:
    from antisite.data.dataset import AntisiteDataset, collate_fn

    ds = AntisiteDataset("examples/PECAN/TRAIN")
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn, shuffle=True)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class AntisiteDataset(Dataset):
    """Loads per-antibody .pt example bundles. Pure disk read — no PDB parsing.

    If ``embeddings_dir`` is provided, each example is also joined with its 6-PLM
    embedding bundle (keyed by pdb_id). Embeddings are stored as a dict under
    ``example["plm_embs"]`` with keys matching ``PLM_DIMS``.
    """

    def __init__(
        self,
        examples_dir: str | Path,
        embeddings_dir: str | Path | None = None,
    ) -> None:
        self.examples_dir = Path(examples_dir)
        if not self.examples_dir.is_dir():
            raise FileNotFoundError(f"Examples dir not found: {self.examples_dir}")
        self.paths = sorted(self.examples_dir.glob("*.pt"))
        if not self.paths:
            raise RuntimeError(f"No .pt examples in {self.examples_dir}")
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        if self.embeddings_dir and not self.embeddings_dir.is_dir():
            raise FileNotFoundError(f"Embeddings dir not found: {self.embeddings_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        ex = torch.load(self.paths[idx], weights_only=False)
        if self.embeddings_dir:
            emb_path = self.embeddings_dir / f"{ex['pdb_id']}.pt"
            ex["plm_embs"] = torch.load(emb_path, weights_only=True)
        return ex


def _pad_1d(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    max_len = max(t.shape[0] for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def _pad_2d(tensors: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Pad [L_i, D] tensors to a single [B, L_max, D] tensor."""
    max_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]
    out = torch.full((len(tensors), max_len, dim), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def collate_fn(batch: list[dict]) -> dict:
    """Pad a batch of examples to the max residue count.

    Returns:
        heavy_seqs:       list[str]           length B
        light_seqs:       list[str]           length B
        chain_ids:        LongTensor[B, L]    0=heavy, 1=light, 0 in padding
        labels:           FloatTensor[B, L]   0.0 in padding
        surface_features: FloatTensor[B, L, 256]   ParaSurf per-residue features
        surface_mask:     BoolTensor[B, L]    False where surface absent OR padded
        valid_mask:       BoolTensor[B, L]    True on real residues, False on padding
        res_ids:          list[list[str]]     B lists, per-residue keys (lossy across batch)
        pdb_ids:          list[str]
        lengths:          LongTensor[B]       real L per example
    """
    pdb_ids   = [ex["pdb_id"] for ex in batch]
    heavy     = [ex["heavy_seq"] for ex in batch]
    light     = [ex["light_seq"] for ex in batch]
    res_ids   = [ex["res_ids"] for ex in batch]
    lengths   = torch.tensor([len(ex["res_ids"]) for ex in batch], dtype=torch.long)

    max_len = int(lengths.max().item())

    valid_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, L in enumerate(lengths.tolist()):
        valid_mask[i, :L] = True

    # ParaSurf surface features. New caches store "surface_*"; older caches store
    # "teacher_*" — read either so both load.
    def _sfeat(ex):
        return ex["surface_features"] if "surface_features" in ex else ex["teacher_features"]
    def _smask(ex):
        return ex["surface_mask"] if "surface_mask" in ex else ex["teacher_mask"]

    chain_ids = _pad_1d([ex["chain_ids"] for ex in batch], pad_value=0)
    labels    = _pad_1d([ex["labels"]    for ex in batch], pad_value=0.0)
    features  = _pad_2d([_sfeat(ex) for ex in batch], pad_value=0.0)

    surface_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, ex in enumerate(batch):
        sm = _smask(ex)
        surface_mask[i, : sm.shape[0]] = sm

    plm_embs: dict[str, torch.Tensor] = {}
    if "plm_embs" in batch[0]:
        for name in batch[0]["plm_embs"]:
            plm_embs[name] = _pad_2d([ex["plm_embs"][name] for ex in batch], pad_value=0.0)

    return {
        "pdb_ids":          pdb_ids,
        "heavy_seqs":       heavy,
        "light_seqs":       light,
        "chain_ids":        chain_ids,
        "labels":           labels,
        "surface_features": features,
        "surface_mask":     surface_mask,
        "valid_mask":       valid_mask,
        "res_ids":          res_ids,
        "lengths":          lengths,
        "plm_embs":         plm_embs,
    }
