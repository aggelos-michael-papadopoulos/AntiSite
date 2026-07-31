"""Frozen ParaSurf extractor — exposes per-residue 256-d pooled surface features.

ParaSurf natively operates on surface points (one per heavy atom, produced by DMS). This
wrapper runs a forward pass over all surface points of an antibody and aggregates to the
residue level:

    residue score    = max over atoms' sigmoid(logit)      # matches ParaSurf Eq. 1
    residue feature  = mean over atoms' 256-d pre-classifier vectors

Features are captured via a forward-pre-hook on the classifier layer, so we do not modify
ParaSurf's model code. Always frozen (requires_grad=False, eval mode).

Run ParaSurf once per antibody, cache the (res_ids, scores, features) to disk, then train
against the cache.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make ParaSurf importable without installing it.
_PARASURF_ROOT = Path(__file__).resolve().parents[2] / "ParaSurf"
if str(_PARASURF_ROOT) not in sys.path:
    sys.path.insert(0, str(_PARASURF_ROOT))

from ParaSurf.model.ParaSurf_model import DilatedBottleneck, ResNet3D_Transformer  # noqa: E402
from ParaSurf.train.features import KalasantyFeaturizer  # noqa: E402
from ParaSurf.train.protein import Protein_pred  # noqa: E402


FEATURE_DIM = 256  # post-GAP, pre-classifier


@dataclass
class ParaSurfOutput:
    """Per-residue ParaSurf outputs for one antibody."""

    res_ids: list[str]          # "resnum_chain" (+ optional insertion code), PDB order of first appearance
    scores: torch.Tensor        # [N_residues] — max-aggregated sigmoid scores
    features: torch.Tensor      # [N_residues, 256] — mean-aggregated pre-classifier features


class ParaSurfExtractor(nn.Module):
    """Frozen ParaSurf wrapper.

    Loads the 3D ResNet + Transformer backbone, registers a pre-hook on the classifier to
    capture 256-d features, and runs inference batched over an antibody's surface points.
    """

    def __init__(
        self,
        weights_path: str | os.PathLike,
        device: str = "cuda",
        grid_size: int = 41,
        feature_channels: int = 22,
        voxel_size: int = 1,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.grid_size = grid_size
        self.feature_channels = feature_channels

        model = ResNet3D_Transformer(
            in_channels=feature_channels,
            block=DilatedBottleneck,
            num_blocks=[3, 4, 6, 3],
            num_classes=1,
        )
        state = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self.featurizer = KalasantyFeaturizer(grid_size, voxel_size)

        # Forward-pre-hook on classifier captures the 256-d vector entering it.
        # During eval, dropout is identity, so this is exactly the post-GAP feature.
        self._feature_buffer: list[torch.Tensor] = []
        self.model.classifier.register_forward_pre_hook(self._capture_features)

        # Populated by compute() — kept around so callers (e.g. infer_3d) can run
        # ParaSurf's binding-site extractor on the same Protein_pred instance.
        self.last_prot = None
        self.last_surf_file: Path | None = None

    def _capture_features(self, _module, inputs):
        self._feature_buffer.append(inputs[0].detach().cpu())

    @torch.no_grad()
    def compute(
        self,
        pdb_path: str | os.PathLike,
        batch_size: int = 64,
        add_forcefields: bool = True,
        add_atom_radius_features: bool = True,
    ) -> ParaSurfOutput:
        """Run ParaSurf on one antibody PDB and return per-residue outputs.

        Expects the PDB to already be cleaned (water/ions removed) as ParaSurf expects.
        Creates a sibling directory for DMS surface files; leaves them on disk so repeat
        calls skip the DMS step.
        """
        pdb_path = Path(pdb_path)

        # DMS surface point generation + featurizer setup — reuse ParaSurf's pipeline.
        prot = Protein_pred(str(pdb_path), save_path=str(pdb_path.parent))
        self.featurizer.get_channels(prot.mol, add_forcefields, add_atom_radius_features)

        # Map surf-point index -> is-atom-type (matches ParaSurf's blind_predict logic).
        surf_file = next(p for p in Path(prot.save_path).iterdir() if "surfpoints" in p.name)
        atom_type_mask: list[bool] = []
        with surf_file.open() as f:
            for line in f:
                parts = line.split()
                atom_type_mask.append(len(parts) > 6 and parts[6] == "A")

        atom_mask_np = np.asarray(atom_type_mask, dtype=bool)
        if atom_mask_np.shape[0] != len(prot.surf_points):
            raise RuntimeError(
                f"Surface-point count mismatch: file has {atom_mask_np.shape[0]} rows, "
                f"but ParaSurf loaded {len(prot.surf_points)} points."
            )

        # AntiSite aggregates only atom-type surface points (one per heavy atom).
        # Reentrant/contact points were previously run through the expensive voxel
        # CNN and then discarded below. Select the retained points before feature
        # construction instead; samples are independent in eval mode, so their
        # scores and 256-d features are unchanged.
        atom_point_indices = np.flatnonzero(atom_mask_np)
        print(
            f"ParaSurf: evaluating {len(atom_point_indices)}/{len(prot.surf_points)} "
            f"atom-type surface points on {self.device} (batch={batch_size})"
        )

        # Forward in batches; the pre-hook captures features in lockstep with scores.
        self._feature_buffer.clear()
        scores_list: list[np.ndarray] = []
        input_data = torch.zeros(
            (batch_size, self.grid_size, self.grid_size, self.grid_size, self.feature_channels),
            device=self.device,
        )
        n_points = len(atom_point_indices)
        batch_cnt = 0
        for point_idx in atom_point_indices:
            p = prot.surf_points[point_idx]
            n = prot.surf_normals[point_idx]
            input_data[batch_cnt] = torch.tensor(
                self.featurizer.grid_feats(p, n, prot.heavy_atom_coords),
                device=self.device,
            )
            batch_cnt += 1
            if batch_cnt == batch_size:
                logits = self.model(input_data)
                scores_list.append(torch.sigmoid(logits).cpu().numpy())
                batch_cnt = 0
        if batch_cnt > 0:
            logits = self.model(input_data[:batch_cnt])
            scores_list.append(torch.sigmoid(logits).cpu().numpy())

        scores_all = np.concatenate(scores_list, axis=0).reshape(-1)  # [n_points]
        features_all = torch.cat(self._feature_buffer, dim=0)          # [n_points, 256]
        assert scores_all.shape[0] == n_points == features_all.shape[0], (
            f"Shape mismatch: scores={scores_all.shape}, features={features_all.shape}, n_points={n_points}"
        )

        # Every computed sample is now an atom-type point, in the original order.
        scores_atoms = scores_all                         # [n_heavy_atoms]
        features_atoms = features_all                     # [n_heavy_atoms, 256]

        # Map each atom to its residue. Keys follow ParaSurf's convention: "resnum_chain"
        # (plus "_insertion" when present). Order = first appearance in the PDB.
        res_id_per_atom: list[str] = []
        with pdb_path.open() as f:
            for line in f:
                if line.startswith("ATOM") and line.split()[2][0] != "H":
                    chain_id = line[21]
                    resnum = line[22:26].strip()
                    insertion = line[26].strip()
                    rid = f"{resnum}_{chain_id}"
                    if insertion:
                        rid = f"{rid}_{insertion}"
                    res_id_per_atom.append(rid)

        if len(res_id_per_atom) != scores_atoms.shape[0]:
            raise RuntimeError(
                f"Heavy-atom count mismatch: PDB has {len(res_id_per_atom)} heavy atoms, "
                f"surface file has {scores_atoms.shape[0]} atom-type points."
            )

        # Aggregate per residue (preserve first-appearance order).
        atom_idx_per_res: dict[str, list[int]] = {}
        ordered_res_ids: list[str] = []
        for i, rid in enumerate(res_id_per_atom):
            if rid not in atom_idx_per_res:
                atom_idx_per_res[rid] = []
                ordered_res_ids.append(rid)
            atom_idx_per_res[rid].append(i)

        n_res = len(ordered_res_ids)
        res_scores = torch.zeros(n_res)
        res_features = torch.zeros(n_res, FEATURE_DIM)
        for i, rid in enumerate(ordered_res_ids):
            idxs = atom_idx_per_res[rid]
            res_scores[i] = float(scores_atoms[idxs].max())
            res_features[i] = features_atoms[idxs].mean(dim=0)

        # Stash for downstream pocket extraction.
        self.last_prot = prot
        self.last_surf_file = Path(surf_file)

        return ParaSurfOutput(
            res_ids=ordered_res_ids,
            scores=res_scores,
            features=res_features,
        )
