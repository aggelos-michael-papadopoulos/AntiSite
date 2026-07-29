"""Evaluate a trained AntiSite checkpoint on a test split (aux + fused heads).

Usage:
    python -m antisite.eval.eval_antisite \
        --ckpt          runs/PECAN/best.pt \
        --test-examples examples/PECAN/TEST \
        --test-embeddings embeddings/PECAN/TEST \
        --dataset PECAN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.dataset import AntisiteDataset, collate_fn
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics, print_table
from antisite.models.antisite import PLM_DIMS, AntiSite


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(
    ckpt: Path, test_ex: Path, test_emb: Path,
    dataset_name: str = "", batch_size: int = 8, num_workers: int = 2,
    device: str | None = None,
    enabled_plms: list[str] | None = None,
) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = AntisiteDataset(test_ex, test_emb)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    blob = torch.load(ckpt, map_location=dev, weights_only=True)
    sd = blob["state_dict"]
    if enabled_plms is None:
        # Prefer the order saved in the checkpoint; fall back to detection.
        enabled_plms = blob.get("enabled_plms")
        if enabled_plms is None:
            present = [n for n in PLM_DIMS if f"fusion.projs.{n}.weight" in sd]
            enabled_plms = present
    # Detect cross-modal config from blob, falling back to state_dict inspection.
    cm_layers = blob.get("cross_modal_layers")
    if cm_layers is None:
        cm_layers = 1 + max(
            (int(k.split(".")[2]) for k in sd if k.startswith("cross_modal.layers.")),
            default=-1,
        )
    model = AntiSite(
        enabled_plms=enabled_plms,
        cross_modal_layers=cm_layers,
        cross_modal_heads=blob.get("cross_modal_heads", 4),
    ).to(dev)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded {ckpt} (epoch {blob.get('epoch','?')}, val PR={blob.get('val_fused_pr', float('nan')):.3f})")

    stu_metrics: list[dict] = []
    fus_metrics: list[dict] = []

    for batch in tqdm(loader, desc=f"eval {dataset_name}"):
        batch = _to_device(batch, dev)
        out = model(
            plm_embs=batch["plm_embs"],
            chain_ids=batch["chain_ids"],
            valid_mask=batch["valid_mask"],
            surface_features=batch["surface_features"],
        )
        s_probs = out["logits_aux"].sigmoid().float().cpu()
        f_probs = out["logits_fused"].sigmoid().float().cpu()
        labels  = batch["labels"].cpu()
        valid   = batch["valid_mask"].cpu()
        for i in range(labels.shape[0]):
            m = valid[i]
            y = labels[i][m].tolist()
            stu_metrics.append(per_protein_metrics(s_probs[i][m].tolist(), y))
            fus_metrics.append(per_protein_metrics(f_probs[i][m].tolist(), y))

    stu_agg = aggregate_metrics(stu_metrics)
    fus_agg = aggregate_metrics(fus_metrics)
    # NOTE: the aux head (formerly "aux") is a vestigial seq-only head from the v1
    # dual-head/KD architecture, kept harmlessly trained alongside the released model.
    # The canonical seq-only path is the FUSED head fed with zero ParaSurf features
    # (modality dropout). Only the FUSED row is reported in the paper.
    print_table(stu_agg, label=f"AntiSite AUX HEAD (legacy, ignore)  |  {dataset_name}")
    print_table(fus_agg, label=f"AntiSite (3D)                       |  {dataset_name}")
    return {"aux": stu_agg, "fused": fus_agg}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",            type=Path, required=True)
    ap.add_argument("--test-examples",   type=Path, required=True)
    ap.add_argument("--test-embeddings", type=Path, required=True)
    ap.add_argument("--dataset",         default="")
    ap.add_argument("--batch-size",      type=int, default=8)
    ap.add_argument("--num-workers",     type=int, default=2)
    ap.add_argument("--enabled-plms",    type=str, nargs="*", default=None,
                    help="Pass training-time PLM order if model used a non-default stack.")
    args = ap.parse_args()
    evaluate(args.ckpt, args.test_examples, args.test_embeddings,
             dataset_name=args.dataset, batch_size=args.batch_size,
             num_workers=args.num_workers, enabled_plms=args.enabled_plms)


if __name__ == "__main__":
    main()
