"""Train the NaiveConcat baseline (4 PLMs + ParaSurf → MLP).

Same data, same val protocol as AntiSite. BCE loss only — no distillation,
no fusion gating, no transformer. The point is to show the architectural
choices in AntiSite are doing real work vs the dumbest reasonable baseline.

Usage:
    python -m antisite.train.train_naive \\
        --train-examples examples/Paragraph_expanded/TRAIN \\
        --train-embeddings embeddings/Paragraph_expanded/TRAIN \\
        --val-examples examples/Paragraph_expanded/VAL \\
        --val-embeddings embeddings/Paragraph_expanded/VAL \\
        --out-dir runs/Paragraph_naive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.dataset import AntisiteDataset, collate_fn
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics
from antisite.models.naive_concat import NaiveConcat
from antisite.train.train import _to_device


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    metrics = []
    for batch in loader:
        batch = _to_device(batch, dev)
        out = model(plm_embs=batch["plm_embs"], chain_ids=batch["chain_ids"],
                    valid_mask=batch["valid_mask"],
                    surface_features=batch["surface_features"])
        probs = out["logits_fused"].sigmoid().float().cpu()
        labels = batch["labels"].cpu()
        valid  = batch["valid_mask"].cpu()
        for i in range(labels.shape[0]):
            m = valid[i]
            y = labels[i][m].tolist()
            metrics.append(per_protein_metrics(probs[i][m].tolist(), y))
    return aggregate_metrics(metrics)


def train_naive(
    train_ex: Path, train_emb: Path, val_ex: Path, val_emb: Path,
    out_dir: Path, dataset_name: str = "",
    epochs: int = 100, patience: int = 7, batch_size: int = 8,
    lr: float = 1e-4, weight_decay: float = 1e-5, num_workers: int = 2,
    seed: int = 0, device: str | None = None,
    enabled_plms: list[str] | None = None,
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AntisiteDataset(train_ex, train_emb)
    val_ds   = AntisiteDataset(val_ex,   val_emb)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  device={dev}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = NaiveConcat(enabled_plms=enabled_plms).to(dev)
    print(f"NaiveConcat: in_dim={model.in_dim}  PLMs={model.plm_names}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_pr, best_epoch, no_improve = [], -1.0, -1, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, nb = 0.0, 0
        for batch in tqdm(train_loader, desc=f"ep {epoch:3d}", leave=False):
            batch = _to_device(batch, dev)
            out = model(plm_embs=batch["plm_embs"], chain_ids=batch["chain_ids"],
                        valid_mask=batch["valid_mask"],
                        surface_features=batch["surface_features"])
            logits = out["logits_fused"]
            mask = batch["valid_mask"]
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], batch["labels"][mask].float()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            nb += 1
        train_avg = running / max(nb, 1)
        val = evaluate(model, val_loader, dev)
        dt = time.time() - t0
        print(f"ep {epoch:3d}  train_L={train_avg:.4f}  "
              f"val PR={val['pr_auc_mean']:.3f} ROC={val['roc_auc_mean']:.3f} "
              f"F1={val['f1_mean']:.3f} MCC={val['mcc_mean']:.3f}  ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_L": train_avg, "val": val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val["pr_auc_mean"] > best_pr:
            best_pr, best_epoch, no_improve = val["pr_auc_mean"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_pr": best_pr,
                        "enabled_plms": model.plm_names}, out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop ep{epoch}  best ep{best_epoch} PR={best_pr:.3f}")
                break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--train-examples",   type=Path, required=True)
    ap.add_argument("--train-embeddings", type=Path, required=True)
    ap.add_argument("--val-examples",     type=Path, required=True)
    ap.add_argument("--val-embeddings",   type=Path, required=True)
    ap.add_argument("--out-dir",          type=Path, required=True)
    ap.add_argument("--epochs",           type=int, default=100)
    ap.add_argument("--patience",         type=int, default=7)
    ap.add_argument("--batch-size",       type=int, default=8)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--weight-decay",     type=float, default=1e-5)
    ap.add_argument("--num-workers",      type=int, default=2)
    ap.add_argument("--seed",             type=int, default=0)
    ap.add_argument("--enabled-plms",     type=str, nargs="*", default=None)
    args = ap.parse_args()
    train_naive(
        args.train_examples, args.train_embeddings, args.val_examples, args.val_embeddings,
        args.out_dir, dataset_name=args.dataset,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, num_workers=args.num_workers,
        seed=args.seed, enabled_plms=args.enabled_plms,
    )


if __name__ == "__main__":
    main()
