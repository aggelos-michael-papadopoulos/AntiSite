"""5-fold CV training for the NaiveConcat baseline. Mirrors train_kfold.py."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.dataset import AntisiteDataset, collate_fn
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics
from antisite.models.naive_concat import NaiveConcat
from antisite.train.train import _to_device


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, k)]


@torch.no_grad()
def _eval(model, loader, dev):
    model.eval()
    metrics = []
    for b in loader:
        b = _to_device(b, dev)
        out = model(plm_embs=b["plm_embs"], chain_ids=b["chain_ids"],
                    valid_mask=b["valid_mask"], surface_features=b["surface_features"])
        probs = out["logits_fused"].sigmoid().float().cpu()
        labels = b["labels"].cpu(); valid = b["valid_mask"].cpu()
        for i in range(labels.shape[0]):
            m = valid[i]
            y = labels[i][m].tolist()
            metrics.append(per_protein_metrics(probs[i][m].tolist(), y))
    return aggregate_metrics(metrics)


def run(train_val_ex, train_val_emb, test_ex, test_emb, out_dir,
        folds=5, epochs=100, patience=7, batch_size=8, lr=1e-4, wd=1e-5,
        seed=0):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    full_ds = AntisiteDataset(train_val_ex, train_val_emb)
    test_ds = AntisiteDataset(test_ex, test_emb)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2, pin_memory=True)
    folds_idx = _kfold_indices(len(full_ds), folds, seed)
    fold_results = []

    for k, val_idx in enumerate(folds_idx):
        torch.manual_seed(seed + k)
        train_idx = np.concatenate([f for j, f in enumerate(folds_idx) if j != k])
        train_ds, val_ds = Subset(full_ds, train_idx.tolist()), Subset(full_ds, val_idx.tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = NaiveConcat().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best_pr, best_ep, no_imp = -1.0, -1, 0
        ckpt = out_dir / f"fold{k}_best.pt"

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            tot, nb = 0.0, 0
            for batch in train_loader:
                batch = _to_device(batch, dev)
                out = model(plm_embs=batch["plm_embs"], chain_ids=batch["chain_ids"],
                            valid_mask=batch["valid_mask"],
                            surface_features=batch["surface_features"])
                m = batch["valid_mask"]
                loss = F.binary_cross_entropy_with_logits(
                    out["logits_fused"][m], batch["labels"][m].float()
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            val = _eval(model, val_loader, dev)
            print(f"fold{k} ep{epoch:3d}  L={tot/max(nb,1):.4f}  "
                  f"val PR={val['pr_auc_mean']:.3f}  ({time.time()-t0:.1f}s)")
            if val["pr_auc_mean"] > best_pr:
                best_pr, best_ep, no_imp = val["pr_auc_mean"], epoch, 0
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_pr": best_pr, "fold": k,
                            "enabled_plms": model.plm_names}, ckpt)
            else:
                no_imp += 1
                if no_imp >= patience:
                    break

        blob = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(blob["state_dict"])
        test_metrics = _eval(model, test_loader, dev)
        print(f"fold{k} TEST: PR={test_metrics['pr_auc_mean']:.4f}  "
              f"F1={test_metrics['f1_mean']:.4f}  MCC={test_metrics['mcc_mean']:.4f}")
        fold_results.append({"fold": k, "best_epoch": best_ep, "best_val_pr": best_pr,
                             "test": test_metrics})
        (out_dir / "kfold_results.json").write_text(json.dumps(fold_results, indent=2))

    print("\n=== Aggregate (mean ± std) ===")
    for m in ("pr_auc", "roc_auc", "f1", "mcc"):
        v = [r["test"][f"{m}_mean"] for r in fold_results]
        print(f"  {m:7s}: {np.mean(v):.4f} ± {np.std(v):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-val-examples",   type=Path, required=True)
    ap.add_argument("--train-val-embeddings", type=Path, required=True)
    ap.add_argument("--test-examples",        type=Path, required=True)
    ap.add_argument("--test-embeddings",      type=Path, required=True)
    ap.add_argument("--out-dir",              type=Path, required=True)
    ap.add_argument("--folds",                type=int, default=5)
    ap.add_argument("--epochs",               type=int, default=100)
    ap.add_argument("--patience",             type=int, default=7)
    ap.add_argument("--batch-size",           type=int, default=8)
    ap.add_argument("--lr",                   type=float, default=1e-4)
    ap.add_argument("--seed",                 type=int, default=0)
    a = ap.parse_args()
    run(a.train_val_examples, a.train_val_embeddings, a.test_examples, a.test_embeddings,
        a.out_dir, folds=a.folds, epochs=a.epochs, patience=a.patience,
        batch_size=a.batch_size, lr=a.lr, seed=a.seed)


if __name__ == "__main__":
    main()
