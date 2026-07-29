"""5-fold CV training for MIPE: train on TRAIN_VAL folds, eval each fold on TEST.

For each of K folds:
    - train subset = K-1 folds of TRAIN_VAL
    - val   subset = 1 held-out fold (early stopping on val fused PR-AUC)
    - test  = held-out TEST split
After all folds, aggregate the K test results (mean ± std).

Usage:
    python -m antisite.train.train_kfold \\
        --train-val-examples   examples/MIPE/TRAIN_VAL \\
        --train-val-embeddings embeddings/MIPE/TRAIN_VAL \\
        --test-examples        examples/MIPE/TEST \\
        --test-embeddings      embeddings/MIPE/TEST \\
        --out-dir runs/MIPE \\
        --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.dataset import AntisiteDataset, collate_fn
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics, print_table
from antisite.models.antisite import AntiSite
from antisite.train.loss import LossConfig, antisite_loss
from antisite.train.train import _plot_history, _to_device, evaluate


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Deterministic shuffled K-fold split of range(n)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, k)]


def _train_one_fold(
    train_ds, val_ds, test_loader, dev, out_dir: Path, fold: int,
    epochs: int, patience: int, batch_size: int, lr: float, weight_decay: float,
    num_workers: int,
    enabled_plms: list[str] | None = None,
    lambda_fused: float = 1.0,
    modality_dropout: float = 0.0,
    cross_modal_layers: int = 0, cross_modal_heads: int = 4,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = AntiSite(enabled_plms=enabled_plms,
                     cross_modal_layers=cross_modal_layers,
                     cross_modal_heads=cross_modal_heads).to(dev)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cfg   = LossConfig(lambda_fused=lambda_fused)

    history: list[dict] = []
    best_pr, best_epoch, no_improve = -1.0, -1, 0
    ckpt_path = out_dir / f"fold{fold}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = {k: 0.0 for k in ("L_total","L_aux","L_fused")}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"fold{fold} ep{epoch:3d}", leave=False)
        for batch in pbar:
            batch = _to_device(batch, dev)
            tf = batch["surface_features"]
            if modality_dropout > 0 and torch.rand(1).item() < modality_dropout:
                tf = torch.zeros_like(tf)
            out = model(plm_embs=batch["plm_embs"], chain_ids=batch["chain_ids"],
                        valid_mask=batch["valid_mask"], surface_features=tf)
            total, parts = antisite_loss(out, batch, cfg)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in running: running[k] += parts[k]
            n_batches += 1
            pbar.set_postfix(L=f"{parts['L_total']:.3f}")

        train_avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        val = evaluate(model, val_loader, dev)
        fused_pr = val["fused"]["pr_auc_mean"]
        dt = time.time() - t0
        print(f"  fold{fold} ep{epoch:3d}  train_L={train_avg['L_total']:.4f}  "
              f"val_aux PR={val['aux']['pr_auc_mean']:.3f}  |  "
              f"val_fuse PR={fused_pr:.3f} ROC={val['fused']['roc_auc_mean']:.3f} "
              f"F1={val['fused']['f1_mean']:.3f} MCC={val['fused']['mcc_mean']:.3f}  ({dt:.1f}s)")

        history.append({"epoch": epoch, "train": train_avg, "val": val})
        (out_dir / f"fold{fold}_history.json").write_text(json.dumps(history, indent=2))
        _plot_history(history, out_dir / f"fold{fold}_metrics.png")

        if fused_pr > best_pr:
            best_pr, best_epoch, no_improve = fused_pr, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "val_fused_pr": best_pr, "fold": fold,
                        "enabled_plms": model.fusion.plm_names}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  fold{fold} early stop at ep{epoch}  best ep{best_epoch} PR={best_pr:.3f}")
                break

    blob = torch.load(ckpt_path, map_location=dev, weights_only=True)
    model.load_state_dict(blob["state_dict"])
    test_metrics = evaluate(model, test_loader, dev)
    print_table(test_metrics["aux"], label=f"fold{fold} TEST  AUX HEAD (legacy)")
    print_table(test_metrics["fused"],   label=f"fold{fold} TEST  AntiSite (3D)")
    return {"fold": fold, "best_epoch": best_epoch, "best_val_pr": best_pr,
            "test": test_metrics}


def run(
    train_val_ex: Path, train_val_emb: Path,
    test_ex: Path,      test_emb: Path,
    out_dir: Path, folds: int = 5,
    epochs: int = 100, patience: int = 7, batch_size: int = 8,
    lr: float = 1e-4, weight_decay: float = 1e-5,
    num_workers: int = 2, seed: int = 0, device: str | None = None,
    enabled_plms: list[str] | None = None,
    lambda_fused: float = 1.0,
    modality_dropout: float = 0.0,
    cross_modal_layers: int = 0, cross_modal_heads: int = 4,
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)

    full_ds = AntisiteDataset(train_val_ex, train_val_emb)
    test_ds = AntisiteDataset(test_ex,      test_emb)
    n = len(full_ds)
    print(f"TRAIN_VAL: {n} ex  |  TEST: {len(test_ds)} ex  |  folds={folds}  |  device={dev}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    folds_idx = _kfold_indices(n, folds, seed)
    fold_results: list[dict] = []

    for k, val_idx in enumerate(folds_idx):
        train_idx = np.concatenate([f for j, f in enumerate(folds_idx) if j != k])
        train_ds = Subset(full_ds, train_idx.tolist())
        val_ds   = Subset(full_ds, val_idx.tolist())
        print(f"\n=== Fold {k}: train={len(train_ds)}  val={len(val_ds)} ===")
        res = _train_one_fold(
            train_ds, val_ds, test_loader, dev, out_dir, fold=k,
            epochs=epochs, patience=patience, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, num_workers=num_workers,
            enabled_plms=enabled_plms,
            lambda_fused=lambda_fused,
            modality_dropout=modality_dropout,
            cross_modal_layers=cross_modal_layers, cross_modal_heads=cross_modal_heads,
        )
        fold_results.append(res)
        (out_dir / "kfold_results.json").write_text(json.dumps(fold_results, indent=2))

    print("\n" + "="*60)
    print("  K-fold TEST aggregate (mean ± std over folds)")
    print("="*60)
    for head in ("aux", "fused"):
        for metric in ("pr_auc", "roc_auc", "f1", "mcc"):
            vals = [r["test"][head][f"{metric}_mean"] for r in fold_results]
            print(f"  {head:7s} {metric:7s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  "
                  f"(per-fold: {[f'{v:.3f}' for v in vals]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--weight-decay",         type=float, default=1e-5)
    ap.add_argument("--num-workers",          type=int, default=2)
    ap.add_argument("--seed",                 type=int, default=0)
    ap.add_argument("--enabled-plms",         type=str, nargs="*", default=None)
    ap.add_argument("--lambda-fused",         type=float, default=1.0)
    ap.add_argument("--modality-dropout",     type=float, default=0.0)
    ap.add_argument("--cross-modal-layers",   type=int, default=0)
    ap.add_argument("--cross-modal-heads",    type=int, default=4)
    args = ap.parse_args()
    run(args.train_val_examples, args.train_val_embeddings,
        args.test_examples, args.test_embeddings, args.out_dir,
        folds=args.folds, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        num_workers=args.num_workers, seed=args.seed,
        enabled_plms=args.enabled_plms,
        lambda_fused=args.lambda_fused, modality_dropout=args.modality_dropout,
        cross_modal_layers=args.cross_modal_layers, cross_modal_heads=args.cross_modal_heads)


if __name__ == "__main__":
    main()
