"""Train AntiSite end-to-end on a dataset split with val PR-AUC early stopping.

Writes per-epoch metrics (train loss components + val PR-AUC / ROC-AUC / F1 / MCC for
both the sequence-only aux head and the fused head) to
``{out_dir}/history.json``, the best-checkpoint state_dict to ``{out_dir}/best.pt``,
and a 2×2 metrics chart to ``{out_dir}/metrics.png`` when training finishes.

Early stopping patience is 7 epochs on val PR-AUC of the **fused head** (the one we
actually ship at structure-available inference time).

Usage:
    python -m antisite.train.train \\
        --dataset PECAN \\
        --train-examples examples/PECAN/TRAIN \\
        --train-embeddings embeddings/PECAN/TRAIN \\
        --val-examples   examples/PECAN/VAL \\
        --val-embeddings embeddings/PECAN/VAL \\
        --out-dir runs/PECAN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.dataset import AntisiteDataset, collate_fn        # noqa: E402
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics  # noqa: E402
from antisite.models.antisite import AntiSite                        # noqa: E402
from antisite.train.loss import LossConfig, antisite_loss            # noqa: E402


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
    model: AntiSite,
    loader: DataLoader,
    device: torch.device,
    zero_surface: bool = False,
) -> dict:
    """Compute per-protein → aggregate metrics for both aux and fused heads.

    If ``zero_surface`` is True, surface_features are replaced with zeros for the
    fused head (ablation row F: did fusion actually learn to use 3D?).
    """
    model.eval()
    stu_metrics: list[dict] = []
    fus_metrics: list[dict] = []

    for batch in loader:
        batch = _to_device(batch, device)
        tf = batch["surface_features"]
        if zero_surface:
            tf = torch.zeros_like(tf)
        out = model(
            plm_embs=batch["plm_embs"],
            chain_ids=batch["chain_ids"],
            valid_mask=batch["valid_mask"],
            surface_features=tf,
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

    return {
        "aux": aggregate_metrics(stu_metrics),
        "fused":   aggregate_metrics(fus_metrics),
    }


def _plot_history(history: list[dict], path: Path) -> None:
    """Save a 2×2 chart of val PR-AUC / ROC-AUC / F1 / MCC across epochs."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    metric_keys = [("pr_auc", "PR AUC"), ("roc_auc", "ROC AUC"), ("f1", "F1"), ("mcc", "MCC")]

    for ax, (key, title) in zip(axes.flat, metric_keys, strict=True):
        fus = [h["val"]["fused"][f"{key}_mean"] for h in history]
        fus_zero = [h["val"].get("fused_zero_surface", {}).get(f"{key}_mean", float("nan"))
                    for h in history]
        ax.plot(epochs, fus,      marker="s", label="AntiSite (3D)",       color="C0")
        ax.plot(epochs, fus_zero, marker="^", label="AntiSite (seq-only)", color="C2", ls="--")
        ax.set_title(f"Val {title}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(
    train_ex: Path, train_emb: Path,
    val_ex: Path,   val_emb: Path,
    out_dir: Path,
    dataset_name: str = "",
    epochs: int = 100,
    patience: int = 7,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    num_workers: int = 2,
    device: str | None = None,
    seed: int = 0,
    lambda_fused: float = 1.0,
    track_head: str = "fused",  # "fused" or "aux" — which val PR-AUC drives early stop / best.pt
    enabled_plms: list[str] | None = None,
    modality_dropout: float = 0.0,  # prob of zeroing surface_features per batch
    cross_modal_layers: int = 0,    # 0 = no cross-modal attention (default AntiSite)
    cross_modal_heads: int = 4,
) -> None:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AntisiteDataset(train_ex, train_emb)
    val_ds   = AntisiteDataset(val_ex,   val_emb)
    print(f"Train: {len(train_ds)} ex  |  Val: {len(val_ds)} ex  |  device={dev}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    model = AntiSite(enabled_plms=enabled_plms,
                     cross_modal_layers=cross_modal_layers,
                     cross_modal_heads=cross_modal_heads).to(dev)
    if enabled_plms is not None:
        print(f"PLMs in use ({len(enabled_plms)}): {enabled_plms}")
    if cross_modal_layers > 0:
        print(f"Cross-modal attention: {cross_modal_layers} layer(s), {cross_modal_heads} heads")
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cfg   = LossConfig(lambda_fused=lambda_fused)
    print(f"Loss weight: λ_fused={cfg.lambda_fused}")
    print(f"Tracking val PR-AUC of: {track_head} head")
    if modality_dropout > 0:
        print(f"Modality dropout: p={modality_dropout} (surface_features zeroed per batch)")
    (out_dir / "config.json").write_text(json.dumps({
        "dataset": dataset_name,
        "lambda_fused": cfg.lambda_fused,
        "track_head":   track_head,
        "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "epochs": epochs, "patience": patience, "seed": seed,
        "modality_dropout": modality_dropout,
    }, indent=2))

    history: list[dict] = []
    best_pr = -1.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running = {k: 0.0 for k in ("L_total", "L_aux", "L_fused")}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"ep {epoch:3d}", leave=False)
        for batch in pbar:
            batch = _to_device(batch, dev)
            tf = batch["surface_features"]
            if modality_dropout > 0 and torch.rand(1).item() < modality_dropout:
                tf = torch.zeros_like(tf)
            out = model(
                plm_embs=batch["plm_embs"],
                chain_ids=batch["chain_ids"],
                valid_mask=batch["valid_mask"],
                surface_features=tf,
            )
            total, parts = antisite_loss(out, batch, cfg)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in running:
                running[k] += parts[k]
            n_batches += 1
            pbar.set_postfix(L=f"{parts['L_total']:.3f}",
                             aux=f"{parts['L_aux']:.3f}",
                             fused=f"{parts['L_fused']:.3f}")

        train_avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        val = evaluate(model, val_loader, dev)
        val["fused_zero_surface"] = evaluate(model, val_loader, dev, zero_surface=True)["fused"]

        fused_pr = val["fused"]["pr_auc_mean"]
        track_pr = val[track_head]["pr_auc_mean"]
        dt = time.time() - t0
        line = (f"ep {epoch:3d}  train_L={train_avg['L_total']:.4f}  "
                f"val_aux PR={val['aux']['pr_auc_mean']:.3f} "
                f"F1={val['aux']['f1_mean']:.3f}  |  "
                f"val_fuse PR={fused_pr:.3f} "
                f"ROC={val['fused']['roc_auc_mean']:.3f} "
                f"F1={val['fused']['f1_mean']:.3f} "
                f"MCC={val['fused']['mcc_mean']:.3f}  "
                f"({dt:.1f}s)")
        print(line)

        history.append({"epoch": epoch, "train": train_avg, "val": val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        _plot_history(history, out_dir / "metrics.png")

        if track_pr > best_pr:
            best_pr = track_pr
            best_epoch = epoch
            no_improve = 0
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_fused_pr": best_pr,
                        "enabled_plms": model.fusion.plm_names}, out_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at ep {epoch} (no val-PR improvement for {patience} epochs). "
                      f"Best ep={best_epoch}  PR={best_pr:.3f}")
                break

    _plot_history(history, out_dir / "metrics.png")
    print(f"Chart saved to {out_dir/'metrics.png'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",          default="")
    ap.add_argument("--train-examples",   type=Path, required=True)
    ap.add_argument("--train-embeddings", type=Path, required=True)
    ap.add_argument("--val-examples",     type=Path, required=True)
    ap.add_argument("--val-embeddings",   type=Path, required=True)
    ap.add_argument("--out-dir",          type=Path, required=True)
    ap.add_argument("--epochs",           type=int,  default=100)
    ap.add_argument("--patience",         type=int,  default=7)
    ap.add_argument("--batch-size",       type=int,  default=8)
    ap.add_argument("--lr",               type=float, default=1e-4)
    ap.add_argument("--weight-decay",     type=float, default=1e-5)
    ap.add_argument("--num-workers",      type=int,  default=2)
    ap.add_argument("--seed",             type=int,  default=0)
    ap.add_argument("--lambda-fused",     type=float, default=1.0)
    ap.add_argument("--track-head",       type=str,   default="fused",
                    choices=["fused", "aux"],
                    help="Which val PR-AUC drives early stopping / best.pt selection.")
    ap.add_argument("--drop-plm",         type=str,   default=None,
                    help="PLM name to ablate (drop). Mutually exclusive with --enabled-plms.")
    ap.add_argument("--enabled-plms",     type=str,   nargs="*", default=None,
                    help="Explicit list of PLM names to keep. Default: all 6.")
    ap.add_argument("--modality-dropout", type=float, default=0.0,
                    help="Prob of zeroing surface_features per batch (modality dropout).")
    ap.add_argument("--cross-modal-layers", type=int, default=0,
                    help="Number of cross-modal attention layers (z_seq queries 3D); "
                         "0 disables (default = current AntiSite).")
    ap.add_argument("--cross-modal-heads", type=int, default=4)
    args = ap.parse_args()
    enabled = args.enabled_plms
    if args.drop_plm is not None:
        from antisite.models.antisite import PLM_DIMS as _PD
        enabled = [n for n in _PD if n != args.drop_plm]

    train(
        train_ex=args.train_examples, train_emb=args.train_embeddings,
        val_ex=args.val_examples,     val_emb=args.val_embeddings,
        out_dir=args.out_dir,         dataset_name=args.dataset,
        epochs=args.epochs,           patience=args.patience,
        batch_size=args.batch_size,   lr=args.lr,
        weight_decay=args.weight_decay, num_workers=args.num_workers,
        seed=args.seed,
        lambda_fused=args.lambda_fused, track_head=args.track_head,
        enabled_plms=enabled,
        modality_dropout=args.modality_dropout,
        cross_modal_layers=args.cross_modal_layers,
        cross_modal_heads=args.cross_modal_heads,
    )


if __name__ == "__main__":
    main()
