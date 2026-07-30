"""Precompute per-residue PLM embeddings for every antibody example.

For each example (heavy_seq + light_seq, Fv-trimmed, length L = L_H + L_L), we compute
six per-residue embeddings and save them aligned to the same residue order used by
``build_dataset`` (heavy first, then light). Precomputing once means the training loop
is pure disk I/O — no PLM forwards in the hot path.

Output: one ``.pt`` per antibody under ``embeddings/{dataset}/{split}/{pdb_id}.pt``:

    {
        "ablang2":   FloatTensor[L, 480],
        "antiberty": FloatTensor[L, 512],
        "esm2":      FloatTensor[L, 1280],
        "prot_t5":   FloatTensor[L, 1024],
        "igt5":      FloatTensor[L, 1024],
        "igbert":    FloatTensor[L, 1024],
    }

We batch per-PLM to amortize model load, then slice each batched output to the example's
true L and store tensors aligned to ``res_ids`` — the same order as build_dataset output.

Usage:
    python -m antisite.data.build_embeddings \\
        --examples-dir examples/PECAN/TEST \\
        --out-dir      embeddings/PECAN/TEST \\
        [--batch-size 16] [--gpu 0] [--plms ablang2,antiberty,esm2,prot_t5,igt5,igbert]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.data.plm_embeddings import (  # noqa: E402
    compute_ablang_embeddings,
    compute_antiberty_embeddings,
    compute_esm_embeddings,
    compute_igbert_embeddings,
    compute_igt5_embeddings,
    compute_t5_embeddings,
)

PLM_FNS = {
    "ablang2":   compute_ablang_embeddings,
    "antiberty": compute_antiberty_embeddings,
    "esm2":      compute_esm_embeddings,
    "prot_t5":   compute_t5_embeddings,
    "igt5":      compute_igt5_embeddings,
    "igbert":    compute_igbert_embeddings,
}


def _load_examples(examples_dir: Path) -> list[dict]:
    """Read minimal metadata per example: pdb_id, heavy_seq, light_seq, L."""
    out = []
    for p in sorted(examples_dir.glob("*.pt")):
        ex = torch.load(p, weights_only=False)
        out.append({
            "pdb_id":    ex["pdb_id"],
            "heavy_seq": ex["heavy_seq"],
            "light_seq": ex["light_seq"],
            "L":         len(ex["res_ids"]),
            "source_path": p,
        })
    return out


def _compute_plm_split(
    plm_name: str,
    examples: list[dict],
    batch_size: int,
    gpu: int,
) -> list[torch.Tensor]:
    """Run one PLM across all examples in batches. Return per-example [L_i, d] tensors."""
    fn = PLM_FNS[plm_name]
    out: list[torch.Tensor] = [None] * len(examples)  # type: ignore[list-item]
    desc = f"  {plm_name}"

    for start in tqdm(range(0, len(examples), batch_size), desc=desc):
        batch = examples[start:start + batch_size]
        heavy = [ex["heavy_seq"] for ex in batch]
        light = [ex["light_seq"] for ex in batch]
        # Paraplume helpers pad to 285 (or 'longest' for T5); we slice back to real length.
        emb = fn(heavy, light, gpu=gpu, single_chain=False)  # [B, T, d]
        for j, ex in enumerate(batch):
            L = ex["L"]
            out[start + j] = emb[j, :L, :].detach().cpu().half().contiguous()
        del emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return out


def build_embeddings(
    examples_dir: Path,
    out_dir: Path,
    batch_size: int,
    gpu: int,
    plms: list[str],
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = _load_examples(examples_dir)
    if not examples:
        raise RuntimeError(f"No examples in {examples_dir}")

    print(f"Loaded {len(examples)} examples from {examples_dir}")

    # Skip examples fully done if not overwriting.
    todo_idx = list(range(len(examples)))
    if not overwrite:
        todo_idx = [
            i for i in todo_idx
            if not _all_plms_present(
                out_dir / f"{examples[i]['pdb_id']}.pt",
                plms,
                examples[i]["source_path"],
            )
        ]
    if not todo_idx:
        print("All examples already have embeddings — nothing to do.")
        return

    # Subset to pending examples; keep index alignment for saving.
    pending = [examples[i] for i in todo_idx]

    # Load any existing partial files so we can merge per-PLM updates.
    cached: dict[str, dict] = {}
    for ex in pending:
        p = out_dir / f"{ex['pdb_id']}.pt"
        source_is_current = (
            p.exists()
            and p.stat().st_mtime_ns >= ex["source_path"].stat().st_mtime_ns
        )
        # Never carry unrequested but stale PLM tensors into a newly timestamped
        # bundle: a later partial run could otherwise mistake them for current.
        cached[ex["pdb_id"]] = (
            torch.load(p, weights_only=True)
            if source_is_current and not overwrite
            else {}
        )

    for plm in plms:
        print(f"\n[{plm}] computing for {len(pending)} examples")
        per_ex = _compute_plm_split(plm, pending, batch_size, gpu)
        for ex, t in zip(pending, per_ex, strict=True):
            cached[ex["pdb_id"]][plm] = t
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for ex in pending:
        torch.save(cached[ex["pdb_id"]], out_dir / f"{ex['pdb_id']}.pt")

    print(f"\nSaved {len(pending)} embedding bundles to {out_dir}")


def _all_plms_present(path: Path, plms: list[str], source_path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_mtime_ns < source_path.stat().st_mtime_ns:
        return False
    try:
        blob = torch.load(path, weights_only=True)
    except Exception:  # noqa: BLE001
        return False
    return all(p in blob for p in plms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples-dir", type=Path, required=True)
    ap.add_argument("--out-dir",      type=Path, required=True)
    ap.add_argument("--batch-size",   type=int, default=16)
    ap.add_argument("--gpu",          type=int, default=0)
    ap.add_argument("--plms",         type=str, default=",".join(PLM_FNS.keys()),
                    help="Comma-separated PLM names. Default: all 6.")
    ap.add_argument("--overwrite",    action="store_true")
    args = ap.parse_args()

    plms = [p.strip() for p in args.plms.split(",") if p.strip()]
    unknown = [p for p in plms if p not in PLM_FNS]
    if unknown:
        raise SystemExit(f"Unknown PLMs: {unknown}. Available: {list(PLM_FNS)}")

    build_embeddings(args.examples_dir, args.out_dir, args.batch_size, args.gpu, plms, args.overwrite)


if __name__ == "__main__":
    main()
