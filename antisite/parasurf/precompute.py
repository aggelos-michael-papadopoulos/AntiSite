"""Pre-compute frozen ParaSurf ParaSurf outputs for a directory of antibody PDBs.

One cache file per PDB:

    {cache_dir}/{pdb_id}.pt  = {
        "res_ids":  list[str],            # "resnum_chain" (+ insertion) in PDB order
        "scores":   FloatTensor[N],       # sigmoid, max-aggregated per residue
        "features": FloatTensor[N, 256],  # mean-aggregated per residue (pre-classifier)
        "weights":  str,                  # basename of the ParaSurf checkpoint used
    }

Run once per dataset (PECAN / Paragraph-expanded / MIPE). Training loads from cache — no
ParaSurf forward in the training loop.

Usage:
    python -m antisite.parasurf.precompute \
        --pdb-dir /path/to/pdbs \
        --weights /path/to/weights/PECAN_best.pth \
        --out-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from antisite.parasurf.parasurf_wrapper import ParaSurfExtractor


def iter_pdb_files(pdb_dir: Path, pattern: str = "*.pdb") -> list[Path]:
    return sorted(pdb_dir.glob(pattern))


def cache_path(out_dir: Path, pdb: Path) -> Path:
    return out_dir / f"{pdb.stem}.pt"


def precompute(
    pdb_dir: Path,
    weights: Path,
    out_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
    overwrite: bool = False,
    pattern: str = "*.pdb",
) -> None:
    pdbs = iter_pdb_files(pdb_dir, pattern)
    if not pdbs:
        print(f"No PDB files found in {pdb_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"ParaSurf weights:   {weights}")
    print(f"PDB input dir:     {pdb_dir}  ({len(pdbs)} files)")
    print(f"Cache output dir:  {out_dir}")
    print(f"Device / batch:    {device} / {batch_size}")
    print()

    ParaSurf = ParaSurfExtractor(weights_path=weights, device=device)
    weights_name = Path(weights).name

    n_done = n_skip = n_fail = 0
    failures: list[tuple[str, str]] = []
    t0 = time.time()

    for pdb in tqdm(pdbs, desc="Precomputing ParaSurf"):
        cp = cache_path(out_dir, pdb)
        if cp.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            out = ParaSurf.compute(pdb, batch_size=batch_size)
            torch.save(
                {
                    "res_ids": out.res_ids,
                    "scores": out.scores,
                    "features": out.features,
                    "weights": weights_name,
                },
                cp,
            )
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            failures.append((pdb.name, f"{type(exc).__name__}: {exc}"))
            if n_fail <= 3:
                traceback.print_exc()

    dt = time.time() - t0
    print(
        f"\nDone. cached={n_done}  skipped={n_skip}  failed={n_fail}  "
        f"total={len(pdbs)}  elapsed={dt:.1f}s ({dt / max(1, n_done):.2f}s/pdb)"
    )
    if failures:
        print("\nFailures:")
        for name, msg in failures[:20]:
            print(f"  {name}: {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-dir", type=Path, required=True, help="Directory of input antibody PDBs.")
    ap.add_argument("--weights", type=Path, required=True, help="ParaSurf ParaSurf checkpoint (.pth).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write per-antibody .pt caches.")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (default: cuda).")
    ap.add_argument("--batch-size", type=int, default=64, help="Surface-point batch size (default: 64).")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if cache file exists.")
    ap.add_argument("--pattern", default="*.pdb", help="Glob to filter PDBs (e.g. '*receptor*.pdb').")
    args = ap.parse_args()
    precompute(
        pdb_dir=args.pdb_dir,
        weights=args.weights,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
