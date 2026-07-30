"""Evaluate cached ParaSurf scores against ground-truth paratope labels.

Usage:
    python -m antisite.eval.eval_parasurf \
        --dataset PECAN \
        --split TEST \
        --pdb-dir test_data/pdbs/PECAN/TEST \
        --meta    Data/PECAN/test.csv \
        --cache   cache/PECAN/TEST
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.eval.labels import get_labels, parse_pdb_chain_ids
from antisite.eval.metrics import aggregate_metrics, per_protein_metrics, print_table


def _load_meta(meta_csv: Path) -> list[dict]:
    import csv
    rows = []
    with meta_csv.open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def evaluate(
    pdb_dir: Path,
    meta_csv: Path,
    cache_dir: Path,
    dataset_name: str = "",
) -> dict:
    rows = _load_meta(meta_csv)
    per_protein = []
    n_skip = 0

    for row in tqdm(rows, desc="Evaluating ParaSurf"):
        pdb_id = row["pdb_code"]
        heavy = row.get("heavy_chain") or row.get("Heavy_chain") or ""
        light = row.get("light_chain") or row.get("Light_chain") or ""
        ag_str = row.get("antigen_chain") or row.get("ag") or ""
        ag_chains = set(parse_pdb_chain_ids(ag_str))

        if not heavy or not light:
            raise ValueError(f"{pdb_id}: metadata must specify paired heavy and light chains")

        receptor_pdb = pdb_dir / f"{pdb_id}_receptor_1.pdb"
        antigen_pdb  = next(pdb_dir.glob(f"{pdb_id}_antigen*.pdb"), None)
        cache_file   = cache_dir / f"{pdb_id}_receptor_1.pt"

        if not receptor_pdb.exists() or antigen_pdb is None or not cache_file.exists():
            n_skip += 1
            continue

        labels = get_labels(receptor_pdb, antigen_pdb, heavy, light or None, ag_chains)
        data   = torch.load(cache_file, weights_only=True)
        res_ids = data["res_ids"]
        scores  = data["scores"].tolist()

        y_scores, y_labels = [], []
        for rid, sc in zip(res_ids, scores):
            if rid in labels:
                y_scores.append(sc)
                y_labels.append(labels[rid])

        if not y_scores:
            n_skip += 1
            continue

        per_protein.append(per_protein_metrics(y_scores, y_labels))

    agg = aggregate_metrics(per_protein)
    print_table(agg, label=f"ParaSurf  |  {dataset_name}")
    if n_skip:
        print(f"  (skipped {n_skip} proteins — missing PDB/cache/antigen)")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="")
    ap.add_argument("--pdb-dir", type=Path, required=True)
    ap.add_argument("--meta",    type=Path, required=True)
    ap.add_argument("--cache",   type=Path, required=True)
    args = ap.parse_args()
    evaluate(args.pdb_dir, args.meta, args.cache, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
