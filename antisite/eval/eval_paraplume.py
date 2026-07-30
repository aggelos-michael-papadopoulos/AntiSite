"""Evaluate Paraplume on a dataset split against ground-truth paratope labels.

Usage:
    python -m antisite.eval.eval_paraplume \
        --dataset PECAN \
        --pdb-dir test_data/pdbs/PECAN/TEST \
        --meta    Data/PECAN/test.csv \
        [--gpu 0] [--no-large]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.eval.labels import (
    extract_sequence,
    get_labels,
    parse_pdb_chain_ids,
    select_fv_residues,
)
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
    dataset_name: str = "",
    gpu: int = 0,
    large: bool = True,
) -> dict:
    from paraplume.infer import predict_paratope_seq

    rows = _load_meta(meta_csv)
    per_protein = []
    n_skip = 0

    for row in tqdm(rows, desc="Evaluating Paraplume"):
        pdb_id = row["pdb_code"]
        heavy = row.get("heavy_chain") or row.get("Heavy_chain") or ""
        light = row.get("light_chain") or row.get("Light_chain") or ""
        ag_str = row.get("antigen_chain") or row.get("ag") or ""
        ag_chains = set(parse_pdb_chain_ids(ag_str))

        if not heavy or not light:
            raise ValueError(f"{pdb_id}: metadata must specify paired heavy and light chains")

        receptor_pdb = pdb_dir / f"{pdb_id}_receptor_1.pdb"
        antigen_pdb  = next(pdb_dir.glob(f"{pdb_id}_antigen*.pdb"), None)

        if not receptor_pdb.exists() or antigen_pdb is None:
            n_skip += 1
            continue

        # Paraplume expects Fv-sized inputs — full Fab overflows ablang2's
        # 285-token pad limit. The centralized selector also handles offset numbering.
        heavy_seq = extract_sequence(receptor_pdb, heavy, max_resnum=128)
        light_seq = extract_sequence(receptor_pdb, light, max_resnum=128) if light else ""

        if not heavy_seq:
            n_skip += 1
            continue

        try:
            heavy_probs, light_probs = predict_paratope_seq(
                sequence_heavy=heavy_seq,
                sequence_light=light_seq,
                custom_model=None,
                gpu=gpu,
                large=large,
                single_chain=(light_seq == ""),
            )
        except Exception as exc:
            print(f"  WARN {pdb_id}: {exc}")
            n_skip += 1
            continue

        labels = get_labels(receptor_pdb, antigen_pdb, heavy, light or None, ag_chains)

        # Build per-residue (score, label) pairs by iterating the PDB in order
        # to stay consistent with how labels keys are defined.
        y_scores, y_labels = [], []

        # Use the identical centralized selection as sequence extraction.
        heavy_res_ids = [r.res_id for r in select_fv_residues(receptor_pdb, heavy)]

        for rid, prob in zip(heavy_res_ids, heavy_probs):
            if rid in labels:
                y_scores.append(float(prob))
                y_labels.append(labels[rid])

        if light_seq and light_probs is not None:
            light_res_ids = [r.res_id for r in select_fv_residues(receptor_pdb, light)]

            for rid, prob in zip(light_res_ids, light_probs):
                if rid in labels:
                    y_scores.append(float(prob))
                    y_labels.append(labels[rid])

        if not y_scores:
            n_skip += 1
            continue

        per_protein.append(per_protein_metrics(y_scores, y_labels))

    agg = aggregate_metrics(per_protein)
    print_table(agg, label=f"Paraplume  |  {dataset_name}")
    if n_skip:
        print(f"  (skipped {n_skip} proteins — missing PDB/antigen/seq)")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  default="")
    ap.add_argument("--pdb-dir",  type=Path, required=True)
    ap.add_argument("--meta",     type=Path, required=True)
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--no-large", action="store_true")
    args = ap.parse_args()
    evaluate(args.pdb_dir, args.meta, dataset_name=args.dataset,
             gpu=args.gpu, large=not args.no_large)


if __name__ == "__main__":
    main()
