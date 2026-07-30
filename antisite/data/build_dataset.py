"""Build per-antibody training examples from the ParaSurf cache + PDBs + chain metadata.

Each example fuses three sources of truth into one tensor bundle, aligned residue-by-residue:

  • Sequence (atom-resolved Fv-sized heavy + light inputs)
  • Ground-truth paratope labels (4.5 A criterion vs antigen)
  • ParaSurf outputs (per-residue 256-d pre-classifier surface features)

Output format (per antibody, saved to {out_dir}/{pdb_id}.pt):

    {
        "pdb_id":           str,
        "heavy_seq":        str,                         # length = L_H
        "light_seq":        str,                         # length = L_L
        "res_ids":          list[str],                   # length = L = L_H + L_L, heavy first
        "chain_ids":        LongTensor[L],               # 0 = heavy, 1 = light
        "labels":           FloatTensor[L],              # paratope binary (4.5 A)
        "surface_features": FloatTensor[L, 256],         # cached pre-classifier, zeros where missing
        "surface_mask":     BoolTensor[L],               # True where ParaSurf covers this residue
    }

Run once per (dataset, split). Training then just loads these bundles — no PDB parsing in hot path.

Usage:
    python -m antisite.data.build_dataset \\
        --pdb-dir   test_data/pdbs/PECAN/TEST \\
        --meta      Data/PECAN/test.csv \\
        --cache     cache/PECAN/TEST \\
        --out-dir   examples/PECAN/TEST
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from antisite.eval.labels import (  # noqa: E402
    FV_MAX_RESNUM,
    extract_sequence,
    get_labels,
    parse_pdb_chain_ids,
    select_fv_residues,
)

SURFACE_FEAT_DIM = 256


def _load_meta(meta_csv: Path) -> list[dict]:
    with meta_csv.open() as f:
        return list(csv.DictReader(f))


def _fv_res_ids_in_order(receptor_pdb: Path, chain: str) -> list[str]:
    """Return exactly the residue IDs used by :func:`extract_sequence`."""
    return [residue.res_id for residue in select_fv_residues(receptor_pdb, chain)]


def build_example(
    pdb_id: str,
    receptor_pdb: Path,
    antigen_pdb: Path,
    cache_file: Path,
    heavy_chain: str,
    light_chain: str,
    ag_chains: set[str],
) -> dict:
    """Build one aligned training example for a single antibody."""
    # 1. Sequences (Fv only).
    heavy_seq = extract_sequence(receptor_pdb, heavy_chain, max_resnum=FV_MAX_RESNUM)
    light_seq = extract_sequence(receptor_pdb, light_chain, max_resnum=FV_MAX_RESNUM) if light_chain else ""

    # 2. Residue IDs in PDB order (heavy first, then light) — the canonical order
    #    everything else aligns to.
    heavy_res_ids = _fv_res_ids_in_order(receptor_pdb, heavy_chain)
    light_res_ids = _fv_res_ids_in_order(receptor_pdb, light_chain) if light_chain else []
    res_ids = heavy_res_ids + light_res_ids

    if len(heavy_res_ids) != len(heavy_seq):
        raise RuntimeError(
            f"{pdb_id}: heavy seq ({len(heavy_seq)}) != heavy res_ids ({len(heavy_res_ids)})"
        )
    if len(light_res_ids) != len(light_seq):
        raise RuntimeError(
            f"{pdb_id}: light seq ({len(light_seq)}) != light res_ids ({len(light_res_ids)})"
        )

    L = len(res_ids)
    if L == 0:
        raise RuntimeError(f"{pdb_id}: zero Fv residues")

    chain_ids = torch.zeros(L, dtype=torch.long)
    chain_ids[len(heavy_res_ids):] = 1

    # 3. Ground-truth labels (full Fab, then filter to Fv).
    all_labels = get_labels(receptor_pdb, antigen_pdb, heavy_chain, light_chain or None, ag_chains)
    labels = torch.zeros(L, dtype=torch.float32)
    missing_labels = 0
    for i, rid in enumerate(res_ids):
        if rid in all_labels:
            labels[i] = float(all_labels[rid])
        else:
            missing_labels += 1  # residue in PDB but not labeled — stays 0

    # 4. ParaSurf cache, aligned to res_ids. ParaSurf ran on full Fab so some res_ids
    #    may be missing (e.g. surface-point generation gaps); we mask those out.
    cache = torch.load(cache_file, weights_only=True)
    cache_index = {rid: i for i, rid in enumerate(cache["res_ids"])}

    surface_features = torch.zeros(L, SURFACE_FEAT_DIM, dtype=torch.float32)
    surface_mask = torch.zeros(L, dtype=torch.bool)
    for i, rid in enumerate(res_ids):
        j = cache_index.get(rid)
        if j is None:
            continue
        surface_features[i] = cache["features"][j]
        surface_mask[i] = True

    return {
        "pdb_id":           pdb_id,
        "heavy_seq":        heavy_seq,
        "light_seq":        light_seq,
        "res_ids":          res_ids,
        "chain_ids":        chain_ids,
        "labels":           labels,
        "surface_features": surface_features,
        "surface_mask":     surface_mask,
    }


def build_split(
    pdb_dir: Path,
    meta_csv: Path,
    cache_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    combined_pdb: bool = False,
) -> None:
    rows = _load_meta(meta_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skip = n_fail = 0
    failures: list[tuple[str, str]] = []

    for row in tqdm(rows, desc=f"Building {out_dir.name}"):
        pdb_id = row["pdb_code"]
        heavy = row.get("heavy_chain") or row.get("Heavy_chain") or ""
        light = row.get("light_chain") or row.get("Light_chain") or ""
        ag_str = row.get("antigen_chain") or row.get("ag") or ""
        ag_chains = set(parse_pdb_chain_ids(ag_str))

        if not heavy or not light:
            raise ValueError(f"{pdb_id}: metadata must specify paired heavy and light chains")

        out_path = out_dir / f"{pdb_id}.pt"
        if out_path.exists() and not overwrite:
            n_done += 1
            continue

        if combined_pdb:
            receptor_pdb = pdb_dir / f"{pdb_id}.pdb"
            antigen_pdb = receptor_pdb
            cache_file = cache_dir / f"{pdb_id}_receptor.pt"
        else:
            receptor_pdb = pdb_dir / f"{pdb_id}_receptor_1.pdb"
            antigen_pdb = next(pdb_dir.glob(f"{pdb_id}_antigen*.pdb"), None)
            cache_file = cache_dir / f"{pdb_id}_receptor_1.pt"

        if not receptor_pdb.exists() or antigen_pdb is None or not cache_file.exists():
            n_skip += 1
            continue

        try:
            example = build_example(
                pdb_id=pdb_id,
                receptor_pdb=receptor_pdb,
                antigen_pdb=antigen_pdb,
                cache_file=cache_file,
                heavy_chain=heavy,
                light_chain=light,
                ag_chains=ag_chains,
            )
            torch.save(example, out_path)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            failures.append((pdb_id, f"{type(exc).__name__}: {exc}"))
            if n_fail <= 3:
                traceback.print_exc()

    print(f"\nBuilt {n_done} examples  skipped={n_skip}  failed={n_fail}  total={len(rows)}")
    if failures:
        print("Failures:")
        for pid, msg in failures[:10]:
            print(f"  {pid}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb-dir",  type=Path, required=True, help="Directory of receptor+antigen PDBs for this split.")
    ap.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="Released Data CSV (legacy Heavy_chain/Light_chain/ag schema is also accepted).",
    )
    ap.add_argument("--cache",    type=Path, required=True, help="Directory of ParaSurf cache .pt files.")
    ap.add_argument("--out-dir",  type=Path, required=True, help="Output directory for per-antibody example .pt files.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--combined-pdb",
        action="store_true",
        help=(
            "Use AACDB-style {pdb_code}.pdb complexes for receptor and antigen, "
            "with {pdb_code}_receptor.pt ParaSurf caches."
        ),
    )
    args = ap.parse_args()
    build_split(
        args.pdb_dir,
        args.meta,
        args.cache,
        args.out_dir,
        args.overwrite,
        args.combined_pdb,
    )


if __name__ == "__main__":
    main()
