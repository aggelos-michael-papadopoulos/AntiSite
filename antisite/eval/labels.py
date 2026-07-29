"""Ground-truth paratope label extraction from PDB complexes.

Labels follow the same convention as PECAN/Paragraph/MIPE/Paraplume:
  residue = paratope  iff  any non-H atom of the residue is within 4.5 Å
                           of any non-H atom of the antigen.

Key function: ``get_labels(receptor_pdb, antigen_pdb, heavy_chain, light_chain)``
Returns a dict  {res_id: int}  where res_id = "resnum_chain" (+ "_ins" if insertion),
matching the convention used by ParaSurfExtractor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


_DIST_CUTOFF = 4.5


def _parse_heavy_atoms(pdb_path: Path, chains: set[str]) -> dict[str, list[np.ndarray]]:
    """Return {res_id: [xyz, ...]} for all heavy atoms in the given chains."""
    res_atoms: dict[str, list[np.ndarray]] = {}
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            if chain not in chains:
                continue
            element = line[76:78].strip() if len(line) > 76 else ""
            atom_name = line[12:16].strip()
            if element == "H" or atom_name.startswith("H"):
                continue
            resnum = line[22:26].strip()
            insertion = line[26].strip()
            rid = f"{resnum}_{chain}"
            if insertion:
                rid = f"{rid}_{insertion}"
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_atoms.setdefault(rid, []).append(np.array([x, y, z], dtype=np.float32))
    return res_atoms


def _parse_antigen_coords(pdb_path: Path, ag_chains: set[str]) -> np.ndarray:
    """Return (M, 3) array of all non-H antigen atom coordinates."""
    coords: list[np.ndarray] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            chain = line[21]
            if ag_chains and chain not in ag_chains:
                continue
            element = line[76:78].strip() if len(line) > 76 else ""
            atom_name = line[12:16].strip()
            if element == "H" or atom_name.startswith("H"):
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append(np.array([x, y, z], dtype=np.float32))
    return np.stack(coords) if coords else np.zeros((0, 3), dtype=np.float32)


def get_labels(
    receptor_pdb: Path,
    antigen_pdb: Path,
    heavy_chain: str,
    light_chain: str | None,
    ag_chains: set[str] | None = None,
    dist_cutoff: float = _DIST_CUTOFF,
) -> dict[str, int]:
    """Return {res_id: 0/1} for every antibody residue (heavy + light).

    res_id format matches ParaSurfExtractor: "resnum_chain" (+ "_ins").
    """
    ab_chains = {heavy_chain}
    if light_chain:
        ab_chains.add(light_chain)

    res_atoms = _parse_heavy_atoms(receptor_pdb, ab_chains)
    ag_coords = _parse_antigen_coords(antigen_pdb, ag_chains or set())

    labels: dict[str, int] = {}
    if ag_coords.shape[0] == 0:
        return {rid: 0 for rid in res_atoms}

    for rid, atom_list in res_atoms.items():
        ab_xyz = np.stack(atom_list)           # (k, 3)
        # min dist between any ab atom and any ag atom
        diffs = ab_xyz[:, None, :] - ag_coords[None, :, :]  # (k, M, 3)
        min_dist = np.sqrt((diffs ** 2).sum(axis=2)).min()
        labels[rid] = int(min_dist <= dist_cutoff)

    return labels


def extract_sequence(receptor_pdb: Path, chain: str, max_resnum: int | None = None) -> str:
    """Return one-letter sequence for a chain by reading CA atoms in order.

    max_resnum: if set, only include residues with PDB resnum <= max_resnum.
    Pass 128 to trim full-Fab chains to the variable (Fv) domain only,
    which is required for Paraplume (padded to 285 combined tokens).
    """
    _aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
        "MSE": "M", "SEC": "U",
    }
    seen: set[str] = set()
    seq_residues: list[tuple[int, str, str]] = []  # (resnum_int, rid, aa)
    with receptor_pdb.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            resnum = line[22:26].strip()
            insertion = line[26].strip()
            resnum_int = int(resnum) if resnum.lstrip("-").isdigit() else 0
            if max_resnum is not None and resnum_int > max_resnum:
                continue
            rid = f"{resnum}_{chain}"
            if insertion:
                rid = f"{rid}_{insertion}"
            if rid in seen:
                continue
            seen.add(rid)
            res_name = line[17:20].strip()
            aa = _aa3to1.get(res_name, "X")
            seq_residues.append((resnum_int, rid, aa))
    return "".join(aa for _, _, aa in seq_residues)
