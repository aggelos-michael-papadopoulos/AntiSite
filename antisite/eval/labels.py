"""Ground-truth paratope label extraction from PDB complexes.

Labels follow the same convention as PECAN/Paragraph/MIPE/Paraplume:
  residue = paratope  iff  any non-H atom of the residue is within 4.5 Å
                           of any non-H atom of the antigen.

Key function: ``get_labels(receptor_pdb, antigen_pdb, heavy_chain, light_chain)``
Returns a dict  {res_id: int}  where res_id = "resnum_chain" (+ "_ins" if insertion),
matching the convention used by ParaSurfExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DIST_CUTOFF = 4.5

# Antibody inputs historically used author residue numbers <= 128 as a proxy for
# the variable domain.  That works for conventionally numbered structures, but a
# substantial number of valid chains start at 127, 200, 300, 500, or 1000.  In
# those cases the literal cutoff returned an empty (or one-residue) chain.
FV_MAX_RESNUM = 128
MIN_PLAUSIBLE_FV_LENGTH = 80


@dataclass(frozen=True)
class ChainResidue:
    """One atom-resolved residue in PDB appearance order."""

    res_id: str
    resnum: int
    aa: str


_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}


def parse_pdb_chain_ids(value: str) -> list[str]:
    """Return classic one-character PDB chain IDs in their declared order.

    Released metadata uses both compact values (``"DC"``) and separated values
    (``"D;C"`` or ``"D, C"``).  Classic PDB chain IDs occupy one character, so
    treating each non-separator character as one ID handles both schemas.
    """
    return [char for char in value if char not in " ,;:|"]


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


def extract_chain_residues(pdb_path: Path, chain: str) -> list[ChainResidue]:
    """Return unique CA residues for ``chain`` in PDB appearance order.

    PDB author residue numbers are identifiers, not zero-based or one-based
    sequence positions.  Callers that need an Fv-sized subset should use
    :func:`select_fv_residues` instead of interpreting those numbers directly.
    """
    seen: set[str] = set()
    residues: list[ChainResidue] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            resnum = line[22:26].strip()
            insertion = line[26].strip()
            if not resnum.lstrip("-").isdigit():
                continue
            resnum_int = int(resnum)
            rid = f"{resnum}_{chain}"
            if insertion:
                rid = f"{rid}_{insertion}"
            if rid in seen:
                continue
            seen.add(rid)
            res_name = line[17:20].strip()
            residues.append(ChainResidue(rid, resnum_int, _AA3TO1.get(res_name, "X")))
    return residues


def select_fv_residues(
    pdb_path: Path,
    chain: str,
    max_resnum: int = FV_MAX_RESNUM,
    min_plausible_length: int = MIN_PLAUSIBLE_FV_LENGTH,
) -> list[ChainResidue]:
    """Select the atom-resolved antibody input residues for one chain.

    For conventionally numbered structures, retain the historical author-number
    selection (``resnum <= 128``).  This preserves long CDR insertions and the
    exact inputs used by the existing pipeline.  If that selection is implausibly
    short, the chain is offset-numbered; fall back to the first 128 observed
    residues by ordinal position.  The fallback keeps sequence, labels, residue
    IDs, and ParaSurf features aligned and prevents valid paired chains from being
    silently turned into empty inputs.

    A physically unresolved chain shorter than ``min_plausible_length`` is
    returned in full.  Missing coordinates are never invented from SEQRES.
    """
    residues = extract_chain_residues(pdb_path, chain)
    by_author_number = [residue for residue in residues if residue.resnum <= max_resnum]
    if len(by_author_number) >= min_plausible_length:
        return by_author_number
    return residues[:max_resnum]


def extract_sequence(receptor_pdb: Path, chain: str, max_resnum: int | None = None) -> str:
    """Return the atom-resolved one-letter sequence for ``chain``.

    With ``max_resnum=None``, return every observed CA residue.  With a numeric
    cutoff (normally 128), use :func:`select_fv_residues`, including its safe
    fallback for offset author numbering.
    """
    residues = (
        extract_chain_residues(receptor_pdb, chain)
        if max_resnum is None
        else select_fv_residues(receptor_pdb, chain, max_resnum=max_resnum)
    )
    return "".join(residue.aa for residue in residues)
