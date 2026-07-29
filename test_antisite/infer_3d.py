"""AntiSite sequence + 3D inference — ParaSurf-style output.

End-to-end: give a PDB and the AntiSite checkpoint, get a B-factor PDB.
Heavy/light chains, sequences, and ParaSurf features are all derived
automatically — you don't pass them.

Usage:
    python -m test_antisite.infer_3d \\
        --receptor 1BVK_receptor_1.pdb \\
        --ckpt     runs/Paragraph/best.pt \\
        [--out-dir results/1BVK]

Output (ParaSurf-style, opens in PyMOL with `color by b-factor`):
    {out_dir}/{pdb_id}_pred.pdb            per-residue B-factor = paratope prob
                                           (residue-resolution; atoms of the
                                           same residue share one score)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ParaSurf"))

from antisite.data.build_dataset import _fv_res_ids_in_order, FV_MAX_RESNUM  # noqa: E402
from antisite.eval.labels import extract_sequence  # noqa: E402
from antisite.models.antisite import PLM_DIMS, AntiSite  # noqa: E402
from antisite.parasurf.parasurf_wrapper import ParaSurfExtractor  # noqa: E402
from ParaSurf.ParaSurf.train.bsite_extraction import Bsite_extractor  # noqa: E402
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

# Default ParaSurf weights — same checkpoint that supervised AntiSite training.
DEFAULT_PARASURF_WEIGHTS = ROOT / "weights" / "Paragraph_expanded_entire_dataset_best.pth"

# Ig framework-4 J motifs (C-terminal end of variable domain) — primary signal.
HEAVY_J_RE = re.compile(r"WG[A-Z]G")          # e.g. WGQG, WGKG, WGRG
LIGHT_J_RE = re.compile(r"FG[A-Z]G")          # e.g. FGQG, FGGG, FGSG, FGTG
# N-terminal V-domain motifs (fallback signal).
HEAVY_N_RE = re.compile(r"^[QE]V[QK]L")       # EVQL, QVQL, EVKL, QVKL
LIGHT_N_RE = re.compile(r"^(D[IV][QV]M|EIVL|QIVL)")  # DIQM/DIVM kappa, EIVL/QIVL lambda


def _chains_in_pdb(pdb: Path) -> list[str]:
    seen: list[str] = []
    with pdb.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            c = line[21]
            if c != " " and c not in seen:
                seen.append(c)
    return seen


def _detect_heavy_light(pdb: Path) -> tuple[str, str]:
    """Auto-detect heavy and light chain IDs from Ig V-domain motifs.

    AntiSite is antibody-specific: the model needs to know which chain is heavy vs
    light because of the chain-id embedding and antibody-trained PLMs (AbLang2,
    IgT5, IgBert). If the PDB doesn't look like an antibody Fv, we error out
    clearly rather than silently produce garbage predictions.
    """
    chains = _chains_in_pdb(pdb)
    classified: dict[str, str] = {}  # chain_id -> 'H' or 'L'
    seqs: dict[str, str] = {}
    for c in chains:
        seq = extract_sequence(pdb, c, max_resnum=FV_MAX_RESNUM)
        if len(seq) < 90:
            continue                 # too short to be a V-domain
        seqs[c] = seq
        # Try strong (FR4) motif first, then fall back to N-terminal motif.
        if HEAVY_J_RE.search(seq):
            classified[c] = "H"
        elif LIGHT_J_RE.search(seq):
            classified[c] = "L"
        elif HEAVY_N_RE.search(seq):
            classified[c] = "H"
        elif LIGHT_N_RE.search(seq):
            classified[c] = "L"

    heavies = [c for c, t in classified.items() if t == "H"]
    lights  = [c for c, t in classified.items() if t == "L"]
    if not heavies and not lights:
        raise SystemExit(
            f"\n{pdb.name} does not look like an antibody Fv.\n"
            f"  Chains scanned: {chains}\n"
            f"  None matched Ig V-domain motifs (conserved Cys-Trp-Cys, FR1/FR4 patterns).\n\n"
            f"AntiSite is antibody-specific (uses AbLang2/IgT5/IgBert PLMs and a heavy/light\n"
            f"chain embedding). For non-antibody binding-site prediction, use ParaSurf or a\n"
            f"general-purpose interface predictor.\n\n"
            f"If this IS an antibody and detection just failed, override with\n"
            f"  --heavy-chain <ID>  [--light-chain <ID>]"
        )
    if not heavies:
        raise SystemExit(
            f"Found a light chain but no heavy chain in {pdb.name} (chains {list(seqs.keys())}).\n"
            f"AntiSite needs at least the heavy chain. Pass --heavy-chain to override."
        )
    if len(heavies) > 1 or len(lights) > 1:
        print(f"  ⚠ multiple Ig candidates — heavy={heavies} light={lights}; picking first of each")
    return heavies[0], (lights[0] if lights else "")


def _compute_plm_bundle(heavy: str, light: str, gpu: int,
                        names: list[str]) -> dict[str, torch.Tensor]:
    L = len(heavy) + len(light)
    bundle: dict[str, torch.Tensor] = {}
    for name in names:
        emb = PLM_FNS[name]([heavy], [light], gpu=gpu, single_chain=False)
        bundle[name] = emb[0, :L, :].detach().cpu().float().contiguous()
        assert bundle[name].shape == (L, PLM_DIMS[name])
    return bundle


def _write_bfactor_pdb(receptor: Path, out_dir: Path, residues_best: dict,
                       pdb_stem: str) -> None:
    """Write per-residue B-factor PDB. AntiSite predicts at residue resolution;
    every atom of a residue gets that residue's score. Non-Fv atoms get 0.0
    (otherwise crystal B-factors would dominate the colour scale in PyMOL).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    res_out = out_dir / f"{pdb_stem}_pred.pdb"
    with receptor.open() as src, res_out.open("w") as dst:
        for line in src:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21]
                resnum = line[22:26].strip()
                ins = line[26].strip()
                rid = f"{resnum}_{chain}" + (f"_{ins}" if ins else "")
                score = residues_best.get(rid, 0.0)
                line = f"{line[:60]}{score:6.3f}{line[66:]}"
            dst.write(line)
    print(f"  → {res_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptor", type=Path, required=True, help="Antibody PDB.")
    ap.add_argument("--ckpt",     type=Path, required=True, help="AntiSite checkpoint (.pt).")
    ap.add_argument("--out-dir",  type=Path, default=None,
                    help="Where to write *_pred.pdb and pocket{i}.pdb. "
                         "Defaults to ./{pdb_id}/ in the current directory.")
    ap.add_argument("--heavy-chain", type=str, default=None,
                    help="(optional) Override auto-detected heavy chain ID.")
    ap.add_argument("--light-chain", type=str, default=None,
                    help="(optional) Override auto-detected light chain ID.")
    ap.add_argument("--parasurf-weights", type=Path, default=DEFAULT_PARASURF_WEIGHTS,
                    help=f"(optional) ParaSurf weights. Default: {DEFAULT_PARASURF_WEIGHTS}")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if not args.parasurf_weights.exists():
        raise SystemExit(f"ParaSurf weights not found at {args.parasurf_weights}. "
                         f"Pass --parasurf-weights <path> to override.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    receptor = args.receptor.resolve()
    pdb_stem = receptor.stem.split("_")[0]
    out_dir  = args.out_dir or (Path.cwd() / pdb_stem)

    # 1. Auto-detect heavy/light chains.
    if args.heavy_chain is None or (args.light_chain is None and args.heavy_chain is None):
        h_auto, l_auto = _detect_heavy_light(receptor)
    else:
        h_auto, l_auto = "", ""
    heavy_chain = args.heavy_chain or h_auto
    light_chain = args.light_chain if args.light_chain is not None else l_auto
    light_disp = repr(light_chain) if light_chain else "—"
    print(f"[{pdb_stem}] Detected chains: heavy={heavy_chain!r}  light={light_disp}")

    # 2. Extract Fv sequences and canonical res_id order (heavy first, then light).
    heavy_seq = extract_sequence(receptor, heavy_chain, max_resnum=FV_MAX_RESNUM)
    light_seq = extract_sequence(receptor, light_chain, max_resnum=FV_MAX_RESNUM) if light_chain else ""
    heavy_ids = _fv_res_ids_in_order(receptor, heavy_chain)
    light_ids = _fv_res_ids_in_order(receptor, light_chain) if light_chain else []
    res_ids   = heavy_ids + light_ids
    L_h, L_l, L = len(heavy_seq), len(light_seq), len(heavy_seq) + len(light_seq)
    print(f"[{pdb_stem}] Fv: |H|={L_h}  |L|={L_l}  L={L}")

    # 3. Run ParaSurf → per-residue 256-d features.
    print(f"[{pdb_stem}] Running ParaSurf (this takes a minute)...")
    extractor = ParaSurfExtractor(weights_path=args.parasurf_weights, device=str(dev))
    tout = extractor.compute(receptor)
    cache_index = {rid: i for i, rid in enumerate(tout.res_ids)}
    surface_features = torch.zeros(L, 256, dtype=torch.float32)
    n_covered = 0
    for i, rid in enumerate(res_ids):
        j = cache_index.get(rid)
        if j is not None:
            surface_features[i] = tout.features[j]
            n_covered += 1
    print(f"[{pdb_stem}] ParaSurf covered {n_covered}/{L} Fv residues")

    # 4. PLM embeddings (only the ones the checkpoint uses).
    blob = torch.load(args.ckpt, map_location=dev, weights_only=True)
    sd = blob["state_dict"]
    enabled_plms = blob.get("enabled_plms") or [
        n for n in PLM_DIMS if f"fusion.projs.{n}.weight" in sd
    ]
    print(f"[{pdb_stem}] Computing PLM embeddings ({len(enabled_plms)}: {enabled_plms})...")
    plm_bundle = _compute_plm_bundle(heavy_seq, light_seq, gpu=args.gpu, names=enabled_plms)
    plm_embs   = {name: t.unsqueeze(0).to(dev) for name, t in plm_bundle.items()}
    chain_ids  = torch.zeros(1, L, dtype=torch.long, device=dev)
    chain_ids[0, L_h:] = 1
    valid_mask = torch.ones(1, L, dtype=torch.bool, device=dev)

    # 5. AntiSite fused inference.
    model = AntiSite(enabled_plms=enabled_plms).to(dev).eval()
    model.load_state_dict(sd)
    print(f"[{pdb_stem}] Loaded AntiSite: {args.ckpt}  "
          f"(epoch {blob.get('epoch','?')}, val fused PR-AUC {blob.get('val_fused_pr', float('nan')):.3f})")
    with torch.no_grad():
        out = model(plm_embs=plm_embs, chain_ids=chain_ids, valid_mask=valid_mask,
                    surface_features=surface_features.unsqueeze(0).to(dev))
    probs = out["logits_fused"][0].sigmoid().cpu().numpy()

    # 6. Top-20 console summary.
    aa_seq = list(heavy_seq) + list(light_seq)
    chain_label = ["H"] * L_h + ["L"] * L_l
    print(f"\n  Top-20 predicted paratope residues (FUSED head):")
    print(f"  {'idx':>4}  {'chain':>5}  {'AA':>2}  {'res_id':>10}  {'prob':>6}")
    for idx in np.argsort(probs)[::-1][:20]:
        print(f"  {idx:>4}  {chain_label[idx]:>5}  {aa_seq[idx]:>2}  "
              f"{res_ids[idx]:>10}  {probs[idx]:>6.3f}")

    # 7. Per-residue B-factor PDB.
    residues_best = {rid: float(probs[i]) for i, rid in enumerate(res_ids)}
    print(f"\n[{pdb_stem}] Writing outputs to {out_dir}/")
    _write_bfactor_pdb(receptor, out_dir, residues_best, pdb_stem)

    # 8. ParaSurf-style pocket extraction (pocket{i}.pdb).
    #    Build per-surface-point fused scores by broadcasting each residue's
    #    fused probability to every surface point belonging to that residue,
    #    then run ParaSurf's MeanShift-based binding-site extractor.
    if extractor.last_prot is not None and extractor.last_surf_file is not None:
        prot = extractor.last_prot
        surf_path = extractor.last_surf_file
        point_rids: list[str] = []
        with surf_path.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                # surfpoints col 1 = "{resnum}{chain}" concatenated (e.g. "52B").
                rid_token = parts[1]
                resnum = "".join(c for c in rid_token if c.isdigit() or c == "-")
                chain  = "".join(c for c in rid_token if c.isalpha())
                point_rids.append(f"{resnum}_{chain}")
        if len(point_rids) != len(prot.surf_points):
            print(f"[{pdb_stem}] ⚠ surfpoint count mismatch "
                  f"({len(point_rids)} vs {len(prot.surf_points)}); skipping pockets.")
        else:
            point_scores = np.array(
                [residues_best.get(rid, 0.0) for rid in point_rids],
                dtype=np.float32,
            ).reshape(-1, 1)
            # Redirect pocket{i}.pdb writes to out_dir.
            prot.save_path = str(out_dir)
            prot.binding_sites = []  # reset in case the object was reused
            Bsite_extractor().extract_bsites(prot, point_scores)
            pockets = sorted(out_dir.glob("pocket*.pdb"))
            if pockets:
                print(f"[{pdb_stem}] Wrote {len(pockets)} pocket file(s):")
                for p in pockets:
                    print(f"  → {p}")
            else:
                print(f"[{pdb_stem}] No pockets above threshold.")


if __name__ == "__main__":
    main()
