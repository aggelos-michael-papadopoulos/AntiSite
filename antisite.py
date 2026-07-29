"""AntiSite — unified inference CLI.

Single command, single .pt file, two operating modes:

    # Sequence-only (no PDB)
    python antisite.py --weights release/checkpoints/antisite_paragraph.pt \\
        -vh DVKLVQSGPGLVAPSQSLSITCTVSGFSLTTYG... \\
        -vl DIAMTQTTSSLSASLGQKVTISCRASQDIGNYL...

    # Sequence + 3D (PDB available; ParaSurf is run upstream to extract features)
    python antisite.py --weights release/checkpoints/antisite_paragraph.pt \\
        --antibody 1A2Y_receptor.pdb \\
        --parasurf-weights ParaSurf/Paragraph_expanded_entire_dataset_best.pth  #frozen ParaSurf

The same checkpoint serves both modes — modality-dropout training taught the
fused head to handle either real ParaSurf features or zeros at inference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from antisite.models.antisite import PLM_DIMS, AntiSite  # noqa: E402

# PLM compute functions (lazy-imported only if needed)
PLM_FNS = None


def _lazy_plm_fns():
    global PLM_FNS
    if PLM_FNS is not None:
        return PLM_FNS
    from antisite.data.plm_embeddings import (
        compute_ablang_embeddings, compute_antiberty_embeddings,
        compute_esm_embeddings, compute_igbert_embeddings,
        compute_igt5_embeddings, compute_t5_embeddings,
    )
    PLM_FNS = {
        "ablang2":   compute_ablang_embeddings,
        "antiberty": compute_antiberty_embeddings,
        "esm2":      compute_esm_embeddings,
        "prot_t5":   compute_t5_embeddings,
        "igt5":      compute_igt5_embeddings,
        "igbert":    compute_igbert_embeddings,
    }
    return PLM_FNS


def _compute_plm_bundle(heavy: str, light: str, gpu: int,
                        names: list[str]) -> dict[str, torch.Tensor]:
    fns = _lazy_plm_fns()
    L = len(heavy) + len(light)
    bundle = {}
    for name in names:
        emb = fns[name]([heavy], [light], gpu=gpu, single_chain=False)
        bundle[name] = emb[0, :L, :].detach().cpu().float().contiguous()
        assert bundle[name].shape == (L, PLM_DIMS[name])
    return bundle


def _detect_heavy_light(pdb: Path) -> tuple[str, str]:
    """Auto-detect heavy/light chains from a PDB by sequence length + IgFold logic."""
    sys.path.insert(0, str(ROOT / "test_antisite"))
    from infer_3d import _detect_heavy_light as _detect  # noqa: PLC0415
    return _detect(pdb)


def _load_model(weights: Path, dev: torch.device) -> tuple[AntiSite, dict, list[str]]:
    blob = torch.load(weights, map_location=dev, weights_only=True)
    sd = blob["state_dict"]
    enabled = blob.get("enabled_plms") or [
        n for n in PLM_DIMS if f"fusion.projs.{n}.weight" in sd
    ]
    n_xa = blob.get("cross_modal_layers")
    if n_xa is None:
        n_xa = len({k.split(".")[2] for k in sd if k.startswith("cross_modal.layers")})
    n_heads = blob.get("cross_modal_heads", 4)
    model = AntiSite(enabled_plms=enabled,
                     cross_modal_layers=n_xa,
                     cross_modal_heads=n_heads).to(dev).eval()
    model.load_state_dict(sd)
    return model, blob, enabled


def _format_output(heavy: str, light: str, probs_h, probs_l, mode: str) -> str:
    lines = [f"# AntiSite paratope prediction  ({mode})", ""]
    if heavy:
        lines += ["===== Heavy Chain =====", f"{'AA':<4}  {'Probability':>10}", "-" * 22]
        for aa, p in zip(heavy, probs_h):
            lines.append(f"{aa:<4}  --> {np.round(float(p), 3):>8.3f}")
    if light:
        lines += ["", "===== Light Chain =====", f"{'AA':<4}  {'Probability':>10}", "-" * 22]
        for aa, p in zip(light, probs_l):
            lines.append(f"{aa:<4}  --> {np.round(float(p), 3):>8.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--weights", type=Path, required=True,
                    help="Path to AntiSite checkpoint (release/checkpoints/antisite_*.pt).")
    # Sequence-only mode
    ap.add_argument("-vh", "--heavy", type=str, default=None,
                    help="Heavy chain sequence (sequence-only mode).")
    ap.add_argument("-vl", "--light", type=str, default="",
                    help="Light chain sequence (sequence-only mode; optional).")
    # Seq + 3D mode
    ap.add_argument("--antibody", "--pdb", type=Path, default=None,
                    help="Antibody PDB (sequence + 3D mode). Triggers ParaSurf to extract 3D features.")
    ap.add_argument("--parasurf-weights", type=Path,
                    default=ROOT / "ParaSurf/Paragraph_expanded_entire_dataset_best.pth",
                    help="ParaSurf weights for the seq+3D path.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Save predictions to file. Default: stdout.")
    ap.add_argument("--plot", type=Path, default=None,
                    help="Also save a per-residue paratope heat-map (heavy/light strips with "
                         "CDR/FR regions) to this PNG.")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    if args.antibody is None and args.heavy is None:
        ap.error("Must provide either --antibody PDB (seq+3D) or -vh sequence (seq-only).")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, blob, enabled = _load_model(args.weights, dev)
    rel = blob.get("release", {})
    print(f"Loaded AntiSite (trained on {rel.get('trained_on','?')}, "
          f"epoch {blob.get('epoch','?')}, val PR={blob.get('val_fused_pr', float('nan')):.3f})")

    # ----------------------------------------------------------------------
    # Path B: --antibody PDB given → run ParaSurf, get features, fuse
    # ----------------------------------------------------------------------
    if args.antibody is not None:
        sys.path.insert(0, str(ROOT / "test_antisite"))
        from infer_3d import (  # noqa: PLC0415
            FV_MAX_RESNUM, ParaSurfExtractor, _fv_res_ids_in_order,
            extract_sequence,
        )
        receptor = args.antibody.resolve()
        pdb_stem = receptor.stem.split("_")[0]
        h_chain, l_chain = _detect_heavy_light(receptor)
        light_disp = repr(l_chain) if l_chain else "—"
        print(f"[{pdb_stem}] heavy={h_chain!r}  light={light_disp}")

        heavy_seq = extract_sequence(receptor, h_chain, max_resnum=FV_MAX_RESNUM)
        light_seq = extract_sequence(receptor, l_chain, max_resnum=FV_MAX_RESNUM) if l_chain else ""
        heavy_ids = _fv_res_ids_in_order(receptor, h_chain)
        light_ids = _fv_res_ids_in_order(receptor, l_chain) if l_chain else []
        res_ids = heavy_ids + light_ids
        L_h, L_l, L = len(heavy_seq), len(light_seq), len(heavy_seq) + len(light_seq)

        print(f"[{pdb_stem}] Running ParaSurf on PDB...")
        extractor = ParaSurfExtractor(weights_path=args.parasurf_weights, device=str(dev))
        tout = extractor.compute(receptor)
        cache_index = {rid: i for i, rid in enumerate(tout.res_ids)}
        surface_features = torch.zeros(L, 256, dtype=torch.float32)
        for i, rid in enumerate(res_ids):
            j = cache_index.get(rid)
            if j is not None:
                surface_features[i] = tout.features[j]

        print(f"[{pdb_stem}] Computing PLM embeddings ({len(enabled)}: {enabled})...")
        plm_bundle = _compute_plm_bundle(heavy_seq, light_seq, args.gpu, enabled)
        plm_embs = {n: t.unsqueeze(0).to(dev) for n, t in plm_bundle.items()}
        chain_ids = torch.zeros(1, L, dtype=torch.long, device=dev)
        chain_ids[0, L_h:] = 1
        valid_mask = torch.ones(1, L, dtype=torch.bool, device=dev)

        with torch.no_grad():
            out = model(plm_embs=plm_embs, chain_ids=chain_ids, valid_mask=valid_mask,
                        surface_features=surface_features.unsqueeze(0).to(dev))
        probs = out["logits_fused"][0].sigmoid().cpu().numpy()
        text = _format_output(heavy_seq, light_seq, probs[:L_h], probs[L_h:],
                              mode="seq + 3D")
        plot_data = (heavy_seq, light_seq, probs[:L_h], probs[L_h:])
        mode_short = "structure-aware (3D)"

    # ----------------------------------------------------------------------
    # Path A: -vh / -vl given → no PDB, feed zero surface_features
    # ----------------------------------------------------------------------
    else:
        heavy, light = args.heavy, args.light
        L_h, L_l = len(heavy), len(light)
        L = L_h + L_l
        print(f"Sequence-only mode  |H|={L_h}  |L|={L_l}")
        print(f"Computing PLM embeddings ({len(enabled)}: {enabled})...")
        plm_bundle = _compute_plm_bundle(heavy, light, args.gpu, enabled)
        plm_embs = {n: t.unsqueeze(0).to(dev) for n, t in plm_bundle.items()}
        chain_ids = torch.zeros(1, L, dtype=torch.long, device=dev)
        chain_ids[0, L_h:] = 1
        valid_mask = torch.ones(1, L, dtype=torch.bool, device=dev)
        # Modality dropout was trained with zeros — so feed zeros.
        surface_features = torch.zeros(1, L, 256, dtype=torch.float32, device=dev)

        with torch.no_grad():
            out = model(plm_embs=plm_embs, chain_ids=chain_ids,
                        valid_mask=valid_mask, surface_features=surface_features)
        probs = out["logits_fused"][0].sigmoid().cpu().numpy()
        text = _format_output(heavy, light, probs[:L_h], probs[L_h:],
                              mode="sequence-only (zeroed surface features)")
        plot_data = (heavy, light, probs[:L_h], probs[L_h:])
        mode_short = "sequence-only"

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nPredictions saved to {args.out}")
    else:
        print()
        print(text)

    if args.plot is not None:
        from antisite.viz import plot_paratope  # noqa: PLC0415
        plot_paratope(*plot_data, args.plot,
                      title=f"AntiSite paratope prediction — {mode_short}")
        print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
