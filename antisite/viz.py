"""Visualise AntiSite per-residue paratope predictions as heavy/light heat strips.

Renders the output of either inference mode as a 1D heat track over the antibody
sequence, with the framework (FR) and CDR regions marked underneath. CDR/FR
boundaries are located from conserved-residue anchors (the two disulfide
cysteines and the C-terminal WG.G / FG.G motif), so this has no ANARCI/HMMER
dependency.
"""

from __future__ import annotations

import re
from pathlib import Path


def find_regions(seq: str, chain: str) -> list[str]:
    """Per-residue region labels (FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4).

    Chothia-approximate, dependency-free: CDR positions are taken relative to the
    conserved cysteines and the C-terminal WG.G (heavy) / FG.G (light) motif.
    """
    n = len(seq)
    cys = [i for i, a in enumerate(seq) if a == "C"]
    chain = chain.upper()
    if chain == "H":
        m = re.search(r"WG.G", seq[85:])
        endi = (85 + m.start()) if m else n - 11
        c1 = next((i for i in cys if 15 <= i <= 30), (cys[0] if cys else 22))
        trp = seq.find("W", c1 + 8)
        c2 = max([i for i in cys if 80 <= i < endi], default=endi - 13)
        cdr = {"CDR1": (c1 + 4, c1 + 11), "CDR2": (trp + 16, trp + 21), "CDR3": (c2 + 3, endi)}
    else:
        m = re.search(r"FG.G", seq[80:])
        endi = (80 + m.start()) if m else n - 10
        c1 = next((i for i in cys if 15 <= i <= 30), (cys[0] if cys else 22))
        trp = seq.find("W", c1 + 8)
        c2 = max([i for i in cys if 75 <= i < endi], default=endi - 13)
        cdr = {"CDR1": (c1 + 1, c1 + 12), "CDR2": (trp + 15, trp + 21), "CDR3": (c2 + 1, endi)}

    lab: list[str | None] = [None] * n
    for name, (a, b) in cdr.items():
        for i in range(max(0, a), min(n, b)):
            lab[i] = name
    # number the framework gaps by how many CDRs precede them
    out: list[str] = []
    ncdr, i = 0, 0
    while i < n:
        if lab[i] and lab[i].startswith("CDR"):
            name = lab[i]
            while i < n and lab[i] == name:
                out.append(name)
                i += 1
            ncdr += 1
        else:
            out.append(f"FR{ncdr + 1}")
            i += 1
    return out


def segments(labels: list[str]) -> list[tuple[int, int, str]]:
    """Collapse a per-residue label list into (start, end, name) segments."""
    segs, i = [], 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        segs.append((i, j, labels[i]))
        i = j
    return segs


def plot_paratope(heavy: str, light: str, probs_h, probs_l, out_path,
                  title: str | None = None, dpi: int = 300) -> None:
    """Save a heavy/light per-residue paratope heat-map with FR/CDR region bands.

    Vertical layout: residues run top -> bottom, with the heavy and light chains
    as side-by-side columns. FR/CDR region boxes sit to the right of each column.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cols = []
    if heavy:
        cols.append(("Heavy", np.asarray(probs_h, dtype=float), segments(find_regions(heavy, "H"))))
    if light:
        cols.append(("Light", np.asarray(probs_l, dtype=float), segments(find_regions(light, "L"))))
    if not cols:
        return

    CDR_FC, FR_FC = "#F4C4C4", "#E9E9EC"
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 7.8), squeeze=False)
    axes = list(axes[0])
    fig.subplots_adjust(wspace=0.55, left=0.04, right=0.96, top=0.90, bottom=0.10)

    ims = []
    for ax, (label, probs, segs) in zip(axes, cols):
        N = len(probs)
        im = ax.imshow(probs[:, None], extent=[0, 1.5, N, 0], aspect="auto",
                       cmap="OrRd", vmin=0, vmax=1)
        ims.append(im)
        ax.set_xlim(-1.2, 8.5)
        ax.set_ylim(N, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.add_patch(plt.Rectangle((0, 0), 1.5, N, fill=False, ec="#999", lw=0.8))
        ax.set_title(label, fontsize=13, fontweight="bold", pad=6)
        # residue numbering (1-based) down the left of the strip
        step = 20 if N >= 60 else 10
        positions = list(range(0, N, step))
        if positions[-1] < N - 1:
            positions.append(N - 1)
        for pos in positions:
            y = pos + 0.5
            ax.plot([-0.18, 0], [y, y], color="#777", lw=0.8, clip_on=False)
            ax.text(-0.32, y, str(pos + 1), ha="right", va="center", fontsize=6.8, color="#555")
        for a, b, name in segs:
            is_cdr = name.startswith("CDR")
            ax.add_patch(plt.Rectangle((2.1, a + 0.4), 4.0, (b - a) - 0.8,
                         fc=CDR_FC if is_cdr else FR_FC, ec="#8a8a8a", lw=0.6, clip_on=False))
            if is_cdr or (b - a) >= 10:
                ax.text(4.1, (a + b) / 2, name, ha="center", va="center",
                        fontsize=8.2 if is_cdr else 7.2,
                        fontweight="bold" if is_cdr else "normal", color="#1a1a1a")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.975)
    cb = fig.colorbar(ims[0], ax=axes, orientation="horizontal", fraction=0.045, pad=0.05)
    cb.set_label("paratope probability", fontsize=9)
    cb.set_ticks([0, 0.5, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
