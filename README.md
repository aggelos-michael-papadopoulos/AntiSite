# AntiSite: Unified paratope prediction for antibodies and nanobodies with or without structure

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hugging Face Space](<https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue>)](https://huggingface.co/spaces/angepapa/AntiSite)
[![Docker Hub](<https://img.shields.io/docker/image-size/angepapa/antisite/latest?logo=docker&label=docker%20image>)](https://hub.docker.com/r/angepapa/antisite)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21705412.svg)](https://doi.org/10.5281/zenodo.21705412)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2026.08.24.745914-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.08.24.745914v1)

AntiSite predicts the **paratope** (the antibody binding site) of conventional antibodies and
nanobodies. Both inference modes are **antigen-agnostic**: no antigen sequence or structure is required.
Frozen **ProtT5 and ESM-2** protein language models provide sequence representations, and the same
trained AntiSite sequence encoder, fusion block and prediction head operate with or without structural
input. **Structural-modality dropout** withholds the surface features on half of the training batches,
teaching these shared weights to handle both settings without retraining.

When a structure is provided, a separately pretrained
[ParaSurf](https://github.com/aggelos-michael-papadopoulos/ParaSurf) encoder provides molecular-surface
features, which AntiSite combines with sequence representations through cross-modal attention.
ParaSurf was pretrained for paratope prediction and remains frozen during AntiSite training;
its classification head is unused. Thus, **both modes share one AntiSite prediction checkpoint**,
while structure-aware inference additionally requires the corresponding ParaSurf encoder weights.

| Mode | Input | When to use |
|---|---|---|
| **Sequence-only** | Heavy + light chain sequences, or a VHH sequence | No antibody structure available |
| **Structure-aware (3D)** | Antibody structure; sequences are extracted automatically | Reliable structural information available |

The paired-antibody-trained model also supports nanobodies without additional training.
The benefit of predicted structures depends on their accuracy.

![AntiSite overview](assets/antisite_overview_new.png)

---

## ⚙️ Installation

The full installation below supports **both** sequence-only and structure-aware inference.
The same AntiSite checkpoint runs either mode; sequence-only prediction does not run ParaSurf
or require its weights.

### 🐳 Docker (recommended — zero setup, nothing to download)

A **fully self-contained** GPU image is the easiest way to run AntiSite. It bakes in the code, the
ParaSurf surface stack (DMS, OpenBabel, pdb2pqr) and **all** model weights — AntiSite checkpoints,
ParaSurf, ProtT5 and ESM-2 — so once you have the image there is **nothing left to download** and no
native dependencies to install. Requirements: an NVIDIA GPU with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(so `--gpus all` works).

**Pull the pre-built image:**

```bash
docker pull angepapa/antisite:latest          # or: sudo docker pull angepapa/antisite:latest
```

👉 For how to **run predictions**, run the **offline smoke test**, and **build the image yourself**, see
**[docker/README.md](docker/README.md)**.

> Prefer a manual install, or no GPU-in-Docker? Use the from-source steps below.

---

### 📦 1. Core package (manual install from source)

```bash
git clone https://github.com/aggelos-michael-papadopoulos/AntiSite.git
cd AntiSite
conda create -n antisite python=3.10 -y
conda activate antisite
pip install -e ".[threed]"
```

This installs AntiSite and PyTorch, the **two protein language models** (**ProtT5** and **ESM-2**;
pretrained weights download automatically from HuggingFace on first use), and the
extra dependencies ParaSurf needs (`torchsummary`, `jsonpickle`).

### 🧬 2. ParaSurf (structure features)

The 3D path runs [ParaSurf](https://github.com/aggelos-michael-papadopoulos/ParaSurf) to extract
molecular-surface features from the antibody structure:

```bash
# 1) Clone ParaSurf into the repo root
git clone https://github.com/aggelos-michael-papadopoulos/ParaSurf.git

# 2) OpenBabel (chemical features): conda only
conda install -c conda-forge openbabel -y

# 3) DMS molecular-surface program (needs sudo); pdb2pqr ships inside ParaSurf
cd ParaSurf/dms && sudo make install && cd ../..

# 4) ParaSurf model weights — download FIRST (required for 3D mode).
#    Download the Paragraph-expanded ParaSurf model (link below) and save it as:
#        ParaSurf/Paragraph_expanded_entire_dataset_best.pth
```

Download the ParaSurf **Paragraph-expanded** weights here: [Paragraph_expanded_entire_dataset_best.pth](https://drive.google.com/uc?export=download&id=1nd3npYK303e8owDBvW8Ygd5m9SD1puhR)

> Do **not** run `pip install -r ParaSurf/requirements.txt` inside this environment; it pins an older
> `torch` and will break the AntiSite install. The steps above are all the 3D path needs.

At inference, point AntiSite at a ParaSurf weight file with `--parasurf-weights` (3D mode); omit the
structure to run sequence-only.

The headline released model is [`release/checkpoints/antisite_paragraph.pt`](release/checkpoints/antisite_paragraph.pt).
Use the Paragraph-expanded ParaSurf weights linked above for its structure-aware mode.
Other AntiSite checkpoints require their corresponding ParaSurf extractors for structure-aware use;
do not assume that the Paragraph-expanded extractor is interchangeable with them.
Checkpoints trained on each benchmark are provided under
[`release/checkpoints/`](release/checkpoints/):

| Checkpoint | Trained on |
|---|---|
| `antisite_paragraph.pt` | Paragraph-expanded (headline model) |
| `antisite_pecan.pt` | PECAN |
| `antisite_aacdb.pt` | AACDB |
| `antisite_mipe_fold{0..4}.pt` | MIPE (5-fold) |

Every checkpoint uses the same 2-PLM (ProtT5 + ESM-2) architecture and runs both inference modes.

---

## 🔮 Inference

Both modes use the **same AntiSite prediction checkpoint**. Structure-aware mode also loads
the frozen ParaSurf encoder specified by `--parasurf-weights`.

### Sequence-only (no structure needed)

```bash
python antisite.py \
    --weights release/checkpoints/antisite_paragraph.pt \
    -vh DVKLVQSGPGLVAPSQSLSITCTVSGFSLTTYGVSWVRQPPGKGLEWLGVIWGDGNTTYHSALISRLSISKDNSRSQVFLKLNSLHTDDTATYYCAGNYYGMDYWGQGTSVTVSS \
    -vl DIAMTQTTSSLSASLGQKVTISCRASQDIGNYLNWYQQKPDGTVRLLIYYTSRLHSGVPSRFSGSGSGTDYSLTISNLESEDIATYFCQNGGTNPWTFGGGTKLEVKR
```

The PLMs encode the sequences, and the shared fusion block and prediction head receive **zero**
surface features, as during modality-dropout training. For a nanobody, provide its VHH sequence
with `-vh` and omit `-vl`.

<p align="center">
  <img src="assets/pathA_output.png" width="500" alt="Sequence-only paratope prediction">
</p>
<p align="center"><em>Sequence-only residue-level paratope probabilities.</em></p>

### 3D (when you have a structure)

```bash
python antisite.py \
    --weights release/checkpoints/antisite_paragraph.pt \
    --antibody 1FSK_antibody.pdb \
    --parasurf-weights ParaSurf/Paragraph_expanded_entire_dataset_best.pth #frozen ParaSurf
```

<p align="center">
  <img src="assets/sample_prediction_1FSK.gif" width="700" alt="AntiSite 3D paratope prediction for PDB 1FSK">
</p>
<p align="center"><em>Example 3D paratope prediction for PDB 1FSK; predicted residues are shown in cyan.</em></p>

---

## 📁 Data

Train / validation / test splits for all four benchmarks (Paragraph, PECAN, AACDB, MIPE) are provided
under **[`Data/`](Data/)** — each row gives the antibody heavy/light and antigen chain IDs **and their
sequences**. See [`Data/README.md`](Data/README.md) for the schema, split sizes, correction notes, and
checkpoint caveat.

The exact processed PDB inputs and corrected split metadata are archived on
[Zenodo](https://doi.org/10.5281/zenodo.21705412).

---

## 🏋️ Training

To reproduce AntiSite from scratch (data preparation, PLM embedding, ParaSurf feature precompute, and
single-dataset / 5-fold training), see **[TRAIN.md](TRAIN.md)**.

The source also includes optional **ESM-C 6B** embedding support (`compute_esmc_embeddings` and
the `esmc_6b` model configuration) used in the supplementary comparison. Released checkpoints and
the prediction CLI use ProtT5 + ESM-2. ESM-C embedding generation requires a separate environment
with `multimolecule` and `transformers==5.7.0`; keep it separate from the standard AntiSite environment,
which pins `transformers==4.42.4`.

---

## 🤗 Optional browser demo

The [Hugging Face Space](https://huggingface.co/spaces/angepapa/AntiSite) provides a browser interface
for sequence-only and structure-aware prediction when the hosted demo is available.
Local installation and downloaded checkpoints allow prediction independently of the hosted demo.

<p align="center">
  <img src="assets/antisite_hugging_face.gif" alt="AntiSite Hugging Face demo" width="800">
</p>

---

## 📄 License

Released under the [MIT License](LICENSE).

## 📖 Citation

If you use AntiSite in your research, please cite the preprint:

```bibtex
@article{papadopoulos2026antisite,
  title={AntiSite: Modality Dropout Enables Antibody Paratope Prediction With or Without Structure From a Single Model},
  author={Papadopoulos, Angelos Michail and Alvarez, Federico and Daras, Petros},
  journal={bioRxiv},
  pages={2026--08},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

**This is a preprint; the citation will be updated here if the paper is published after peer review.**
