# AntiSite — Docker 🐳

A **fully self-contained** GPU image: the code, the ParaSurf surface stack (DMS, OpenBabel, pdb2pqr)
and **all** model weights — AntiSite checkpoints, ParaSurf, ProtT5 and ESM-2 — are baked in. Once you
have the image there is **nothing left to download** and no native dependencies to install.

**Just want to run it?** Pull the pre-built image — no build required:

```bash
docker pull angepapa/antisite:latest          # or: sudo docker pull angepapa/antisite:latest
```

The rest of this page is for **building the image yourself**.

## Requirements

- An NVIDIA GPU with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed (so `--gpus all` works).
- The ParaSurf model weights at `docker/weights/Paragraph_expanded_entire_dataset_best.pth` — these are
  baked into the image and are **not** versioned in git. Download the *Paragraph-expanded* ParaSurf model
  ([Paragraph_expanded_entire_dataset_best.pth]()) and place it there.
- Building needs internet once (~20–40 min, ~25 GB transient disk). The final image is ~16 GB.

## 🚀 Build the image

> Run this from the **repository root** (where the `Dockerfile` lives) — the build copies in the
> code, checkpoints and bundled sample, so the build context must be the repo root, not this folder.

```bash
docker build -t antisite:latest .
```

## 🧪 Run the offline smoke test

Runs **both** modes (sequence-only AND 3D) on the bundled sample — no network, no extra downloads:

```bash
docker run --rm --gpus all antisite:latest bash smoke_test.sh
```

## 🔍 Run a prediction

```bash
# Sequence-only (no structure needed)
docker run --rm --gpus all antisite:latest python antisite.py \
    --weights release/checkpoints/antisite_paragraph.pt -vh <HEAVY_SEQ> -vl <LIGHT_SEQ>

# 3D — mount the folder that holds your structure
docker run --rm --gpus all -v "$PWD":/data antisite:latest python antisite.py \
    --weights release/checkpoints/antisite_paragraph.pt \
    --antibody /data/my_antibody.pdb --parasurf-weights "$PARASURF_WEIGHTS"
```

`$PARASURF_WEIGHTS` is already set inside the image, so you can pass it as-is.

## 📤 Publish to Docker Hub (maintainers)

```bash
docker login
docker build -t antisite:latest .
docker tag antisite:latest angepapa/antisite:latest
docker push angepapa/antisite:latest

# optional: also publish an immutable version tag for citation
docker tag antisite:latest angepapa/antisite:1.0
docker push angepapa/antisite:1.0
```
