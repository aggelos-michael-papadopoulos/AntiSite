# AntiSite — fully self-contained GPU image.
# Everything (code, ParaSurf native stack, ALL model weights) is baked in, so an
# end user pulls the image and runs predictions with ZERO downloads.
#
# Baked weights:
#   - AntiSite checkpoints            (release/checkpoints/*.pt)          ~110 MB
#   - ParaSurf (Paragraph-expanded)   (docker/weights/*.pth)             ~186 MB
#   - ProtT5 half  (Rostlab/prot_t5_xl_half_uniref50-enc)               ~2.3 GB
#   - ESM-2 650M   (esm2_t33_650M_UR50D)                                ~2.5 GB
#
# Build (needs internet once, ~20-40 min, ~25 GB transient disk):
#   sudo docker build -t antisite:latest .
# Run the offline smoke test (both modes) on the bundled sample:
#   sudo docker run --rm --gpus all antisite:latest bash smoke_test.sh
# Sequence-only prediction:
#   sudo docker run --rm --gpus all antisite:latest python antisite.py \
#     --weights release/checkpoints/antisite_paragraph.pt -vh <HEAVY> -vl <LIGHT>
# 3D prediction on your own structure (mount it in):
#   sudo docker run --rm --gpus all -v $PWD:/data antisite:latest python antisite.py \
#     --weights release/checkpoints/antisite_paragraph.pt --antibody /data/ab.pdb \
#     --parasurf-weights $PARASURF_WEIGHTS

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# --- system tools: git/wget + build toolchain for the ParaSurf DMS surface program ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

# --- Miniforge (conda) — used only for OpenBabel, which ParaSurf needs for chemical features ---
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p $CONDA_DIR \
    && rm /tmp/mf.sh
RUN $CONDA_DIR/bin/conda create -y -n antisite python=3.10 \
    && $CONDA_DIR/bin/conda install -y -n antisite -c conda-forge openbabel \
    && $CONDA_DIR/bin/conda clean -afy
# put the env's python/pip first so every following command uses it (no `conda run` needed)
ENV PATH=$CONDA_DIR/envs/antisite/bin:$PATH

WORKDIR /app

# --- Python deps: pinned CUDA torch first, then the AntiSite package (+ 3D extras) ---
COPY pyproject.toml README.md LICENSE /app/
COPY antisite /app/antisite
COPY antisite.py /app/antisite.py
COPY test_antisite/infer_3d.py /app/test_antisite/infer_3d.py
RUN pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -e ".[threed]"

# --- ParaSurf native stack (clone repo, build DMS; pdb2pqr ships inside the repo) ---
# NOTE: we deliberately do NOT run ParaSurf/requirements.txt (it pins an older torch).
RUN git clone --depth 1 https://github.com/aggelos-michael-papadopoulos/ParaSurf.git /app/ParaSurf \
    && (cd /app/ParaSurf/dms && make install)

# --- bake ALL weights into the image ---
COPY release/checkpoints /app/release/checkpoints
COPY docker/weights/Paragraph_expanded_entire_dataset_best.pth /app/ParaSurf/
# ProtT5 (half) -> HuggingFace cache; ESM-2 650M -> torch hub cache. Both land in image layers.
RUN python -c "from transformers import T5EncoderModel, T5Tokenizer; \
T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False); \
T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')" \
    && python -c "import esm; esm.pretrained.esm2_t33_650M_UR50D()"

# --- bundled sample + offline smoke test ---
COPY docker/sample /app/sample
COPY docker/smoke_test.sh /app/smoke_test.sh

ENV PARASURF_WEIGHTS=/app/ParaSurf/Paragraph_expanded_entire_dataset_best.pth \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# default: print the CLI help
CMD ["python", "antisite.py", "--help"]
