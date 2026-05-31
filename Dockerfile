# syntax=docker/dockerfile:1.4
# Multi-stage scientific-computing / kernel-execution image for hypotest.
#
# Stages
#   core - Ubuntu + Miniconda + core Python scientific stack + the Python Jupyter
#          kernel + the standalone kernel server. Arch-aware (amd64 + arm64), so
#          it builds natively on Apple Silicon for local testing. Lightweight.
#   full - core + the full R / bioconda / chemistry / bioinformatics stack and
#          CPU PyTorch. amd64 only (bioconda has poor linux-aarch64 coverage).
#          This is the production kernel/exec image (formerly the only image).
#
# Supply-chain cutoff: network installs are bounded to BUILD_CUTOFF_DATE so no
# package published after the cutoff is pulled (defense vs Shai-Hulud-style
# attacks on popular packages).
#   - pip  -> `uv pip install --exclude-newer ${BUILD_CUTOFF_DATE}`
#   - apt  -> Ubuntu archive snapshot (snapshot.ubuntu.com) at the cutoff
#   - conda/mamba -> bounded by a build-time repodata-filtering proxy
#                    (FOLLOW-UP: not yet wired; conda installs below are still
#                    exact-pinned, which fixes direct deps but not transitive).
#   - torch -> exact-pinned (its wheel index lacks upload-time metadata, so
#              --exclude-newer can't bound it; see the full stage).
#
# Build via the Makefile (`make image` / `make image-core`), which measures the
# cutoff date and passes it as a build-arg. Building this Dockerfile directly
# falls back to "today" for the cutoff.

ARG BUILD_CUTOFF_DATE

# =============================================================================
# core: lightweight, arch-aware base (amd64 + arm64)
# =============================================================================
FROM ubuntu:22.04 AS core

ARG BUILD_CUTOFF_DATE
ARG TARGETARCH

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies pinned to a dated Ubuntu archive snapshot so no .deb newer
# than the cutoff is installed. snapshot.ubuntu.com serves the main archive
# (amd64) but NOT ubuntu-ports, so the sed below applies on amd64 (the full/prod
# image) and no-ops on arm64 — the arm64 core image (local testing only) falls
# through to the live ports archive. ca-certificates is installed first (from the
# default archive) so apt can verify TLS for the HTTPS snapshot (which 301-redirects
# from http); that one bootstrap package is the only un-dated apt fetch.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    CUTOFF="${BUILD_CUTOFF_DATE:-$(date -u +%Y-%m-%d)}"; \
    SNAP_TS="$(date -u -d "${CUTOFF} 00:00:00" +%Y%m%dT000000Z)"; \
    apt-get update -qq; \
    apt-get install -yq --no-install-recommends ca-certificates; \
    sed -i -E "s|https?://(archive\|security)\.ubuntu\.com/ubuntu|https://snapshot.ubuntu.com/ubuntu/${SNAP_TS}|g" \
        /etc/apt/sources.list; \
    apt-get -o Acquire::Check-Valid-Until=false update -qq; \
    apt-get install -yq --no-install-recommends \
        util-linux \
        git \
        openssh-client \
        wget \
        gpg \
        software-properties-common \
        build-essential; \
    rm -rf /var/lib/apt/lists/*

# Miniconda (arch-aware: x86_64 on amd64, aarch64 on arm64).
RUN --mount=type=cache,target=/root/.cache/miniconda \
    set -eux; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "$arch" in \
        amd64|x86_64)  MC_ARCH=x86_64 ;; \
        arm64|aarch64) MC_ARCH=aarch64 ;; \
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    wget -q "https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-${MC_ARCH}.sh" \
        -O /tmp/miniconda.sh; \
    bash /tmp/miniconda.sh -b -p /app/miniconda; \
    rm /tmp/miniconda.sh; \
    /app/miniconda/bin/conda init bash

# Override base-image python with the miniconda python.
RUN ln -sf /app/miniconda/bin/python /usr/local/bin/python && \
    ln -sf /app/miniconda/bin/python3 /usr/local/bin/python3 && \
    ln -sf /app/miniconda/bin/pip /usr/local/bin/pip && \
    ln -sf /app/miniconda/bin/pip3 /usr/local/bin/pip3

ENV VIRTUAL_ENV="/app/miniconda"
ENV PATH="/app/miniconda/bin:$PATH"
ENV PYTHONPATH="/app/miniconda/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Install uv (exact-pinned: bootstraps the cutoff-aware installer itself).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir uv==0.8.19

# Conda installs, bounded to the build-date cutoff via the repodata-filtering
# proxy (conda has no native --exclude-newer). All conda/mamba work + the kernel
# registration + cache cleanup run in ONE layer, so package tarballs never
# persist across layers (keeps the image small). CONDA_REPODATA_USE_ZST/_FNS and
# the proxy's own 404s force the client onto the full, plain repodata.json we filter.
COPY docker/cutoff_proxy.py /opt/cutoff_proxy.py
RUN set -eux; \
    CUTOFF="${BUILD_CUTOFF_DATE:-$(date -u +%Y-%m-%d)}"; \
    python /opt/cutoff_proxy.py --port 8723 --cutoff "${CUTOFF}" & \
    PROXY=$!; \
    for _ in $(seq 1 50); do wget -q -O - http://127.0.0.1:8723/healthz >/dev/null 2>&1 && break; sleep 0.2; done; \
    export CONDA_REPODATA_USE_ZST=false CONDA_REPODATA_FNS=repodata.json; \
    CF="http://127.0.0.1:8723/conda-forge"; \
    conda install --override-channels -c "${CF}" mamba==2.3.2 -y; \
    conda create -p /app/kernel_env --override-channels -c "${CF}" python=3.12 -y; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
        numpy=1.26.4 pandas=2.3.2 scipy=1.16.2 scikit-learn=1.7.2 matplotlib=3.10.6 \
        seaborn=0.13.2 plotly=6.3.0 openpyxl=3.1.5 jupyter=1.1.1 ipykernel=6.30.1 nbconvert=7.16.6; \
    /app/kernel_env/bin/python -m ipykernel install --name python3 --display-name "Python 3 (ipykernel)"; \
    mamba clean --all -y; conda clean --all -y; \
    find /app/kernel_env /app/miniconda \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + 2>/dev/null || true; \
    kill "${PROXY}" 2>/dev/null || true

# Kernel server dependencies, bounded to the cutoff date.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eux; \
    CUTOFF="${BUILD_CUTOFF_DATE:-$(date -u +%Y-%m-%d)}"; \
    uv pip install --python /app/kernel_env/bin/python --exclude-newer "${CUTOFF}" \
        fastapi uvicorn

# Copy the standalone kernel server for container-based execution.
COPY src/hypotest/env/kernel_server.py /envs/kernel_server.py

# Put the conda env's shared libs on the loader path. conda-forge binaries
# (e.g. scipy) are built against the env's newer libstdc++/libgcc, not Ubuntu's
# system ones; without this, any process that runs kernel_env's python WITHOUT
# activating the env (the in-process kernel path, and the docker exec path) hits
# `GLIBCXX_3.4.x not found`. LD_LIBRARY_PATH is in REQUIRED_PATH_ENV_VARS, so the
# in-process Interpreter propagates it into the kernel subprocess. The enroot
# path runs `env -i` + `source activate`, so it is unaffected by this.
ENV LD_LIBRARY_PATH="/app/kernel_env/lib"

WORKDIR /workspace
EXPOSE 8000

# Default for the kernel/exec image is the standalone kernel server (the
# docker/enroot execution paths override this command / bind-mount the server,
# so they are unaffected; the bundled dataset-server image overrides CMD).
CMD ["/app/kernel_env/bin/python", "/envs/kernel_server.py", "--work_dir", "/workspace"]

# =============================================================================
# full: production stack (amd64 only) — adds R, bioconda, chemistry, torch
# =============================================================================
FROM core AS full

ARG BUILD_CUTOFF_DATE

# Full scientific stack (R / ML / bio / chem / bioinformatics), bounded to the
# build-date cutoff via the same proxy as core and built in ONE layer (cutoff +
# small image). bioconda is added for the bioinformatics tools; the proxy serves
# any /<channel> path.
RUN set -eux; \
    CUTOFF="${BUILD_CUTOFF_DATE:-$(date -u +%Y-%m-%d)}"; \
    python /opt/cutoff_proxy.py --port 8723 --cutoff "${CUTOFF}" & \
    PROXY=$!; \
    for _ in $(seq 1 50); do wget -q -O - http://127.0.0.1:8723/healthz >/dev/null 2>&1 && break; sleep 0.2; done; \
    export CONDA_REPODATA_USE_ZST=false CONDA_REPODATA_FNS=repodata.json; \
    CF="http://127.0.0.1:8723/conda-forge"; BIO="http://127.0.0.1:8723/bioconda"; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
            r-base=4.3.3 \
            r-r.utils=2.13.0 \
            r-recommended=4.3 \
            r-irkernel=1.3.2 \
            r-tidyverse=2.0.0 \
            r-readxl=1.4.5 \
            r-seurat=5.3.0 \
            rpy2=3.5.11 \
            r-factominer=2.12 \
            r-rcolorbrewer=1.1_3 \
            r-devtools=2.4.5 \
            r-broom=1.0.9 \
            r-data.table=1.17.8 \
            r-enrichr=3.4 \
            r-factoextra=1.0.7 \
            r-ggnewscale=0.5.2 \
            r-ggrepel=0.9.6 \
            r-ggpubr=0.6.1 \
            r-ggvenn=0.1.10 \
            r-janitor=2.2.1 \
            r-multcomp=1.4_28 \
            r-matrix=1.6_5 \
            r-pheatmap=1.0.13 \
            r-reshape=0.8.10 \
            r-rstatix=0.7.2 \
            r-viridis=0.6.5 \
            r-hdf5r=1.3.11; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
            keras=3.11.2 \
            optuna=4.5.0 \
            imbalanced-learn=0.14.0 \
            lightgbm=4.6.0 \
            statsmodels=0.14.5; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
            anndata=0.12.2 \
            scanpy=1.11.4 \
            biopython=1.85 \
            muon=0.1.6 \
            umap-learn=0.5.9.post2 \
            leidenalg=0.10.2 \
            python-igraph=0.11.9; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
            matplotlib-venn=1.1.2 \
            ete3=3.1.3 \
            fcsparser=0.2.8 \
            datasets=2.2.1 \
            udocker=1.3.17 \
            sqlite=3.50.4; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -y \
            rdkit=2025.09.2 \
            pubchempy=1.0.5 \
            chempy=0.10.1; \
    mamba install -p /app/kernel_env --override-channels -c "${CF}" -c "${BIO}" -y \
            biokit=0.5.0 \
            gseapy=1.1.10 \
            blast=2.17.0 \
            clipkit=2.6.1 \
            clustalo=1.2.4 \
            fastqc=0.12.1 \
            iqtree=3.0.1 \
            mafft=7.526 \
            metaeuk=7.bba0d80 \
            mygene=3.2.2 \
            perl=5.32.1 \
            phykit=2.0.3 \
            pydeseq2=0.5.2 \
            spades=4.2.0 \
            trim-galore=0.6.10 \
            harmonypy=0.0.10 \
            bioconductor-enhancedvolcano=1.20.0 \
            bioconductor-deseq2=1.42.0 \
            bioconductor-clusterprofiler=4.10.0 \
            bioconductor-org.hs.eg.db=3.18.0 \
            bioconductor-genomicranges=1.54.1 \
            bioconductor-summarizedexperiment=1.32.0 \
            bioconductor-apeglm=1.24.0 \
            bioconductor-flowcore=2.14.0 \
            bioconductor-flowmeans=1.62.0 \
            bioconductor-limma=3.58.1 \
            bioconductor-geoquery=2.70.0 \
            r-wgcna=1.73 \
            r-coloc=5.2.3 \
            r-susier=0.14.2 \
            r-mendelianrandomization=0.10.0 \
            r-ldlinkr=1.4.0 \
            r-arrow=13.0.0 \
            hmmer=3.4 \
            hhsuite=3.3.0 \
            mmseqs2=18.8cc5c \
            samtools=1.22.1 \
            gatk=3.8; \
    /app/kernel_env/bin/R -e 'IRkernel::installspec(user = FALSE, name = "ir", displayname = "R")'; \
    mamba clean --all -y; conda clean --all -y; \
    find /app/miniconda /app/kernel_env \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + 2>/dev/null || true; \
    find /app/miniconda /app/kernel_env -type f \( -name '*.a' -o -name '*.js.map' \) -delete 2>/dev/null || true; \
    kill "${PROXY}" 2>/dev/null || true

# PyTorch (CPU). Exact-pinned: the PyTorch wheel index does not expose upload-time
# metadata, so uv --exclude-newer cannot bound it; the version pin is the control.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/kernel_env/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
