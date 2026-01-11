# syntax=docker/dockerfile:1.4
# Standalone scientific computing image (CPU)
#
# Build: docker build -f Dockerfile.standalone -t scientific-env .
# Run:   docker run -it -p 8888:8888 -v $(pwd):/work scientific-env

FROM ubuntu:22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update -qq && \
    apt-get install -yq --no-install-recommends \
    util-linux \
    git \
    openssh-client \
    wget \
    gpg \
    software-properties-common \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN --mount=type=cache,target=/root/.cache/miniconda \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /app/miniconda && \
    rm ~/miniconda.sh && \
    /app/miniconda/bin/conda init bash

# Override base image python with miniconda python
RUN ln -sf /app/miniconda/bin/python /usr/local/bin/python && \
    ln -sf /app/miniconda/bin/python3 /usr/local/bin/python3 && \
    ln -sf /app/miniconda/bin/pip /usr/local/bin/pip && \
    ln -sf /app/miniconda/bin/pip3 /usr/local/bin/pip3

ENV VIRTUAL_ENV="/app/miniconda"
ENV PATH="/app/miniconda/bin:$PATH"
ENV PYTHONPATH="/app/miniconda/lib/python3.12/site-packages:${PYTHONPATH:-}"

# Install uv and mamba
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir uv==0.8.19
RUN conda install -c conda-forge mamba==2.3.2 -y

# Create kernel environment with all analysis packages
RUN conda create -p /app/kernel_env python=3.12 -y

# Install R packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
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
            r-hdf5r=1.3.11

# Install core Python scientific stack
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            numpy=1.26.4 \
            pandas=2.3.2 \
            scipy=1.16.2 \
            scikit-learn=1.7.2 \
            matplotlib=3.10.6 \
            seaborn=0.13.2 \
            plotly=6.3.0 \
            openpyxl=3.1.5 \
            jupyter=1.1.1 \
            ipykernel=6.30.1 \
            nbconvert=7.16.6

# Install ML/optimization packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            keras=3.11.2 \
            optuna=4.5.0 \
            imbalanced-learn=0.14.0 \
            lightgbm=4.6.0 \
            statsmodels=0.14.5

# Install bioinformatics Python packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            anndata=0.12.2 \
            scanpy=1.11.4 \
            biopython=1.85 \
            muon=0.1.6 \
            umap-learn=0.5.9.post2 \
            leidenalg=0.10.2 \
            python-igraph=0.11.9

# Install visualization and utility packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            matplotlib-venn=1.1.2 \
            ete3=3.1.3 \
            fcsparser=0.2.8 \
            datasets=2.2.1 \
            udocker=1.3.17 \
            sqlite=3.50.4

# Install chemistry packages
RUN mamba install -p /app/kernel_env -c conda-forge -y \
            rdkit=2025.09.2 \
            pubchempy=1.0.5 \
            chempy=0.10.1

# Install bioinformatics tools
RUN mamba install -p /app/kernel_env -c conda-forge -c bioconda -y \
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
            gatk=3.8

# Install pytorch (CPU)
RUN /app/kernel_env/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Jupyter kernels
RUN /app/kernel_env/bin/python -m ipykernel install --name python3 --display-name "Python 3 (ipykernel)" && \
    export PATH="/app/kernel_env/bin:$PATH" && \
    /app/kernel_env/bin/R -e 'IRkernel::installspec(user = FALSE, name = "ir", displayname = "R")'

# Install kernel server dependencies
RUN /app/kernel_env/bin/pip install --no-cache-dir fastapi uvicorn

# Clean up conda caches
RUN mamba clean -all -y && \
    find /app/miniconda \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + || true && \
    find /app/miniconda -type f -name '*.a' -delete && \
    find /app/miniconda -type f -name '*.js.map' -delete && \
    find /app/kernel_env \( -type d -name __pycache__ -o -type d -name tests -o -type d -name '*.tests' -o -type d -name 'test' \) -exec rm -rf {} + || true && \
    find /app/kernel_env -type f -name '*.a' -delete && \
    find /app/kernel_env -type f -name '*.js.map' -delete

# Copy kernel server for Docker-based execution
COPY src/hypotest/env/kernel_server.py /envs/kernel_server.py

WORKDIR /workspace
EXPOSE 8000

CMD ["/app/kernel_env/bin/python", "/envs/kernel_server.py", "--work_dir", "/workspace"]
