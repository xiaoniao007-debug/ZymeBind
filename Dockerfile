# ZymeBind GPU/CPU Docker Image
# Base: PyTorch 2.5.1 + CUDA 12.1 + cuDNN 9 (compatible with NVIDIA drivers >= 530)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL org.opencontainers.image.title="ZymeBind" \
      org.opencontainers.image.description="Presence-aware multimodal enzyme EC prediction, zero-shot retrieval, and mechanistic interpretability" \
      org.opencontainers.image.source="https://github.com/xiaoniao007-debug/ZymeBind" \
      org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

WORKDIR /app

# System utilities and optional bioinformatics tools (curl, git, ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install Foldseek and MMseqs2 static Linux x86_64 binaries for 3Di extraction and homology search
RUN curl -fsSL https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz | tar -xz -C /opt && \
    ln -s /opt/foldseek/bin/foldseek /usr/local/bin/foldseek && \
    curl -fsSL https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz | tar -xz -C /opt && \
    ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs

# Install Python dependencies first for optimal Docker layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && \
    pip install \
        torch-geometric==2.6.1 \
        transformers==4.45.2 \
        tokenizers==0.20.1 \
        sentencepiece==0.2.0 \
        safetensors==0.4.5 \
        huggingface-hub==0.26.2 \
        fair-esm==2.0.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        scipy==1.14.1 \
        scikit-learn==1.5.2 \
        pyarrow==17.0.0 \
        h5py==3.12.1 \
        biopython==1.84 \
        networkx==3.4.1 \
        pyyaml==6.0.2 \
        tqdm==4.66.5 \
        requests==2.32.3 \
        captum==0.7.0 \
        shap==0.46.0 \
        matplotlib==3.9.2 \
        seaborn==0.13.2

# Copy package source and install `zymebind` CLI
COPY README.md LICENSE ./
COPY configs ./configs
COPY examples ./examples
COPY zymebind ./zymebind
RUN pip install --no-deps -e .

RUN mkdir -p /cache/huggingface /cache/torch /app/checkpoints /app/data/processed /app/outputs

ENTRYPOINT ["zymebind"]
CMD ["info"]
