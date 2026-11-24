# ZymeBind

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.24492676-blue.svg)](https://zenodo.org/records/24492676)

**ZymeBind** is a presence-aware multimodal deep learning framework for enzyme Commission (EC) number annotation, zero-shot unseen-EC4 retrieval, and multi-level mechanistic interpretability.

---

## Overview

Real-world enzyme annotation routinely encounters incomplete input modalities (for example, uncharacterized metagenomic sequences lacking experimental structures, literature descriptions, or InterPro annotations). ZymeBind addresses this through four coordinated architectural components:

1. **Dual Protein Language Model Sequence Representation**: Concatenates residue-level representations from `ESM-2 650M` (1,280-d) and `ProstT5-3B` (1,024-d) into a 2,304-d sequence embedding, optionally paired with Foldseek 3Di structural tokens (`ProstT5-3B`), BioBERT reaction descriptions, and InterPro/Pfam domain signatures.
2. **Presence-Aware Modality Fusion**: Detects absent modalities explicitly and masks them with $-\infty$ attention weights (`src_key_padding_mask`), preventing zero-vector dilution during cross-modal self-attention.
3. **Modality-Specific Auxiliary Heads & Presence-Weighted Ensemble Prediction**: Combines a fused multimodal head with four modality-specific auxiliary heads via a presence-gated softmax ensemble, trained under structured modality dropout, cross-modal knowledge distillation, Asymmetric Loss (ASL), and exponential moving average (EMA) weight smoothing.
4. **Two-Stage Semantic Ontology Alignment & L1/L2/L3 Interpretability**:
   - **Stage 1 (Zero-Shot EC Retrieval)**: Aligns enzyme embeddings to BioBERT-initialized, two-layer EC DAG graph convolutional prototypes on an $\ell_2$-normalized hypersphere for unseen fourth-level EC4 retrieval.
   - **L1 / L2 / L3 Mechanistic Interpretability**: Residue-level catalytic-site masking and Integrated Gradients (L1), exact $2^4 = 16$ coalition Shapley modality attribution (L2), and Top-$K$ Sparse Autoencoder (SAE) catalytic motif discovery (L3).

---

## Installation

### Option 1: Conda / Mamba (recommended for full pipeline including Foldseek & MMseqs2)

```bash
git clone https://github.com/xiaoniao007-debug/ZymeBind.git
cd ZymeBind
conda env create -f environment.yml
conda activate zymebind
```

### Option 2: Pip (editable install with CLI)

First install PyTorch matching your CUDA toolkit (see [pytorch.org](https://pytorch.org/get-started/locally/)), then install `zymebind`:

```bash
git clone https://github.com/xiaoniao007-debug/ZymeBind.git
cd ZymeBind
pip install -e ".[gpu]"
# Optional: include interpretability and plotting tools
pip install -e ".[gpu,interpret]"
```

### Option 3: Docker (GPU / CPU containerized deployment)

The included `Dockerfile` bundles PyTorch 2.5.1 (CUDA 12.1), `torch-geometric`, `transformers`, `fair-esm`, `foldseek`, `mmseqs2`, and the `zymebind` CLI:

```bash
git clone https://github.com/xiaoniao007-debug/ZymeBind.git
cd ZymeBind

# Build image (prefix with sudo if your user is not in the docker group)
docker build -t zymebind:1.0.0 .

# Verify container environment
docker run --rm --gpus all zymebind:1.0.0 info

# Run EC prediction on a FASTA file (mounts local checkpoints, ontology, and output folder)
mkdir -p outputs
docker run --rm --gpus all \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/examples:/app/examples:ro" \
  -v "$(pwd)/outputs:/app/outputs" \
  zymebind:1.0.0 predict \
    --fasta /app/examples/sample_enzymes.fasta \
    --ckpt /app/checkpoints/zymebind_stage2.pt \
    --vocab /app/data/processed/ec_vocab.json \
    --out /app/outputs/predictions.tsv
```

For CPU-only execution, omit `--gpus all` and add `--device cpu`. You can also run via Docker Compose (`docker compose run --rm zymebind info`).

Verify the installation:

```bash
zymebind info
```

---

## Pretrained Checkpoints & Vocabulary

Pretrained model checkpoints (`zymebind_stage2_seed42.pt`, `zymebind_stage2_seed2026.pt`, `zymebind_l3_sae.pt`), the 6,162-node EC ontology package (`zymebind_ontology.tar.gz`), 40%-cluster-disjoint dataset splits and benchmarks (`zymebind_splits_and_benchmarks.tar.gz`), and machine-readable source tables (`zymebind_source_tables.tar.gz`) are archived on Zenodo:

- **Zenodo Record**: https://zenodo.org/records/24492676 (DOI: `10.5281/zenodo.24492676`)

After downloading the release files from Zenodo, verify checksums and place them in the repository directory:

```bash
# 1. Verify file integrity
shasum -a 256 -c SHA256SUMS.txt

# 2. Place checkpoints and extract EC ontology
mkdir -p checkpoints data/processed
cp zymebind_stage2_seed42.pt checkpoints/zymebind_stage2.pt
tar -xzf zymebind_ontology.tar.gz -C data/processed/

# 3. (Optional) Extract training/validation/test splits and external benchmarks
tar -xzf zymebind_splits_and_benchmarks.tar.gz -C .
```

Expected local layout (or pass explicit paths via `--ckpt` and `--vocab`):

```text
checkpoints/
└── zymebind_stage2.pt
data/processed/
└── ec_vocab.json
```

---

## Quick Start (CLI)

### 1. Predict EC numbers from a FASTA file (Sequence-Only mode)

Uses `ESM-2 650M + ProstT5-3B` sequence embeddings with native missing-modality masking for absent structure, text, and domain inputs:

```bash
zymebind predict \
  --fasta examples/sample_enzymes.fasta \
  --ckpt checkpoints/zymebind_stage2.pt \
  --vocab data/processed/ec_vocab.json \
  --mode sequence \
  --threshold 0.62 \
  --topk 5 \
  --out predictions.tsv
```

To export structured JSON instead of TSV, pass `--format json --out predictions.json`.

### 2. Predict EC numbers with Foldseek 3Di structures (Seq + Structure mode)

If you have Foldseek 3Di strings (FASTA or two-column TSV `protein_id<TAB>3di_sequence`), pass `--threedi` and `--mode seq_struct`:

```bash
zymebind predict \
  --fasta examples/sample_enzymes.fasta \
  --threedi examples/sample_3di.fasta \
  --ckpt checkpoints/zymebind_stage2.pt \
  --vocab data/processed/ec_vocab.json \
  --mode seq_struct \
  --out predictions_seq_struct.tsv
```

### 3. Zero-Shot Unseen-EC4 Retrieval (Stage 1 Hypersphere)

Retrieve nearest fourth-level EC4 ontology prototypes for enzymes whose EC4 class may not have been observed during supervised training:

```bash
zymebind zeroshot \
  --fasta examples/sample_enzymes.fasta \
  --ckpt checkpoints/zymebind_stage2.pt \
  --vocab data/processed/ec_vocab.json \
  --topk 10 \
  --out zeroshot_hits.json
```

### 4. Mechanistic Interpretability (L1 / L2 / L3)

```bash
# Level 1: Residue-level active-site masking
zymebind explain --level l1_masking --help

# Level 1: Integrated Gradients residue attribution
zymebind explain --level l1_ig --help

# Level 2: Exact 2^4 = 16 coalition Shapley modality attribution
zymebind explain --level l2_shapley --help

# Level 3: Top-K Sparse Autoencoder (SAE) training & motif analysis
zymebind explain --level l3_sae --help
```

### 5. Training from Cached Multimodal Embeddings

```bash
zymebind train --help
```

---

## Repository Structure

```text
ZymeBind/
├── Dockerfile                     # GPU/CPU container image (PyTorch 2.5.1 + CUDA 12.1 + Foldseek + MMseqs2)
├── docker-compose.yml             # Docker Compose service with GPU reservation and volume mounts
├── pyproject.toml                 # Package metadata & `zymebind` CLI entry point
├── requirements.txt               # Pip requirements
├── environment.yml                # Conda environment (Python 3.10 + Foldseek + MMseqs2)
├── configs/                       # Default data and model YAML configurations
├── examples/
│   └── sample_enzymes.fasta       # Example enzyme sequences (TEM-1, DHFR, TIM)
├── zymebind/
│   ├── cli.py                     # `zymebind` CLI (`info`, `predict`, `zeroshot`, `train`, `explain`)
│   ├── models/                    # Presence-aware fusion, EC ontology GNN, ASL, MoE projection
│   ├── data/                      # FASTA/3Di I/O, EC DAG ontology, cached multimodal datasets
│   ├── inference/                 # Supervised EC prediction and zero-shot EC4 retrieval
│   ├── training/                  # Two-stage training pipeline (Stage 1 contrastive + Stage 2 EMA/KD/ASL)
│   ├── interpretability/          # L1 active-site masking/IG, L2 Shapley, L3 Top-K SAE
│   └── evaluation/                # CAFA hierarchical metrics, Fmax/AUPR/Macro-F1, multi-view evaluation
├── scripts/
│   ├── reproducibility/           # Blind-test, metagenome, partial-EC, DMS, and pathway-hole scripts
│   └── figures/                   # Main/Supplementary figure and LaTeX table generation scripts
└── tests/
    └── test_cli_and_model.py      # CLI and I/O unit tests
```

---

## Authors & Contact

- **Chen Chen**\* (`nxsfcc@nxnu.edu.cn`) — School of Mathematics and Computer Science, Ningxia Normal University, Guyuan, China
- **Hongbo Zhou** — College of Teacher Education, Ningxia University, Yinchuan, China
- **Zishang Gao** — School of Mathematics and Computer Science, Ningxia Normal University, Guyuan, China
- **Qifu Li** — School of Mathematics and Computer Science, Ningxia Normal University, Guyuan, China
- **Yanshan Tian**\* (`tianysh@nxnu.edu.cn`) — School of Mathematics and Computer Science, Ningxia Normal University, Guyuan, China

\* Corresponding authors.

---

## Citation

If you use ZymeBind in your research, please cite:

```bibtex
@article{chen2026zymebind,
  title   = {ZymeBind: Presence-Aware Multimodal Learning for Enzyme Function Prediction, Zero-Shot EC Retrieval, and Mechanistic Interpretability},
  author  = {Chen, Chen and Zhou, Hongbo and Gao, Zishang and Li, Qifu and Tian, Yanshan},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
