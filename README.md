# VaDeMamba: Learning the Value of Mamba Depth for Spatially Adaptive 3D Brain Tumor Segmentation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange)](requirements.txt)

Official implementation and reproducibility release for:  
**"VaDeMamba: Learning the Value of Mamba Depth for Spatially Adaptive 3D Brain Tumor Segmentation"**

---

## 1. What is VaDeMamba?

**VaDeMamba** (*Value-of-Depth Mamba*) is a spatially adaptive deep sequential architecture for 3D volumetric medical image segmentation. 

While State Space Models (Mamba) achieve linear computational complexity $O(L)$ with sequence length, standard deep backbones apply uniform computational depth everywhere. In 3D brain MRI volumes, large spatial regions consist of homogeneous healthy tissue or empty background where deep state transitions yield diminishing returns.

Instead of uniform computation, VaDeMamba estimates the **marginal value of an additional Mamba depth stage** spatially before executing it, dynamically allocating complete Mamba blocks only where deeper computation improves segmentation quality.

---

## 2. Key Architecture & Progressive Exits

The architecture consists of **three sequential computational depths ($K=1, 2, 3$)** within **one single backbone**:

```
Input X (4 x 128^3)
   │
   ▼
Two-Stage Conv3D Stem (Stride 4)
   │
   ▼
F0 (96 x 32^3, L = 32,768)
   │
   ▼
Global Block 1 (Always executed globally across all 8 macro-tiles)
   │
   ├── Exit 1: P1 = H(F1)
   │
   ├── Gate 1 (7-D Huber MLP): Evaluates P1 uncertainty & class distributions
   │           Selects top M1=4 tiles for Block 2 (S1)
   │
   ▼
Block 2 (Group-Batched Mamba): Executed ONLY on S1 tiles; unselected tiles retain F1
   │
   ├── Exit 2: P2 = H(F2)
   │
   ├── Gate 2 (16-D Z2 Pairwise RankNet): Evaluates depth-response residual features
   │           Selects top M2=2 tiles from within S1 for Block 3 (S2 ⊂ S1)
   │
   ▼
Block 3 (Group-Batched Mamba): Executed ONLY on S2 tiles; unselected tiles retain F2
   │
   ├── Exit 3: P3 = H(F3)
   │
   ▼
Shared Readout Head H: Conv3D 1x1x1 + Trilinear Upsampling (Shared across all exits)
```

### Key Architectural Properties
- **Cumulative Sequential Depths**: $K=1, 2, 3$ are sequential exits within ONE model, not separately trained networks.
- **Strictly Shared Readout**: Head $H$ is identical across all exits; no exit-specific parameters.
- **Complete Block Execution**: Tiling is executed at macro-scale ($16^3$ latent octants); every tile selected for a block receives the **complete, full-depth bidirectional Mamba block**. Zero compute is expended on bypassed tiles.
- **Subset Invariant**: $S_2 \subset S_1$ unconditionally.

---

## 3. Deployment Budget & Analytical Compute

Under the canonical deployment configuration:
- 4 tiles remain at $K=1$
- 2 tiles receive $K=2$ processing
- 2 tiles receive full $K=3$ processing

### Analytical GMAC Breakdown
- **Fixed Full-Depth $K_3$**: 9.773 GMAC
- **VaDeMamba 4/2**: 7.973 GMAC
- **Whole-Model GMAC Reduction**: **18.4%** relative to full-depth $K_3$ (50% stage reduction at Block 2; 75% stage reduction at Block 3).
- **Wall-Clock Latency Disclosure**: An isolated timing benchmark stalled during CUDA initialization; therefore, **no valid end-to-end wall-clock latency is claimed**. Compute claims are strictly analytical GMAC.

---

## 4. Summary of Empirical Findings (Held-Out Test Cohort, N=12 Scans / 8 Subjects)

All test results originate from the sealed, untouched test cohort:
- **Segmentation Quality**: VaDeMamba achieves **0.6842 $\pm$ 0.1496 Mean Dice**, outperforming full-depth $K_3$ (0.6738 $\pm$ 0.1552) and fixed $K_1$ (0.6790 $\pm$ 0.1504).
- **Head-to-Head Win Rate**: VaDeMamba outperforms matched random allocation on **10 of 12 scans (83.3%)**.
- **Quality-Compute Gap**: Closes **55.1%** of the gap between matched random allocation and a retrospective 4/2 oracle.
- **Multiclass Brier Error**: Retained test error is **0.008464** for VaDeMamba vs 0.008691 for $K_3$ (unretained comparator values are labeled `Not retained`).

---

## 5. Dataset & Cohort Partitions

Experiments use the **BraTS 2023 Adult Glioma** dataset across 28 unique biological subjects (40 scans). All partitions are strictly **subject-disjoint**:
- **`BACKBONE_TRAIN`** (14 subjects / 20 scans): Optimizing backbone weights.
- **`PREDICTOR_CASCADE_DEV`** (6 subjects / 8 scans): Feature extraction, Gate-2 candidate selection (OOF CV), and predictor fitting.
- **`UNTOUCHED_FINAL_TEST`** (8 subjects / 12 scans): Held-out sealed benchmark cohort.

> **Data Access**: Patient NIfTI volumes cannot be redistributed in this repository under challenge data use agreements. Users should obtain data directly from the BraTS Challenge portal and place it under `data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`.

---

## 6. Quick Reproduction

See [`docs/reproducibility.md`](docs/reproducibility.md) for workflow context and locked cohort/checkpoint provenance.

### Available directly from this public repository

The public clone supports environment setup, inspection of the active source and experiment scripts, configuration/split-manifest review, and the lightweight synthetic test suite. It does not include raw patient data, trained weight binaries, or sealed final-test outputs.

```bash
conda env create -f environment.yml
conda activate vademamba
pytest tests/
```

The test suite uses synthetic tensors and does not require BraTS data, checkpoints, or a GPU. The active implementation is in `src/`, with the final evaluation procedure documented in `experiments/run_final_test_campaign.py`.

### Requires separately approved artifacts

The following cannot currently be reproduced directly from the public clone:

- Held-out final-test inference and exact final-test reruns require authorized BraTS data and released trained checkpoints.
- Final publication tables and figures require retained sealed final-test artifacts; `python experiments/generate_final_figures_and_tables.py` is not runnable against the public clone alone.

This separation is intentional: raw BraTS volumes are omitted for dataset licensing/access considerations, checkpoint binaries are not committed to normal Git history, and sealed patient-derived final-test artifacts are internal unless separately released after appropriate review.

### Internal/approved-artifact workflows

With the necessary data, checkpoint, and sealed-artifact access approved, the scripts below define the corresponding workflows:

```bash
python experiments/run_final_test_campaign.py
python experiments/generate_final_figures_and_tables.py
```

---

## 7. Repository Structure

```
VaDeMamba/
├── checkpoints/
│   └── deployment_predictors_manifest.json  # SHA-256 provenance; no weight binaries
├── configs/                  # Deployment and model configurations
├── data/
│   └── splits/               # Subject-disjoint cohort split manifest
├── docs/
│   └── reproducibility.md
├── experiments/              # Audited active experiment/reproduction scripts
├── results/
│   └── codebase_audit/
│       └── public_artifact_curation.md
├── src/
├── tests/                    # Lightweight synthetic tests
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── environment.yml
└── requirements.txt
```

Raw BraTS data, checkpoint `.pt`/`.pth` binaries, workspace traces, quarantine/historical material, development features, caches, and sealed final-test outputs are intentionally omitted from the public repository.

---

## 8. License

This research codebase is licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
