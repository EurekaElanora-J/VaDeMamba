# Reproducibility Guide: VaDeMamba

**Paper Title**: *VaDeMamba: Learning the Value of Mamba Depth for Spatially Adaptive 3D Brain Tumor Segmentation*

This document distinguishes public code inspection from authorized reproduction
using retained sealed artifacts. The public clone does not contain patient data,
weight binaries, or patient-derived final-test outputs.

---

## 1. System Specifications & Environment

The experiments were executed on the following validated environment:
- **Operating System**: Windows-11-10.0.26200-SP0
- **Python Version**: 3.12.7
- **PyTorch Version**: 2.5.1
- **CUDA Device**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA Active: True)
- **Host CPU**: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- **System Memory**: 15.69 GB RAM

---

## 2. Environment Setup

### Using Conda (Recommended):
```bash
conda env create -f environment.yml
conda activate vademamba
```

### Using Pip:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 3. Data Splits & Cohort Partitions

All 40 scans from 28 subjects are strictly partitioned by patient identity into three subject-disjoint cohorts (`data/splits/cohort_split_manifest.json`). Longitudinal scans are retained within their subject's assigned cohort; the sealed final-test cohort is never used for training, predictor selection, or threshold tuning:
1. **BACKBONE_TRAIN** (14 subjects, 20 scans): Used solely for optimizing the 3-stage Mamba backbone (`checkpoints/main_backbone_best.pt`).
2. **PREDICTOR_CASCADE_DEV** (6 subjects, 8 scans): Used for feature development, cross-validation, and fitting the frozen deployment depth-value gates.
3. **UNTOUCHED_FINAL_TEST** (8 subjects, 12 scans): Held out entirely until the final evaluation campaign. Zero retraining or threshold sweeps were performed on this cohort.

> **Data Redistribution Notice**:
> The raw brain MRI volumes originate from the **ASNR-MICCAI BraTS 2023 Challenge** dataset. Due to challenge data use agreements, raw medical scans (`.nii.gz`) cannot be redistributed directly within this repository. To reproduce experiments involving inference or training from scratch, download the training data from the BraTS Challenge portal and place it in `data/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`.

---

## 4. Reproduction Workflows

### Mode A: Inspecting Final Manuscript Tables from Retained Artifacts (No GPU or BraTS Required)

The submitted manuscript's Table 6 controlled comparison and Table 8 mechanism
validation are supported by separately retained sealed artifacts. The public
clone excludes those artifacts. With authorized access, use the read-only
reporter; it prints to stdout and does not rerun the 12-scan campaign:

```bash
python experiments/report_final_manuscript_artifacts.py \
  --controlled-json path/to/reviewer_raw_metrics.json \
  --gate1-csv path/to/mech_k12_tile_data.csv \
  --gate2-csv path/to/mech_k23_tile_data.csv
```

The reporter loads the four Table 6 methods (Full $K_3$, Entropy, MSP, and
VaDeMamba 4/2) directly from the controlled-comparison artifact. It keeps
analytical GMAC, measured synchronized wall-clock latency, peak GPU memory, and
segmentation quality as distinct reported quantities.

For Table 8 it calculates descriptive association/ranking statistics from the
retained tile records: Gate 1 is continuous $\Delta E_{1\to2}$ prediction and
Gate 2 is ordinal 16-D RankNet ordering, not calibrated $\Delta E$ regression.
The in-memory adapter accepts historical `AdaDepth_*` columns only where they
are relevant to old per-scan schema; it never rewrites sealed files.

The retained tile schemas do not encode the original 10,000-permutation
protocol. The reporter therefore does not recompute or approximate those
manuscript p-values; they remain sealed retained evidence.

`generate_final_figures_and_tables.py` is preserved as a legacy
retained-artifact generator. It is not the reporting path for final-manuscript
Tables 6 and 8 because its historical latency fields predate the controlled
comparison.

---

### Mode B: Reproducing Inference and Evaluation from Provided Checkpoints (Requires BraTS Data & GPU)
To evaluate the sealed final test cohort using pretrained checkpoints obtained separately from normal Git distribution:

1. **Verify Checkpoint Integrity**:
   Confirm that all checkpoints match the locked SHA-256 hashes listed in Section 5.
2. **Execute Single-Pass Test Campaign**:
   ```bash
   python experiments/run_final_test_campaign.py
   ```
   This script evaluates all 12 test scans across Fixed $K_1$, Fixed $K_2$, Fixed $K_3$, Matched Random 4/2 (5 seeds), Proposed VaDeMamba (4/2), Ablations C & D, and the Retrospective 4/2 Oracle. It is a sealed-campaign workflow and must not be rerun merely to inspect reported final results.
3. **Regenerate Presentation Artifacts**:
   ```bash
   python experiments/generate_final_figures_and_tables.py
   ```

---

### Mode C: Full Pipeline Reproduction from Scratch (Requires BraTS Data & GPU)
To train the entire pipeline from scratch:

1. **Train 3-Stage Backbone**:
   ```bash
   python experiments/train_main_backbone.py
   ```
   Trains for 10 epochs using coupled deep supervision loss $\mathcal{L}_{\text{train}} = \frac{1}{3} \sum_{k=1}^3 [\text{CE}_k + \text{SoftDice}_k]$ on `BACKBONE_TRAIN`. Checkpoint selected by DEV Mean Dice.
2. **Harvest DEV Macro-Tile Features**:
   ```bash
   python experiments/harvest_experiment2_dev_features.py
   ```
3. **Evaluate Gate 2 Candidates on DEV Out-Of-Fold Folds**:
   ```bash
   python experiments/evaluate_gate2_candidates_dev_oof.py
   ```
4. **Fit Hardened Deployment Predictors on DEV Cohort**:
   ```bash
   python experiments/train_hardened_deployment_predictors.py
   ```
5. **Run Final Test Campaign**:
   ```bash
   python experiments/run_final_test_campaign.py
   ```
6. **Generate Legacy Tables and Figures**:
   ```bash
   python experiments/generate_final_figures_and_tables.py
   ```

   For final-manuscript Tables 6 and 8, use Mode A's reporter against the
   retained controlled-comparison and mechanism artifacts instead.

---

## 5. Authoritative Checkpoint Hashes

| Artifact | Filepath | SHA-256 Checksum | Description |
| :--- | :--- | :--- | :--- |
| **Backbone Checkpoint** | `checkpoints/main_backbone_best.pt` | `e1177279c69dd5272dbf2ad075f20017f6b2b009cafd47200f68e2cc4f442133` | Epoch 10 3-Stage Modular Mamba Backbone |
| **Gate 1 Predictor** | `checkpoints/gate1_deployment.pt` | `7594f0cef779660b724c23270039699b062f90cdefe60b407b275ed47e3dafeb` | 7-D Family-B1 Huber MLP |
| **Gate 2 Predictor** | `checkpoints/gate2_deployment.pt` | `71a6fbd9313963fb5eb573da8a573a375764096be464378069b77c9a533d9009` | 16-D $Z_2$ Linear Pairwise RankNet |
| **Gate 1 Complexity Ablation** | `checkpoints/gate1_complexity_deployment.pt` | `e353f3334659360c224e019dba572aca59542af9ec57a488098a41bebe2b0a78` | 3-D Complexity Huber MLP (Ablation C) |
| **Gate 2 Baseline Ablation** | `checkpoints/gate2_baseline_deployment.pt` | `3b59a5884720824f9cd479982f4ad45b25e2a00011596b11affa20b3856f6f25` | 7-D $P_2$-Only Ridge Baseline (Ablation D) |
| **Deployment Manifest** | `checkpoints/deployment_predictors_manifest.json` | `1bbfbe448f46fc2d3685ff5831df0942e8421bb0c2986469009b0f82a80d1450` | Feature definitions and metadata |
| **Cohort Split Manifest** | `data/splits/cohort_split_manifest.json` | `06977b71667c39a640269f996d9a7d22a8822071264325ff9be518ebf415cfd6` | Subject-disjoint partition mapping |
| **Final Test Results** | `results/final_campaign/final_test_per_scan_results.csv` | `85e84a96f9e28fe5ba176cf9cbe80cb81d1b43d99371daed45e79401e3813b9e` | Primary per-scan test outcomes |
| **Final Test Allocations** | `results/final_campaign/final_test_tile_allocations.csv` | `1cdd1e2a0eb342ec230a4bb15cd3a50fcd9af507a6d5d029a34074155c3d6eeb` | Primary tile allocation records |
