# Public Artifact Curation

This note classifies local artifacts for a future VaDeMamba research-code release.
It does not alter any sealed result or dataset artifact.

## PUBLIC-CANDIDATE

- `results/final_campaign/tables/*.csv` and `*.md`: publication-support tables,
  subject to the authors' final review of their aggregate and per-scan content.
- Non-patient-identifying aggregate figures in `results/final_campaign/figures/`:
  publication-support graphics, subject to final review.
- `data/splits/cohort_split_manifest.json`: cohort metadata and split provenance;
  it contains identifiers but no NIfTI volume data.

## REQUIRES DATASET-LICENSE REVIEW

- Qualitative final-test figures, especially segmentation/image panels: they may
  be derived from BraTS data even though they are not raw NIfTI volumes.
- `results/final_campaign/final_test_per_scan_results.csv` and
  `final_test_tile_allocations.csv`: sealed, patient-/scan-level derived records
  with historical `AdaDepth_*` output fields.

## PRIVATE / DO NOT PUBLISH

- `data/raw/`: BraTS patient volumes and all NIfTI data.
- `results/final_campaign/visual_slice_cache.pt`: cached image slices and masks.
- `results/experiment2_prediction/dev_features_*.csv`, oracle `.npz` caches,
  pilot outputs, diagnostics, and workspace directory `brain/`: generated
  development data, derived caches, or local system artifacts.
- `checkpoints/*.pt`: distribute separately with documented hashes if released.

## Retained-final schema compatibility

The sealed retained final CSV uses `AdaDepth_*` field names and the retained
visual cache uses `pred_adadepth`. The corrected active generator expects
`VaDeMamba_*` and `pred_vademamba`. A future compatibility layer may safely
map these names in memory when reading historical artifacts, but it must not
rewrite the sealed CSV, cache, or any scientific value.
