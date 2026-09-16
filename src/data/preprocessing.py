import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion

def zscore_normalize_channel(img_data):
    """
    Z-score normalization over non-zero brain voxels.
    Background voxels (0) remain 0.
    Outliers clipped to [-5.0, 5.0].
    """
    mask = img_data > 0
    if not np.any(mask):
        return np.zeros_like(img_data, dtype=np.float32)
    
    mean = np.mean(img_data[mask])
    std = np.std(img_data[mask])
    
    if std < 1e-6:
        normalized = img_data - mean
    else:
        normalized = (img_data - mean) / std
        
    normalized[~mask] = 0.0
    normalized = np.clip(normalized, -5.0, 5.0)
    return normalized.astype(np.float32)

def extract_brain_bbox(brain_mask, margin=4):
    """
    Finds the 3D bounding box of brain mask with optional margin.
    """
    coords = np.argwhere(brain_mask)
    if coords.size == 0:
        return (0, brain_mask.shape[0]), (0, brain_mask.shape[1]), (0, brain_mask.shape[2])
    
    d_min, h_min, w_min = coords.min(axis=0)
    d_max, h_max, w_max = coords.max(axis=0) + 1
    
    d_min = max(0, d_min - margin)
    h_min = max(0, h_min - margin)
    w_min = max(0, w_min - margin)
    
    d_max = min(brain_mask.shape[0], d_max + margin)
    h_max = min(brain_mask.shape[1], h_max + margin)
    w_max = min(brain_mask.shape[2], w_max + margin)
    
    return (d_min, d_max), (h_min, h_max), (w_min, w_max)

def get_tumor_strata_masks(seg_data):
    """
    Given multi-class segmentation (0: BG, 1: NCR, 2: ED, 3: ET),
    generates subregion masks:
      - WT: Whole Tumor (1, 2, 3)
      - TC: Tumor Core (1, 3)
      - ET: Enhancing Tumor (3)
      - Boundary: Morphological contour of WT (dilation - erosion)
      - Core_Interior: TC eroded / inner core
    """
    wt = (seg_data > 0).astype(np.uint8)
    tc = np.isin(seg_data, [1, 3]).astype(np.uint8)
    et = (seg_data == 3).astype(np.uint8)
    
    if np.any(wt):
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        wt_dilated = binary_dilation(wt, structure=structure).astype(np.uint8)
        wt_eroded = binary_erosion(wt, structure=structure).astype(np.uint8)
        boundary = (wt_dilated ^ wt_eroded).astype(np.uint8)
    else:
        boundary = np.zeros_like(wt, dtype=np.uint8)
        
    return {
        "WT": wt,
        "TC": tc,
        "ET": et,
        "Boundary": boundary,
        "ED": (seg_data == 2).astype(np.uint8),
        "NCR": (seg_data == 1).astype(np.uint8)
    }

def load_and_preprocess_patient(patient_dir, patient_id):
    """
    Loads 4 MRI modalities + segmentation for a single patient,
    applies z-score normalization per channel, and returns:
      - modalities: torch.Tensor of shape (4, D, H, W)
      - seg: torch.Tensor of shape (D, H, W), int64 in {0, 1, 2, 3}
      - brain_mask: torch.Tensor of shape (D, H, W), bool
    """
    modality_keys = ['t1n', 't1c', 't2w', 't2f']
    channels = []
    
    for mod in modality_keys:
        path = f"{patient_dir}/{patient_id}-{mod}.nii.gz"
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        norm_data = zscore_normalize_channel(data)
        channels.append(norm_data)
        
    stacked = np.stack(channels, axis=0) # (4, D, H, W)
    
    seg_path = f"{patient_dir}/{patient_id}-seg.nii.gz"
    seg_img = nib.load(seg_path)
    seg_data = np.round(seg_img.get_fdata()).astype(np.int64) # {0, 1, 2, 3}
    
    brain_mask = (stacked[0] != 0) | (stacked[1] != 0) | (stacked[2] != 0) | (stacked[3] != 0)
    
    return {
        "patient_id": patient_id,
        "image": torch.from_numpy(stacked),
        "seg": torch.from_numpy(seg_data),
        "brain_mask": torch.from_numpy(brain_mask),
        "affine": seg_img.affine
    }

if __name__ == "__main__":
    patient_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    test_p = "BraTS-GLI-00000-000"
    p_dir = patient_root / test_p
    
    print(f"Testing preprocessing on {test_p}...")
    sample = load_and_preprocess_patient(p_dir, test_p)
    
    print("Image shape:", sample["image"].shape)
    print("Seg shape:", sample["seg"].shape)
    print("Brain mask voxels:", int(sample["brain_mask"].sum()))
    print("Modality means inside brain:", [float(sample["image"][c][sample["brain_mask"]].mean()) for c in range(4)])
    print("Modality stds inside brain:", [float(sample["image"][c][sample["brain_mask"]].std()) for c in range(4)])
    
    strata = get_tumor_strata_masks(sample["seg"].numpy())
    print("Strata voxel counts:")
    for k, v in strata.items():
        print(f"  {k}: {int(v.sum())} voxels")
    print("Checkpoint 2 (Preprocessing) verification passed!")
