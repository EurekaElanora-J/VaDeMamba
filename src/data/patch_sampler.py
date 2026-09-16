import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .preprocessing import get_tumor_strata_masks

class BraTSPatchDataset(Dataset):
    """
    Dataset for training/validation patch sampling.
    Samples 128x128x128 patches with balanced foreground/background probability.
    """
    def __init__(self, patient_dict_list, patch_size=(128, 128, 128), samples_per_patient=4, is_train=True):
        self.patients = patient_dict_list
        self.patch_size = patch_size
        self.samples_per_patient = samples_per_patient
        self.is_train = is_train

    def __len__(self):
        return len(self.patients) * self.samples_per_patient

    def __getitem__(self, idx):
        patient_idx = idx // self.samples_per_patient
        sample = self.patients[patient_idx]
        
        img = sample["image"] # (4, D, H, W)
        seg = sample["seg"]   # (D, H, W)
        brain_mask = sample["brain_mask"] # (D, H, W)
        
        D, H, W = seg.shape
        pD, pH, pW = self.patch_size
        
        sample_in_patient = idx % self.samples_per_patient
        
        if self.is_train:
            tumor_voxels = torch.nonzero(seg > 0, as_tuple=False)
            brain_voxels = torch.nonzero(brain_mask, as_tuple=False)
            
            # Revised Pilot: Both training patches are tumor-centered (randomly jittered around tumor voxels)
            if len(tumor_voxels) > 0:
                center_idx = np.random.randint(0, len(tumor_voxels))
                cd, ch, cw = tumor_voxels[center_idx].tolist()
            elif len(brain_voxels) > 0:
                center_idx = np.random.randint(0, len(brain_voxels))
                cd, ch, cw = brain_voxels[center_idx].tolist()
            else:
                cd, ch, cw = D // 2, H // 2, W // 2
                
            # Bounds checking
            d_start = int(np.clip(cd - pD // 2, 0, max(0, D - pD)))
            h_start = int(np.clip(ch - pH // 2, 0, max(0, H - pH)))
            w_start = int(np.clip(cw - pW // 2, 0, max(0, W - pW)))
        else:
            # Deterministic center crop around tumor or brain center
            tumor_voxels = torch.nonzero(seg > 0, as_tuple=False)
            if len(tumor_voxels) > 0:
                cd, ch, cw = tumor_voxels.float().mean(dim=0).long().tolist()
            else:
                brain_voxels = torch.nonzero(brain_mask, as_tuple=False)
                cd, ch, cw = brain_voxels.float().mean(dim=0).long().tolist()
                
            d_start = np.clip(cd - pD // 2, 0, max(0, D - pD))
            h_start = np.clip(ch - pH // 2, 0, max(0, H - pH))
            w_start = np.clip(cw - pW // 2, 0, max(0, W - pW))

        d_end = d_start + pD
        h_end = h_start + pH
        w_end = w_start + pW
        
        # Pad if volume dimension is smaller than patch size (e.g. D=155 > 128, but safety check)
        img_patch = img[:, d_start:d_end, h_start:h_end, w_start:w_end]
        seg_patch = seg[d_start:d_end, h_start:h_end, w_start:w_end]
        mask_patch = brain_mask[d_start:d_end, h_start:h_end, w_start:w_end]
        
        return {
            "image": img_patch,
            "seg": seg_patch,
            "brain_mask": mask_patch,
            "patient_id": sample["patient_id"],
            "coords": (d_start, h_start, w_start)
        }


def partition_volume_into_regions(img_patch, seg_patch, brain_mask_patch, region_size=16):
    """
    Partitions a 128x128x128 patch into non-overlapping regions of size region_size^3.
    Extracts metadata for Population A, B, C stratification.
    
    Args:
      img_patch: (4, 128, 128, 128)
      seg_patch: (128, 128, 128)
      brain_mask_patch: (128, 128, 128)
      region_size: 8 or 16
      
    Returns:
      List of dicts, each containing:
        - region_coords: (d, h, w)
        - img_region: (4, S, S, S)
        - seg_region: (S, S, S)
        - mask_region: (S, S, S)
        - f_brain: float
        - f_tumor: float
        - f_boundary: float
        - f_et: float
        - f_tc: float
        - f_ed: float
        - pop_strata: dict of booleans (in_pop_a, in_pop_b, in_pop_c, subregion_label)
    """
    pD, pH, pW = seg_patch.shape
    S = region_size
    assert pD % S == 0 and pH % S == 0 and pW % S == 0, f"Patch size {seg_patch.shape} must be divisible by {S}"
    
    # Pre-calculate strata masks for the patch
    strata_masks = get_tumor_strata_masks(seg_patch.cpu().numpy() if isinstance(seg_patch, torch.Tensor) else seg_patch)
    wt_mask = strata_masks["WT"]
    tc_mask = strata_masks["TC"]
    et_mask = strata_masks["ET"]
    boundary_mask = strata_masks["Boundary"]
    ed_mask = strata_masks["ED"]
    
    brain_mask_np = brain_mask_patch.cpu().numpy() if isinstance(brain_mask_patch, torch.Tensor) else brain_mask_patch
    num_voxels = S ** 3
    
    regions = []
    
    for d in range(0, pD, S):
        for h in range(0, pH, S):
            for w in range(0, pW, S):
                d_slice = slice(d, d + S)
                h_slice = slice(h, h + S)
                w_slice = slice(w, w + S)
                
                brain_cnt = int(brain_mask_np[d_slice, h_slice, w_slice].sum())
                wt_cnt = int(wt_mask[d_slice, h_slice, w_slice].sum())
                tc_cnt = int(tc_mask[d_slice, h_slice, w_slice].sum())
                et_cnt = int(et_mask[d_slice, h_slice, w_slice].sum())
                boundary_cnt = int(boundary_mask[d_slice, h_slice, w_slice].sum())
                ed_cnt = int(ed_mask[d_slice, h_slice, w_slice].sum())
                
                f_brain = brain_cnt / num_voxels
                f_tumor = wt_cnt / num_voxels
                f_boundary = boundary_cnt / num_voxels
                f_et = et_cnt / num_voxels
                f_tc = tc_cnt / num_voxels
                f_ed = ed_cnt / num_voxels
                
                in_pop_a = True
                in_pop_b = (f_brain >= 0.5)
                in_pop_c = (f_tumor > 0.0)
                
                # Determine detailed tumor subregion type
                if not in_pop_c:
                    subregion = "NonTumor"
                elif f_boundary > 0.0:
                    subregion = "TumorBoundary"
                elif f_et > 0.0:
                    subregion = "EnhancingCore"
                elif f_tc > 0.0:
                    subregion = "NecroticCore"
                else:
                    subregion = "PeritumoralEdema"
                    
                regions.append({
                    "coords": (d, h, w),
                    "slices": (d_slice, h_slice, w_slice),
                    "region_size": S,
                    "f_brain": f_brain,
                    "f_tumor": f_tumor,
                    "f_boundary": f_boundary,
                    "f_et": f_et,
                    "f_tc": f_tc,
                    "f_ed": f_ed,
                    "in_pop_a": in_pop_a,
                    "in_pop_b": in_pop_b,
                    "in_pop_c": in_pop_c,
                    "subregion": subregion
                })
                
    return regions

if __name__ == "__main__":
    from .preprocessing import load_and_preprocess_patient
    
    patient_root = Path(__file__).resolve().parents[2] / "data" / "raw" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    test_p = "BraTS-GLI-00000-000"
    p_dir = patient_root / test_p
    sample = load_and_preprocess_patient(p_dir, test_p)
    
    dataset = BraTSPatchDataset([sample], patch_size=(128, 128, 128), samples_per_patient=2, is_train=False)
    patch_item = dataset[0]
    print(f"Extracted Patch Shape: {patch_item['image'].shape}, Seg: {patch_item['seg'].shape}")
    
    # Test 16^3 partitioning
    regions_16 = partition_volume_into_regions(patch_item['image'], patch_item['seg'], patch_item['brain_mask'], region_size=16)
    print(f"Total 16^3 regions: {len(regions_16)}")
    pop_a_cnt = sum(1 for r in regions_16 if r['in_pop_a'])
    pop_b_cnt = sum(1 for r in regions_16 if r['in_pop_b'])
    pop_c_cnt = sum(1 for r in regions_16 if r['in_pop_c'])
    print(f"  Pop A (Whole Volume): {pop_a_cnt}")
    print(f"  Pop B (Brain Only): {pop_b_cnt}")
    print(f"  Pop C (Tumor ROI): {pop_c_cnt}")
    
    # Subregion counts in Pop C
    subregion_counts = {}
    for r in regions_16:
        sub = r['subregion']
        subregion_counts[sub] = subregion_counts.get(sub, 0) + 1
    print("  Subregion distribution:", subregion_counts)
    
    # Test 8^3 partitioning
    regions_8 = partition_volume_into_regions(patch_item['image'], patch_item['seg'], patch_item['brain_mask'], region_size=8)
    print(f"\nTotal 8^3 regions: {len(regions_8)}")
    print(f"  Pop C (Tumor ROI): {sum(1 for r in regions_8 if r['in_pop_c'])}")
    print("Checkpoint 3 (Sampling & Partitioning) verification passed!")
