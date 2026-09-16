"""
Feature & Target Harvesting Script for Experiment 2 (PREDICTOR_CASCADE_DEV only)
--------------------------------------------------------------------------------
Extracts candidate feature vectors and continuous marginal benefit targets:
- Gate 1: z1(r) = [Family A, Family B1, Family C1] -> Target Delta_E_12
- Gate 2: z2(r) = [Family A, Family B2, Family C2] -> Target Delta_E_23

Strict Temporal Enforcement:
- Gate 1 inputs use ONLY information available after Exit 1 and BEFORE Block 2.
- Gate 2 inputs use ONLY information available after Exit 2 and BEFORE Block 3.

Primary scale: 16^3 macro-regions.
Secondary scale: 8^3 micro-regions.
FINAL_TEST is strictly off-limits.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import load_and_preprocess_patient
from src.data.patch_sampler import BraTSPatchDataset
from src.models.modular_backbone import ModularMambaBackbone
from src.metrics.brier import compute_voxelwise_brier_map
from src.complexity.estimators import compute_gradient_complexity, compute_frequency_complexity, compute_intensity_entropy

def extract_predictive_vector(prob_subvol):
    """
    prob_subvol: (4, S, S, S) float tensor
    Returns 7-dim vector:
      [U_entropy, U_margin, P_BG, P_NCR, P_ED, P_ET, P_max]
    """
    eps = 1e-7
    # Shannon entropy: - sum_c p_c * log(p_c)
    entropy_map = -torch.sum(prob_subvol * torch.log(prob_subvol + eps), dim=0)
    u_entropy = float(entropy_map.mean().item())
    
    # Top-margin: p_(1) - p_(2)
    top2_vals, _ = torch.topk(prob_subvol, k=2, dim=0)
    margin_map = top2_vals[0] - top2_vals[1]
    u_margin = float(margin_map.mean().item())
    
    # Mean class posteriors
    mean_p = prob_subvol.mean(dim=(1, 2, 3)).cpu().numpy() # [P_BG, P_NCR, P_ED, P_ET]
    
    # Max probability
    p_max = float(top2_vals[0].mean().item())
    
    return [u_entropy, u_margin, float(mean_p[0]), float(mean_p[1]), float(mean_p[2]), float(mean_p[3]), p_max]

def harvest_features_for_scale(dev_dataset, model, device, region_size=16):
    print(f"\nHarvesting features at scale {region_size}x{region_size}x{region_size}...")
    stride_ratio = region_size // 4 # Stem downsamples by 4
    num_regions_per_dim = 128 // region_size
    
    records = []
    
    with torch.no_grad():
        for b_idx in range(len(dev_dataset)):
            batch = dev_dataset[b_idx]
            p_id = batch["patient_id"]
            s_id = '-'.join(p_id.split('-')[:3])
            
            x = batch["image"].unsqueeze(0).to(device) # (1, 4, 128, 128, 128)
            y = batch["seg"].unsqueeze(0).to(device).long() # (1, 128, 128, 128)
            brain_mask = batch["brain_mask"] # (128, 128, 128)
            
            out = model(x)
            
            # Latent representations (1, 96, 32, 32, 32)
            f1 = out["features"]["f1"].squeeze(0)
            f2 = out["features"]["f2"].squeeze(0)
            
            # Predictions (1, 4, 128, 128, 128)
            p1 = out["probs"]["exit1"].squeeze(0)
            p2 = out["probs"]["exit2"].squeeze(0)
            p3 = out["probs"]["exit3"].squeeze(0)
            
            # Voxelwise Brier maps (128, 128, 128)
            e1_map = compute_voxelwise_brier_map(p1.unsqueeze(0), y).squeeze(0)
            e2_map = compute_voxelwise_brier_map(p2.unsqueeze(0), y).squeeze(0)
            e3_map = compute_voxelwise_brier_map(p3.unsqueeze(0), y).squeeze(0)
            
            # Iterate over 3D grid
            for d in range(num_regions_per_dim):
                for h in range(num_regions_per_dim):
                    for w in range(num_regions_per_dim):
                        # Voxel coordinates
                        d0, d1 = d * region_size, (d + 1) * region_size
                        h0, h1 = h * region_size, (h + 1) * region_size
                        w0, w1 = w * region_size, (w + 1) * region_size
                        
                        # Latent coordinates (stride 4)
                        ld0, ld1 = d * stride_ratio, (d + 1) * stride_ratio
                        lh0, lh1 = h * stride_ratio, (h + 1) * stride_ratio
                        lw0, lw1 = w * stride_ratio, (w + 1) * stride_ratio
                        
                        # Subvolumes
                        img_sub = x[0, :, d0:d1, h0:h1, w0:w1]
                        seg_sub = y[0, d0:d1, h0:h1, w0:w1]
                        brain_sub = brain_mask[d0:d1, h0:h1, w0:w1]
                        
                        # Regional Brier & Targets
                        e1_val = float(e1_map[d0:d1, h0:h1, w0:w1].mean().item())
                        e2_val = float(e2_map[d0:d1, h0:h1, w0:w1].mean().item())
                        e3_val = float(e3_map[d0:d1, h0:h1, w0:w1].mean().item())
                        
                        delta_e_12 = e1_val - e2_val # Target for Gate 1
                        delta_e_23 = e2_val - e3_val # Target for Gate 2
                        delta_e_13 = e1_val - e3_val
                        
                        # Tissue composition
                        f_brain = float((brain_sub > 0).float().mean().item())
                        f_tumor = float((seg_sub > 0).float().mean().item())
                        
                        # Family A: Intrinsic Handcrafted Complexity (3 dims)
                        c_grad = compute_gradient_complexity(img_sub)
                        c_freq = compute_frequency_complexity(img_sub)
                        c_entropy = compute_intensity_entropy(img_sub)
                        
                        # Family B1: Predictive Information at Exit 1 (7 dims)
                        p1_sub = p1[:, d0:d1, h0:h1, w0:w1]
                        pred_vec1 = extract_predictive_vector(p1_sub)
                        
                        # Family B2: Predictive Information at Exit 2 (7 dims)
                        p2_sub = p2[:, d0:d1, h0:h1, w0:w1]
                        pred_vec2 = extract_predictive_vector(p2_sub)
                        
                        # Family C1: Learned Features at Exit 1 (96 dims - spatial mean pool)
                        f1_sub = f1[:, ld0:ld1, lh0:lh1, lw0:lw1]
                        f1_mean = f1_sub.mean(dim=(1, 2, 3)).cpu().numpy().tolist() # 96 dims
                        
                        # Family C2: Learned Features at Exit 2 (96 dims - spatial mean pool)
                        f2_sub = f2[:, ld0:ld1, lh0:lh1, lw0:lw1]
                        f2_mean = f2_sub.mean(dim=(1, 2, 3)).cpu().numpy().tolist() # 96 dims
                        
                        rec = {
                            "scan_id": p_id,
                            "subject_id": s_id,
                            "region_size": region_size,
                            "coord_d": d, "coord_h": h, "coord_w": w,
                            "f_brain": f_brain,
                            "f_tumor": f_tumor,
                            "E_1": e1_val, "E_2": e2_val, "E_3": e3_val,
                            "Delta_E_12": delta_e_12,
                            "Delta_E_23": delta_e_23,
                            "Delta_E_13": delta_e_13,
                            # Family A
                            "C_grad": c_grad,
                            "C_freq": c_freq,
                            "C_entropy": c_entropy,
                            # Family B1
                            "U1_entropy": pred_vec1[0], "U1_margin": pred_vec1[1],
                            "P1_BG": pred_vec1[2], "P1_NCR": pred_vec1[3], "P1_ED": pred_vec1[4], "P1_ET": pred_vec1[5],
                            "P1_max": pred_vec1[6],
                            # Family B2
                            "U2_entropy": pred_vec2[0], "U2_margin": pred_vec2[1],
                            "P2_BG": pred_vec2[2], "P2_NCR": pred_vec2[3], "P2_ED": pred_vec2[4], "P2_ET": pred_vec2[5],
                            "P2_max": pred_vec2[6],
                        }
                        
                        # Append 96 latent features for Exit 1
                        for ch in range(96):
                            rec[f"F1_{ch:02d}"] = f1_mean[ch]
                            
                        # Append 96 latent features for Exit 2
                        for ch in range(96):
                            rec[f"F2_{ch:02d}"] = f2_mean[ch]
                            
                        records.append(rec)
                        
            print(f"  Processed scan {p_id} ({b_idx+1}/{len(dev_dataset)}) -> {len(records)} regions accumulated.", flush=True)
            
    df = pd.DataFrame(records)
    return df

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"HARVESTING EXPERIMENT 2 DEV FEATURES ON {device}")
    print("=" * 80)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(project_root, "data", "raw", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    splits_dir = os.path.join(project_root, "data", "splits")
    ckpt_path = os.path.join(project_root, "checkpoints", "main_backbone_best.pt")
    results_dir = os.path.join(project_root, "results", "experiment2_prediction")
    os.makedirs(results_dir, exist_ok=True)

    # Load Locked Manifest
    with open(os.path.join(splits_dir, "cohort_split_manifest.json"), "r") as f:
        manifest = json.load(f)

    dev_scans = manifest["cohorts"]["PREDICTOR_CASCADE_DEV"]["scans"]
    print(f"Verified PREDICTOR_CASCADE_DEV cohort ({len(dev_scans)} scans): {dev_scans}")

    # Load Checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing main backbone checkpoint at: {ckpt_path}. Train the backbone first!")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Loaded Main Backbone Checkpoint: Epoch {checkpoint['epoch']} with Best MeanDice: {checkpoint['best_mean_dice']:.4f}")

    # Load Model
    model = ModularMambaBackbone(in_channels=4, hidden_dim=96, num_classes=4, d_state=16).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load Dev Patients
    dev_patients = [load_and_preprocess_patient(os.path.join(data_root, s), s) for s in dev_scans]
    dev_dataset = BraTSPatchDataset(dev_patients, patch_size=(128, 128, 128), samples_per_patient=1, is_train=False)

    # 1. Harvest Primary Scale (16x16)
    df16 = harvest_features_for_scale(dev_dataset, model, device, region_size=16)
    csv16_path = os.path.join(results_dir, "dev_features_16x16.csv")
    df16.to_csv(csv16_path, index=False)
    print(f"Saved primary 16x16 features to: {csv16_path} (Shape: {df16.shape})")

    # 2. Harvest Secondary Scale (8x8)
    df8 = harvest_features_for_scale(dev_dataset, model, device, region_size=8)
    csv8_path = os.path.join(results_dir, "dev_features_8x8.csv")
    df8.to_csv(csv8_path, index=False)
    print(f"Saved secondary 8x8 features to: {csv8_path} (Shape: {df8.shape})")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"FEATURE HARVESTING COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 80)

if __name__ == "__main__":
    main()
