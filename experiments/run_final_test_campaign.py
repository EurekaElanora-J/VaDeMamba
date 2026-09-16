"""
VaDeMamba: Learning the Value of Mamba Depth for Spatially Adaptive 3D Brain Tumor Segmentation
----------------------------------------------------------------------
Executes the final, publication-grade evaluation on UNTOUCHED_FINAL_TEST (12 scans, 8 subjects).
Zero retraining, zero hyperparameter tuning, zero architectural modifications.

Evaluates:
  1. Fixed K1 (all 8 tiles at K1)
  2. Fixed K2 (all 8 tiles at K2)
  3. Fixed K3 (all 8 tiles at K3)
  4. Random 4/2 Routing (5 deterministic seeds: 42, 43, 44, 45, 46)
  5. Gate-1-Only Ablation (4 tiles K2, 0 tiles K3)
  6. Gate-1 + Random Gate-2 (4 tiles G1, 2 random G2, 5 seeds)
  7. Gate-1 Feature Ablation (Complexity-only G1 + proposed G2)
  8. Gate-2 Information Ablation (Proposed G1 + P2-only G2)
  9. Proposed VaDeMamba Cascade (Proposed G1 + G2-D Linear RankNet G2)
 10. Retrospective Oracle 4/2 (actual Delta_E_12 top 4, actual Delta_E_23 top 2)

Saves all per-scan metrics, tile records, and visual slice caches to results/final_campaign/.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import load_and_preprocess_patient
from src.data.patch_sampler import BraTSPatchDataset
from src.models.modular_backbone import ModularMambaBackbone
from src.metrics.brier import compute_voxelwise_brier_map
from src.complexity.estimators import compute_gradient_complexity, compute_frequency_complexity, compute_intensity_entropy
from experiments.train_hardened_deployment_predictors import PredictorMLP, LinearRankNet

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
SPLITS_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "cohort_split_manifest.json")
BACKBONE_CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "main_backbone_best.pt")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "final_campaign")

RANDOM_SEEDS = [42, 43, 44, 45, 46]
M1 = 4
M2 = 2

# Locked GMAC constants
GMAC_STEM = 5.44
GMAC_HEAD = 0.013
GMAC_BLOCK = 1.44

def calculate_gmac(m1, m2):
    return GMAC_STEM + GMAC_HEAD + GMAC_BLOCK + (m1 / 8.0) * GMAC_BLOCK + (m2 / 8.0) * GMAC_BLOCK


def partition_latent_tiles(features):
    """Partition one 96 x 32 x 32 x 32 latent tensor into eight 16^3 tiles."""
    if features.shape != (1, 96, 32, 32, 32):
        raise ValueError(f"Expected (1, 96, 32, 32, 32), got {tuple(features.shape)}")
    return features.view(1, 96, 2, 16, 2, 16, 2, 16).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(8, 96, 16, 16, 16)


def reassemble_latent_tiles(tiles):
    """Reassemble eight 16^3 tiles into one 96 x 32 x 32 x 32 latent tensor."""
    if tiles.shape != (8, 96, 16, 16, 16):
        raise ValueError(f"Expected (8, 96, 16, 16, 16), got {tuple(tiles.shape)}")
    return tiles.view(1, 2, 2, 2, 96, 16, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(1, 96, 32, 32, 32)


def apply_block_to_selected_tiles(tiles, selected_tiles, block):
    """Apply a complete Mamba block only to selected, unique indices in [0, 7]."""
    selected_tiles = list(selected_tiles)
    if len(set(selected_tiles)) != len(selected_tiles) or any(i < 0 or i >= 8 for i in selected_tiles):
        raise ValueError("Tile selections must be unique indices in [0, 7].")
    routed = tiles.clone()
    if selected_tiles:
        routed[selected_tiles] = block(tiles[selected_tiles])
    return routed

def compute_all_metrics(probs, seg_gt):
    """Computes Brier and BraTS Dice metrics (WT, TC, ET, Mean)."""
    brier_map = compute_voxelwise_brier_map(probs, seg_gt)
    brier_val = float(brier_map.mean().item())
    
    pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
    gt_mask = seg_gt.squeeze(0).cpu().numpy()
    
    # WT: classes 1, 2, 3
    wt_pred = (pred_mask > 0)
    wt_gt = (gt_mask > 0)
    wt_dice = 2.0 * np.sum(wt_pred & wt_gt) / (np.sum(wt_pred) + np.sum(wt_gt) + 1e-8) if (np.sum(wt_pred) + np.sum(wt_gt)) > 0 else 1.0
    
    # TC: classes 1, 3
    tc_pred = (pred_mask == 1) | (pred_mask == 3)
    tc_gt = (gt_mask == 1) | (gt_mask == 3)
    tc_dice = 2.0 * np.sum(tc_pred & tc_gt) / (np.sum(tc_pred) + np.sum(tc_gt) + 1e-8) if (np.sum(tc_pred) + np.sum(tc_gt)) > 0 else 1.0
    
    # ET: class 3
    et_pred = (pred_mask == 3)
    et_gt = (gt_mask == 3)
    et_dice = 2.0 * np.sum(et_pred & et_gt) / (np.sum(et_pred) + np.sum(et_gt) + 1e-8) if (np.sum(et_pred) + np.sum(et_gt)) > 0 else 1.0
    
    mean_dice = (wt_dice + tc_dice + et_dice) / 3.0
    return {
        "brier": brier_val,
        "wt_dice": float(wt_dice),
        "tc_dice": float(tc_dice),
        "et_dice": float(et_dice),
        "mean_dice": float(mean_dice),
        "pred_mask": pred_mask,
        "brier_map": brier_map.squeeze(0).cpu().numpy()
    }

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"STARTING HARDENED FINAL TEST CAMPAIGN ON {device.upper()}")
    print("=" * 80)

    # 1. Load Split Manifest
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SPLITS_PATH) as f:
        manifest = json.load(f)
    test_scans = manifest["cohorts"]["UNTOUCHED_FINAL_TEST"]["scans"]
    print(f"Loaded {len(test_scans)} final test scans across {len(manifest['cohorts']['UNTOUCHED_FINAL_TEST']['subjects'])} biological subjects.")

    # 2. Load Backbone Checkpoint
    checkpoint = torch.load(BACKBONE_CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded Backbone Checkpoint: Epoch {checkpoint['epoch']} (Dev All-Exit Mean Dice: {checkpoint['best_mean_dice']:.4f})")
    model = ModularMambaBackbone(in_channels=4, hidden_dim=96, num_classes=4, d_state=16).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Frozen Deployment Predictors
    g1_ckpt = torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "gate1_deployment.pt"), map_location=device, weights_only=False)
    g1_model = PredictorMLP(in_dim=g1_ckpt["in_dim"], hidden_dim=g1_ckpt["hidden_dim"]).to(device).eval()
    g1_model.load_state_dict(g1_ckpt["model_state_dict"])
    g1_mu = torch.tensor(g1_ckpt["mu"], device=device).float()
    g1_std = torch.tensor(g1_ckpt["std"], device=device).float()
    g1_scale = g1_ckpt["target_scale"]

    g1_comp_ckpt = torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "gate1_complexity_deployment.pt"), map_location=device, weights_only=False)
    g1_comp_model = PredictorMLP(in_dim=g1_comp_ckpt["in_dim"], hidden_dim=g1_comp_ckpt["hidden_dim"]).to(device).eval()
    g1_comp_model.load_state_dict(g1_comp_ckpt["model_state_dict"])
    g1_comp_mu = torch.tensor(g1_comp_ckpt["mu"], device=device).float()
    g1_comp_std = torch.tensor(g1_comp_ckpt["std"], device=device).float()

    g2_ckpt = torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "gate2_deployment.pt"), map_location=device, weights_only=False)
    g2_model = LinearRankNet(in_dim=16).to(device).eval()
    g2_model.load_state_dict(g2_ckpt["model_state_dict"])
    g2_mu = torch.tensor(g2_ckpt["mu"], device=device).float()
    g2_std = torch.tensor(g2_ckpt["std"], device=device).float()

    g2_base_ckpt = torch.load(os.path.join(PROJECT_ROOT, "checkpoints", "gate2_baseline_deployment.pt"), map_location=device, weights_only=False)
    g2_base_coef = torch.tensor(g2_base_ckpt["coef"], device=device).float()
    g2_base_intercept = float(g2_base_ckpt["intercept"])
    g2_base_mu = torch.tensor(g2_base_ckpt["mu"], device=device).float()
    g2_base_std = torch.tensor(g2_base_ckpt["std"], device=device).float()

    print("All deployment predictors loaded successfully.")

    # 4. Load Final Test Scans
    print("\nLoading and preprocessing 12 final test scans...")
    t0 = time.time()
    test_patients = [load_and_preprocess_patient(os.path.join(DATA_ROOT, s), s) for s in test_scans]
    test_dataset = BraTSPatchDataset(test_patients, patch_size=(128, 128, 128), samples_per_patient=1, is_train=False)
    print(f"Loaded all test scans in {time.time() - t0:.1f}s.")

    all_scan_results = []
    all_tile_results = []
    visual_cache = {}

    tile_coords = []
    for d in range(2):
        for h in range(2):
            for w in range(2):
                tile_coords.append((d, h, w))

    print("\n" + "=" * 80)
    print("BEGINNING COMPREHENSIVE FINAL EVALUATION ACROSS ALL 12 TEST SCANS")
    print("=" * 80)

    with torch.no_grad():
        for scan_idx, item in enumerate(test_dataset):
            scan_id = item["patient_id"]
            subj_id = '-'.join(scan_id.split('-')[:3])
            x = item["image"].unsqueeze(0).to(device) # (1, 4, 128, 128, 128)
            y = item["seg"].unsqueeze(0).to(device).long() # (1, 128, 128, 128)
            target_shape = (128, 128, 128)

            print(f"\nEvaluating Scan [{scan_idx+1}/{len(test_scans)}]: {scan_id} (Subject: {subj_id})...")

            # ---------------------------------------------------------
            # 1. FIXED DEPTH INFERENCE & TIMING
            # ---------------------------------------------------------
            # Fixed-depth reference evaluations.  These are not used as inputs to
            # the adaptive Gate-2 route below.
            f0 = model.stem(x)
            f1 = model.block1(f0)
            _, p1 = model.shared_head(f1, target_shape=target_shape)
            m_k1 = compute_all_metrics(p1, y)
            m_k1.update({"config": "Fixed K1", "m1": 0, "m2": 0, "gmac": calculate_gmac(0, 0)})

            # Fixed K2
            f1_tiles = partition_latent_tiles(f1)
            f2_full_tiles = model.block2(f1_tiles)
            f2_full = reassemble_latent_tiles(f2_full_tiles)
            _, p2_full = model.shared_head(f2_full, target_shape=target_shape)
            m_k2 = compute_all_metrics(p2_full, y)
            m_k2.update({"config": "Fixed K2", "m1": 8, "m2": 0, "gmac": calculate_gmac(8, 0)})

            # Fixed K3
            f3_full_tiles = model.block3(f2_full_tiles)
            f3_full = reassemble_latent_tiles(f3_full_tiles)
            _, p3_full = model.shared_head(f3_full, target_shape=target_shape)
            m_k3 = compute_all_metrics(p3_full, y)
            m_k3.update({"config": "Fixed K3", "m1": 8, "m2": 8, "gmac": calculate_gmac(8, 8)})

            # Ground truth regional errors for retrospective target evaluation
            e1_map = compute_voxelwise_brier_map(p1, y).squeeze(0)
            e2_map = compute_voxelwise_brier_map(p2_full, y).squeeze(0)
            e3_map = compute_voxelwise_brier_map(p3_full, y).squeeze(0)

            # ---------------------------------------------------------
            # 2. EXTRACT REGIONAL TILE PROPERTIES & TARGETS
            # ---------------------------------------------------------
            p1_tiles = p1.view(1, 4, 2, 64, 2, 64, 2, 64).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(8, 4, 64, 64, 64)

            # Vectorized Gate 1 Features
            eps = 1e-7
            entropy1 = -torch.sum(p1_tiles * torch.log(p1_tiles + eps), dim=1).mean(dim=(1, 2, 3))
            top2_1, _ = torch.topk(p1_tiles, k=2, dim=1)
            margin1 = (top2_1[:, 0] - top2_1[:, 1]).mean(dim=(1, 2, 3))
            mean_p1 = p1_tiles.mean(dim=(2, 3, 4))
            p1_max = top2_1[:, 0].mean(dim=(1, 2, 3))
            feats_g1 = torch.stack([entropy1, margin1, mean_p1[:, 0], mean_p1[:, 1], mean_p1[:, 2], mean_p1[:, 3], p1_max], dim=1)
            g1_preds_raw = g1_model((feats_g1 - g1_mu) / g1_std)
            pred_delta_e12 = (g1_preds_raw / g1_scale).cpu().numpy().flatten()
            s1_proposed = torch.topk(g1_preds_raw, k=M1).indices.tolist()

            # Adaptive path: only the four Gate-1 selections receive Block 2.
            # The full-depth K2 tensors above are retained solely for the fixed-K2
            # reference and retrospective oracle targets; they are never used by Gate 2.
            f2_proposed_tiles = apply_block_to_selected_tiles(f1_tiles, s1_proposed, model.block2)
            f2_proposed = reassemble_latent_tiles(f2_proposed_tiles)
            _, p2_proposed = model.shared_head(f2_proposed, target_shape=target_shape)
            p2_tiles = p2_proposed.view(1, 4, 2, 64, 2, 64, 2, 64).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(8, 4, 64, 64, 64)

            # Regional Ground-Truth Targets
            tile_records = []
            actual_delta_e12_list = []
            actual_delta_e23_list = []
            comp_feats_list = []

            for t_idx, (D, H, W) in enumerate(tile_coords):
                d0, d1 = D * 64, (D + 1) * 64
                h0, h1 = H * 64, (H + 1) * 64
                w0, w1 = W * 64, (W + 1) * 64
                img_sub = x[0, :, d0:d1, h0:h1, w0:w1]
                e1_val = float(e1_map[d0:d1, h0:h1, w0:w1].mean().item())
                e2_val = float(e2_map[d0:d1, h0:h1, w0:w1].mean().item())
                e3_val = float(e3_map[d0:d1, h0:h1, w0:w1].mean().item())
                d12 = e1_val - e2_val
                d23 = e2_val - e3_val
                actual_delta_e12_list.append(d12)
                actual_delta_e23_list.append(d23)

                cg = compute_gradient_complexity(img_sub)
                cf = compute_frequency_complexity(img_sub)
                ce = compute_intensity_entropy(img_sub)
                comp_feats_list.append([cg, cf, ce])

            # Gate 1 Complexity ranking (Ablation C)
            feats_comp = torch.tensor(comp_feats_list, device=device).float()
            g1_comp_scores = g1_comp_model((feats_comp - g1_comp_mu) / g1_comp_std)
            s1_comp = torch.topk(g1_comp_scores, k=M1).indices.tolist()

            # Oracle Gate 1 ranking
            s1_oracle = np.argsort(actual_delta_e12_list)[::-1][:M1].tolist()

            # Gate 2 Feature Extraction on S1 candidates
            p2_cand = p2_tiles[s1_proposed]
            p1_cand = p1_tiles[s1_proposed]
            u2_ent = -torch.sum(p2_cand * torch.log(p2_cand + eps), dim=1).mean(dim=(1, 2, 3))
            u1_ent = -torch.sum(p1_cand * torch.log(p1_cand + eps), dim=1).mean(dim=(1, 2, 3))
            top2_2, _ = torch.topk(p2_cand, k=2, dim=1)
            top1_1, _ = torch.topk(p1_cand, k=2, dim=1)
            u2_mar = (top2_2[:, 0] - top2_2[:, 1]).mean(dim=(1, 2, 3))
            u1_mar = (top1_1[:, 0] - top1_1[:, 1]).mean(dim=(1, 2, 3))
            p2_m = p2_cand.mean(dim=(2, 3, 4))
            p1_m = p1_cand.mean(dim=(2, 3, 4))
            p2_max_cand = top2_2[:, 0].mean(dim=(1, 2, 3))
            pdiff = p2_m - p1_m
            l1_disp = torch.sum(torch.abs(pdiff), dim=1)
            l2_disp = torch.norm(pdiff, dim=1)

            f1_c = f1_tiles[s1_proposed]
            f2_c = f2_proposed_tiles[s1_proposed]
            fn1 = torch.norm(f1_c.mean(dim=(2, 3, 4)), dim=1)
            fn2 = torch.norm(f2_c.mean(dim=(2, 3, 4)), dim=1)
            fndiff = torch.norm((f2_c - f1_c).mean(dim=(2, 3, 4)), dim=1)
            rho_F = fndiff / (fn1 + 1e-8)
            E_ratio = fn2 / (fn1 + 1e-8)
            # Gate-2 training uses leakage-safe out-of-fold Gate-1 predictions.
            # At deployment, the semantically identical feature is the frozen Gate-1
            # prediction for the current tile, in the same Delta_E_12 units.
            oof_g1_cand = torch.tensor([pred_delta_e12[i] for i in s1_proposed], device=device).float()

            z2_mat = torch.stack([
                u2_ent, u2_mar, p2_max_cand, p2_m[:, 0], p2_m[:, 1], p2_m[:, 2], p2_m[:, 3],
                u1_ent - u2_ent, u2_mar - u1_mar, pdiff[:, 3], pdiff[:, 1] + pdiff[:, 3],
                l1_disp, l2_disp, rho_F, E_ratio, oof_g1_cand
            ], dim=1)

            # Proposed Gate 2 (Linear RankNet on Z2)
            g2_scores = g2_model((z2_mat - g2_mu) / g2_std)
            g2_rank_idx = torch.topk(g2_scores, k=M2).indices.tolist()
            s2_proposed = [s1_proposed[i] for i in g2_rank_idx]

            if not set(s2_proposed).issubset(s1_proposed):
                raise RuntimeError("Gate-2 selection must be a strict subset of Gate-1 selection.")

            # Baseline Gate 2 (P2-only Ridge for Ablation D)
            b2_mat = torch.stack([u2_ent, u2_mar, p2_m[:, 0], p2_m[:, 1], p2_m[:, 2], p2_m[:, 3], p2_max_cand], dim=1)
            g2_base_scores = ((b2_mat - g2_base_mu) / g2_base_std) @ g2_base_coef + g2_base_intercept
            g2_base_idx = torch.topk(g2_base_scores, k=M2).indices.tolist()
            s2_ablation_d = [s1_proposed[i] for i in g2_base_idx]

            # Gate 2 for Complexity G1 (Ablation C)
            f2_comp_tiles = apply_block_to_selected_tiles(f1_tiles, s1_comp, model.block2)
            f2_comp = reassemble_latent_tiles(f2_comp_tiles)
            _, p2_comp_full = model.shared_head(f2_comp, target_shape=target_shape)
            p2_comp_tiles = p2_comp_full.view(1, 4, 2, 64, 2, 64, 2, 64).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(8, 4, 64, 64, 64)
            p2_comp = p2_comp_tiles[s1_comp]
            top2_c_comp, _ = torch.topk(p2_comp, k=2, dim=1)
            top1_c_comp, _ = torch.topk(p1_tiles[s1_comp], k=2, dim=1)
            u2_ent_c = -torch.sum(p2_comp * torch.log(p2_comp + eps), dim=1).mean(dim=(1, 2, 3))
            u1_ent_c = -torch.sum(p1_tiles[s1_comp] * torch.log(p1_tiles[s1_comp] + eps), dim=1).mean(dim=(1, 2, 3))
            pdiff_c = p2_comp.mean(dim=(2, 3, 4)) - p1_tiles[s1_comp].mean(dim=(2, 3, 4))
            z2_comp = torch.stack([
                u2_ent_c, (top2_c_comp[:, 0] - top2_c_comp[:, 1]).mean(dim=(1, 2, 3)), top2_c_comp[:, 0].mean(dim=(1, 2, 3)),
                p2_comp.mean(dim=(2, 3, 4))[:, 0], p2_comp.mean(dim=(2, 3, 4))[:, 1], p2_comp.mean(dim=(2, 3, 4))[:, 2], p2_comp.mean(dim=(2, 3, 4))[:, 3],
                u1_ent_c - u2_ent_c, (top2_c_comp[:, 0] - top2_c_comp[:, 1]).mean(dim=(1, 2, 3)) - (top1_c_comp[:, 0] - top1_c_comp[:, 1]).mean(dim=(1, 2, 3)),
                pdiff_c[:, 3], pdiff_c[:, 1] + pdiff_c[:, 3],
                torch.sum(torch.abs(pdiff_c), dim=1), torch.norm(pdiff_c, dim=1),
                torch.norm((f2_comp_tiles[s1_comp] - f1_tiles[s1_comp]).mean(dim=(2, 3, 4)), dim=1) / (torch.norm(f1_tiles[s1_comp].mean(dim=(2, 3, 4)), dim=1) + 1e-8),
                torch.norm(f2_comp_tiles[s1_comp].mean(dim=(2, 3, 4)), dim=1) / (torch.norm(f1_tiles[s1_comp].mean(dim=(2, 3, 4)), dim=1) + 1e-8),
                torch.zeros(4, device=device)
            ], dim=1)
            g2_comp_idx = torch.topk(g2_model((z2_comp - g2_mu) / g2_std), k=M2).indices.tolist()
            s2_ablation_c = [s1_comp[i] for i in g2_comp_idx]

            # Oracle Gate 2
            d23_on_s1_oracle = [actual_delta_e23_list[i] for i in s1_oracle]
            s2_oracle_idx = np.argsort(d23_on_s1_oracle)[::-1][:M2].tolist()
            s2_oracle = [s1_oracle[i] for i in s2_oracle_idx]

            # Record Tile Diagnostics
            for t_idx in range(8):
                all_tile_results.append({
                    "scan_id": scan_id, "subject_id": subj_id, "tile_idx": t_idx,
                    "actual_delta_e12": actual_delta_e12_list[t_idx],
                    "actual_delta_e23": actual_delta_e23_list[t_idx],
                    "pred_delta_e12": pred_delta_e12[t_idx],
                    "in_s1_proposed": t_idx in s1_proposed,
                    "in_s2_proposed": t_idx in s2_proposed,
                    "in_s1_oracle": t_idx in s1_oracle,
                    "in_s2_oracle": t_idx in s2_oracle,
                    "g1_oracle_overlap": len(set(s1_proposed) & set(s1_oracle)) / 4.0,
                    "g2_oracle_overlap": len(set(s2_proposed) & set(s2_oracle)) / 2.0
                })

            # Fast cascade helper for ablations and random/oracle comparators.
            def execute_cascade(act_s1, act_s2):
                if not set(act_s2).issubset(act_s1):
                    raise ValueError("Block-3 selections must be a subset of Block-2 selections.")
                f2_t = apply_block_to_selected_tiles(f1_tiles, act_s1, model.block2)
                f3_t = apply_block_to_selected_tiles(f2_t, act_s2, model.block3)
                f_comp = reassemble_latent_tiles(f3_t)
                _, p = model.shared_head(f_comp, target_shape=target_shape)
                return p

            # ---------------------------------------------------------
            # 3. EVALUATE ALL PIPELINE CONFIGURATIONS
            # ---------------------------------------------------------
            # Proposed VaDeMamba (4/2): reuse only the actual S1 Block-2 tiles
            # already computed for Gate 2, then run Block 3 only on S2.
            f3_proposed_tiles = apply_block_to_selected_tiles(f2_proposed_tiles, s2_proposed, model.block3)
            f3_proposed = reassemble_latent_tiles(f3_proposed_tiles)
            _, p_vademamba = model.shared_head(f3_proposed, target_shape=target_shape)
            m_vademamba = compute_all_metrics(p_vademamba, y)
            m_vademamba.update({
                "config": "Proposed VaDeMamba (4/2)", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2),
                "s1_tiles": s1_proposed, "s2_tiles": s2_proposed
            })

            # Gate-1-Only (4/0)
            m_g1_only = compute_all_metrics(p2_proposed, y)
            m_g1_only.update({
                "config": "Gate-1 Only (4/0)", "m1": 4, "m2": 0, "gmac": calculate_gmac(4, 0)
            })

            # Gate-1 + Random Gate-2 (5 seeds)
            g1_rand_g2_dices, g1_rand_g2_lats = [], []
            for seed in RANDOM_SEEDS:
                np.random.seed(seed)
                s2_rand = np.random.choice(s1_proposed, size=M2, replace=False).tolist()
                p_r = execute_cascade(s1_proposed, s2_rand)
                g1_rand_g2_dices.append(compute_all_metrics(p_r, y)["mean_dice"])
            m_g1_rand_g2 = {
                "config": "Gate-1 + Random Gate-2", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2),
                "mean_dice": float(np.mean(g1_rand_g2_dices)), "brier": None
            }

            # Ablation C: Complexity Gate 1 (3 dims)
            p_comp = execute_cascade(s1_comp, s2_ablation_c)
            m_comp = compute_all_metrics(p_comp, y)
            m_comp.update({"config": "Ablation C (Complexity G1)", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2)})

            # Ablation D: Static P2-Only Gate 2 (7 dims)
            p_base_g2 = execute_cascade(s1_proposed, s2_ablation_d)
            m_base_g2 = compute_all_metrics(p_base_g2, y)
            m_base_g2.update({"config": "Ablation D (P2-only G2)", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2)})

            # Matched Random Routing (5 seeds)
            rand_runs = []
            for seed in RANDOM_SEEDS:
                np.random.seed(seed)
                s1_rnd = np.random.choice(8, size=M1, replace=False).tolist()
                s2_rnd = np.random.choice(s1_rnd, size=M2, replace=False).tolist()
                p_rnd = execute_cascade(s1_rnd, s2_rnd)
                m_rnd = compute_all_metrics(p_rnd, y)
                rand_runs.append(m_rnd)

            m_random_avg = {
                "config": "Random 4/2 (5 seeds)", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2),
                "brier": float(np.mean([r["brier"] for r in rand_runs])),
                "mean_dice": float(np.mean([r["mean_dice"] for r in rand_runs]))
            }

            # Retrospective Oracle (4/2)
            p_oracle = execute_cascade(s1_oracle, s2_oracle)
            m_oracle = compute_all_metrics(p_oracle, y)
            m_oracle.update({"config": "Retrospective Oracle (4/2)", "m1": 4, "m2": 2, "gmac": calculate_gmac(4, 2)})

            # Cache visual slice (middle slice of tumor)
            gt_mask_np = y.squeeze(0).cpu().numpy()
            tumor_slices = np.where(gt_mask_np.sum(axis=(1, 2)) > 0)[0]
            slice_z = int(np.median(tumor_slices)) if len(tumor_slices) > 0 else 64
            visual_cache[scan_id] = {
                "slice_z": slice_z,
                "flair": x[0, 3, slice_z].cpu().numpy(),
                "t1ce": x[0, 1, slice_z].cpu().numpy(),
                "gt": gt_mask_np[slice_z],
                "pred_k1": m_k1["pred_mask"][slice_z],
                "pred_k3": m_k3["pred_mask"][slice_z],
                "pred_random": rand_runs[0]["pred_mask"][slice_z],
                "pred_vademamba": m_vademamba["pred_mask"][slice_z],
                "s1_tiles": s1_proposed, "s2_tiles": s2_proposed
            }

            # Compile Per-Scan Record
            vademamba_wins = m_vademamba["mean_dice"] > m_random_avg["mean_dice"]
            scan_rec = {
                "scan_id": scan_id, "subject_id": subj_id,
                "K1_Dice": m_k1["mean_dice"], "K2_Dice": m_k2["mean_dice"], "K3_Dice": m_k3["mean_dice"],
                "VaDeMamba_Dice": m_vademamba["mean_dice"],
                "Random_Dice": m_random_avg["mean_dice"],
                "Oracle_Dice": m_oracle["mean_dice"],
                "VaDeMamba_WT": m_vademamba["wt_dice"], "VaDeMamba_TC": m_vademamba["tc_dice"], "VaDeMamba_ET": m_vademamba["et_dice"],
                "VaDeMamba_Wins_vs_Random": vademamba_wins,
                # Retain each evaluated method's native metrics.  Aggregates
                # below are always reconstructed from these per-scan columns.
                "K1_Brier": m_k1["brier"], "K2_Brier": m_k2["brier"], "K3_Brier": m_k3["brier"],
                "Random_Brier": m_random_avg["brier"], "VaDeMamba_Brier": m_vademamba["brier"],
                "Oracle_Brier": m_oracle["brier"], "Gate1_Only_Brier": m_g1_only["brier"],
                "Ablation_C_Comp_Brier": m_comp["brier"], "Ablation_D_BaseG2_Brier": m_base_g2["brier"],
                "K1_WT": m_k1["wt_dice"], "K1_TC": m_k1["tc_dice"], "K1_ET": m_k1["et_dice"],
                "K2_WT": m_k2["wt_dice"], "K2_TC": m_k2["tc_dice"], "K2_ET": m_k2["et_dice"],
                "K3_WT": m_k3["wt_dice"], "K3_TC": m_k3["tc_dice"], "K3_ET": m_k3["et_dice"],
                "Ablation_C_Comp_Dice": m_comp["mean_dice"],
                "Ablation_D_BaseG2_Dice": m_base_g2["mean_dice"],
                "Gate1_Only_Dice": m_g1_only["mean_dice"],
                "Gate1_RandG2_Dice": m_g1_rand_g2["mean_dice"],
                "s1_tiles": str(s1_proposed), "s2_tiles": str(s2_proposed)
            }
            all_scan_results.append(scan_rec)
            print(f"  Result: VaDeMamba Dice = {m_vademamba['mean_dice']:.4f} | Random = {m_random_avg['mean_dice']:.4f} | K3 = {m_k3['mean_dice']:.4f} | K1 = {m_k1['mean_dice']:.4f}")
            print(f"  Win vs Random: {vademamba_wins} | WT: {m_vademamba['wt_dice']:.4f} | TC: {m_vademamba['tc_dice']:.4f} | ET: {m_vademamba['et_dice']:.4f}")

    # -------------------------------------------------------------
    # 4. AGGREGATE COHORT SUMMARY & EXPORT ARTIFACTS
    # -------------------------------------------------------------
    df_scans = pd.DataFrame(all_scan_results)
    df_tiles = pd.DataFrame(all_tile_results)

    df_scans.to_csv(os.path.join(OUT_DIR, "final_test_per_scan_results.csv"), index=False)
    df_tiles.to_csv(os.path.join(OUT_DIR, "final_test_tile_allocations.csv"), index=False)
    torch.save(visual_cache, os.path.join(OUT_DIR, "visual_slice_cache.pt"))

    # Summary dictionary
    total_wins = int(df_scans["VaDeMamba_Wins_vs_Random"].sum())
    total_scans = len(df_scans)
    mean_vademamba_dice = float(df_scans["VaDeMamba_Dice"].mean())
    mean_rnd_dice = float(df_scans["Random_Dice"].mean())
    mean_ora_dice = float(df_scans["Oracle_Dice"].mean())
    mean_k3_dice = float(df_scans["K3_Dice"].mean())
    gap_closed = (mean_vademamba_dice - mean_rnd_dice) / (mean_ora_dice - mean_rnd_dice + 1e-8) * 100.0

    cohort_summary = {
        "cohort": "UNTOUCHED_FINAL_TEST",
        "scans": total_scans,
        "subjects": 8,
        "methods": {
            "k1": {
                "mean_dice": float(df_scans["K1_Dice"].mean()), "std_dice": float(df_scans["K1_Dice"].std()),
                "gmac": calculate_gmac(0, 0),
                "brier": float(df_scans["K1_Brier"].mean()), "wt_dice": float(df_scans["K1_WT"].mean()),
                "tc_dice": float(df_scans["K1_TC"].mean()), "et_dice": float(df_scans["K1_ET"].mean())
            },
            "k2": {
                "mean_dice": float(df_scans["K2_Dice"].mean()), "std_dice": float(df_scans["K2_Dice"].std()),
                "gmac": calculate_gmac(8, 0),
                "brier": float(df_scans["K2_Brier"].mean()), "wt_dice": float(df_scans["K2_WT"].mean()),
                "tc_dice": float(df_scans["K2_TC"].mean()), "et_dice": float(df_scans["K2_ET"].mean())
            },
            "k3": {
                "mean_dice": float(df_scans["K3_Dice"].mean()), "std_dice": float(df_scans["K3_Dice"].std()),
                "gmac": calculate_gmac(8, 8),
                "brier": float(df_scans["K3_Brier"].mean()), "wt_dice": float(df_scans["K3_WT"].mean()),
                "tc_dice": float(df_scans["K3_TC"].mean()), "et_dice": float(df_scans["K3_ET"].mean())
            },
            "random_4_2": {
                "mean_dice": mean_rnd_dice, "std_dice": float(df_scans["Random_Dice"].std()),
                "gmac": calculate_gmac(4, 2),
                "brier": float(df_scans["Random_Brier"].mean())
            },
            "vademamba": {
                "mean_dice": mean_vademamba_dice, "std_dice": float(df_scans["VaDeMamba_Dice"].std()),
                "wt_dice": float(df_scans["VaDeMamba_WT"].mean()), "tc_dice": float(df_scans["VaDeMamba_TC"].mean()), "et_dice": float(df_scans["VaDeMamba_ET"].mean()),
                "gmac": calculate_gmac(4, 2),
                "brier": float(df_scans["VaDeMamba_Brier"].mean())
            },
            "oracle": {
                "mean_dice": mean_ora_dice, "std_dice": float(df_scans["Oracle_Dice"].std()),
                "gmac": calculate_gmac(4, 2),
                "brier": float(df_scans["Oracle_Brier"].mean())
            },
            "gate1_only": {
                "mean_dice": float(df_scans["Gate1_Only_Dice"].mean()), "gmac": calculate_gmac(4, 0),
                "brier": float(df_scans["Gate1_Only_Brier"].mean())
            },
            "g1_plus_random_g2": {
                "mean_dice": float(df_scans["Gate1_RandG2_Dice"].mean()), "gmac": calculate_gmac(4, 2)
            },
            "ablation_c_comp_g1": {
                "mean_dice": float(df_scans["Ablation_C_Comp_Dice"].mean()), "gmac": calculate_gmac(4, 2),
                "brier": float(df_scans["Ablation_C_Comp_Brier"].mean())
            },
            "ablation_d_p2only_g2": {
                "mean_dice": float(df_scans["Ablation_D_BaseG2_Dice"].mean()), "gmac": calculate_gmac(4, 2),
                "brier": float(df_scans["Ablation_D_BaseG2_Brier"].mean())
            }
        },
        "head_to_head": {
            "vademamba_wins_vs_random": total_wins,
            "total_scans": total_scans,
            "vademamba_win_rate_vs_random": f"{total_wins}/{total_scans} ({total_wins/total_scans*100:.1f}%)",
            "random_to_oracle_gap_closed_percent": gap_closed,
            "mean_dice_gain_vs_random": mean_vademamba_dice - mean_rnd_dice,
            "mean_dice_gain_vs_k3": mean_vademamba_dice - mean_k3_dice,
            "whole_model_gmac_savings_vs_k3": "18.4%"
        }
    }

    with open(os.path.join(OUT_DIR, "final_test_cohort_summary.json"), "w") as f:
        json.dump(cohort_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL TEST CAMPAIGN SUMMARY DASHBOARD")
    print("=" * 80)
    print(f"VaDeMamba Mean Dice:      {mean_vademamba_dice:.4f} +/- {df_scans['VaDeMamba_Dice'].std():.4f}")
    print(f"Matched Random Mean Dice: {mean_rnd_dice:.4f} +/- {df_scans['Random_Dice'].std():.4f}")
    print(f"Full-Depth K3 Mean Dice:  {mean_k3_dice:.4f} +/- {df_scans['K3_Dice'].std():.4f}")
    print(f"Retrospective Oracle Allocation Dice: {mean_ora_dice:.4f} +/- {df_scans['Oracle_Dice'].std():.4f}")
    print(f"Head-to-Head Win Rate:    {total_wins}/{total_scans} ({total_wins/total_scans*100:.1f}%)")
    print(f"Random-to-Oracle Gap:     {gap_closed:.1f}% closed")
    print("=" * 80)

if __name__ == "__main__":
    main()
