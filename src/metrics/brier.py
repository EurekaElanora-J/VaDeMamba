import torch
import torch.nn.functional as F
import numpy as np

def compute_voxelwise_brier_map(prob, target, num_classes=4):
    """
    Computes 4-class Brier score map for a volume.
    prob: (B, num_classes, D, H, W) probabilities or (num_classes, D, H, W)
    target: (B, D, H, W) integer labels in [0, num_classes - 1] or (D, H, W)
    Returns: (B, D, H, W) or (D, H, W) Brier error map.
    """
    squeeze = False
    if prob.ndim == 4:
        prob = prob.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
        
    B, C, D, H, W = prob.shape
    # One-hot encode target: (B, C, D, H, W)
    target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
    
    # Brier score: 1/C * sum_c (P_c - Y_c)^2
    brier_map = torch.mean((prob - target_one_hot) ** 2, dim=1) # (B, D, H, W)
    
    if squeeze:
        brier_map = brier_map.squeeze(0)
    return brier_map

def compute_voxelwise_ce_map(prob, target, eps=1e-7):
    """
    Computes cross-entropy error map: -log(P_true).
    """
    squeeze = False
    if prob.ndim == 4:
        prob = prob.unsqueeze(0)
        target = target.unsqueeze(0)
        squeeze = True
        
    B, C, D, H, W = prob.shape
    target_one_hot = F.one_hot(target.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
    ce_map = -torch.sum(target_one_hot * torch.log(prob + eps), dim=1) # (B, D, H, W)
    
    if squeeze:
        ce_map = ce_map.squeeze(0)
    return ce_map

def compute_region_deltas(brier_map1, brier_map2, brier_map3, region_slice):
    """
    Given Brier error maps from exits 1, 2, 3 and spatial slice (d_s, h_s, w_s),
    computes:
      E_1, E_2, E_3
      Delta_E_13 = E_1 - E_3 (Primary)
      Delta_E_12 = E_1 - E_2
      Delta_E_23 = E_2 - E_3
    """
    d_s, h_s, w_s = region_slice
    e1 = float(brier_map1[d_s, h_s, w_s].mean().item())
    e2 = float(brier_map2[d_s, h_s, w_s].mean().item())
    e3 = float(brier_map3[d_s, h_s, w_s].mean().item())
    
    return {
        "E_1": e1,
        "E_2": e2,
        "E_3": e3,
        "Delta_E_13": e1 - e3,
        "Delta_E_12": e1 - e2,
        "Delta_E_23": e2 - e3
    }

def compute_patch_dice(prob, target, num_classes=4):
    """
    Computes Dice score for WT, TC, ET across the entire patch.
    prob: (C, D, H, W)
    target: (D, H, W)
    """
    pred_mask = torch.argmax(prob, dim=0) # (D, H, W)
    
    # WT: classes 1, 2, 3
    pred_wt = (pred_mask > 0).float()
    true_wt = (target > 0).float()
    dice_wt = (2.0 * (pred_wt * true_wt).sum()) / (pred_wt.sum() + true_wt.sum() + 1e-6)
    
    # TC: classes 1, 3
    pred_tc = ((pred_mask == 1) | (pred_mask == 3)).float()
    true_tc = ((target == 1) | (target == 3)).float()
    dice_tc = (2.0 * (pred_tc * true_tc).sum()) / (pred_tc.sum() + true_tc.sum() + 1e-6)
    
    # ET: class 3
    pred_et = (pred_mask == 3).float()
    true_et = (target == 3).float()
    dice_et = (2.0 * (pred_et * true_et).sum()) / (pred_et.sum() + true_et.sum() + 1e-6)
    
    return {
        "Dice_WT": float(dice_wt.item()),
        "Dice_TC": float(dice_tc.item()),
        "Dice_ET": float(dice_et.item())
    }

if __name__ == "__main__":
    # Test with synthetic probability tensors
    target = torch.randint(0, 4, (128, 128, 128))
    prob1 = torch.softmax(torch.randn(4, 128, 128, 128), dim=0)
    # Simulate prob3 being closer to target than prob1
    prob3 = prob1 * 0.5 + F.one_hot(target, num_classes=4).permute(3, 0, 1, 2).float() * 0.5
    prob2 = (prob1 + prob3) / 2.0
    
    b1 = compute_voxelwise_brier_map(prob1, target)
    b2 = compute_voxelwise_brier_map(prob2, target)
    b3 = compute_voxelwise_brier_map(prob3, target)
    
    region_slice = (slice(0, 16), slice(0, 16), slice(0, 16))
    deltas = compute_region_deltas(b1, b2, b3, region_slice)
    print("Test Brier Deltas:", deltas)
    assert deltas["Delta_E_13"] > 0, "Delta_E_13 should be positive when Exit 3 has lower error"
    print("Brier metrics verification passed!")
