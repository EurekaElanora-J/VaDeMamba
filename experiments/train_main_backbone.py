"""
Main Backbone Training Script: 14 Unique Subjects (20 Scans)
------------------------------------------------------------
Trains the progressive-depth 3D Mamba backbone (K1, K2, K3 coupled exits)
using the locked BACKBONE_TRAIN cohort from data/splits/cohort_split_manifest.json.
Validates on PREDICTOR_CASCADE_DEV (8 scans) to select the best checkpoint.
UNTOUCHED_FINAL_TEST is strictly off-limits.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import load_and_preprocess_patient
from src.data.patch_sampler import BraTSPatchDataset
from src.models.modular_backbone import ModularMambaBackbone
from src.metrics.brier import compute_voxelwise_brier_map, compute_patch_dice
from src.losses.compound import RevisedExperiment0Loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    start_time = time.time()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"MAIN BACKBONE TRAINING ON {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 80)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(project_root, "data", "raw", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
    splits_dir = os.path.join(project_root, "data", "splits")
    save_dir = os.path.join(project_root, "checkpoints")
    results_dir = os.path.join(project_root, "results", "experiment2_prediction")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load Locked Manifest
    manifest_path = os.path.join(splits_dir, "cohort_split_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing locked manifest at: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_scans = manifest["cohorts"]["BACKBONE_TRAIN"]["scans"]
    dev_scans = manifest["cohorts"]["PREDICTOR_CASCADE_DEV"]["scans"]
    test_scans = manifest["cohorts"]["UNTOUCHED_FINAL_TEST"]["scans"]

    print(f"\n[Cohort Verification]")
    print(f"  BACKBONE_TRAIN Scans ({len(train_scans)}):        {train_scans}")
    print(f"  PREDICTOR_CASCADE_DEV Scans ({len(dev_scans)}):  {dev_scans}")
    print(f"  UNTOUCHED_FINAL_TEST Scans ({len(test_scans)}):  [STRICTLY ISOLATED - {len(test_scans)} scans]")
    
    assert len(train_scans) == 20, f"Expected 20 train scans, got {len(train_scans)}"
    assert len(dev_scans) == 8, f"Expected 8 dev scans, got {len(dev_scans)}"
    assert len(set(train_scans).intersection(set(dev_scans))) == 0, "Leakage between train and dev!"
    assert len(set(train_scans).intersection(set(test_scans))) == 0, "Leakage between train and test!"

    # Load Patients
    print(f"\nLoading and preprocessing {len(train_scans)} training volumes...")
    t0 = time.time()
    train_patients = [load_and_preprocess_patient(os.path.join(data_root, s), s) for s in train_scans]
    print(f"Training volumes loaded in {time.time()-t0:.1f}s.")

    print(f"\nLoading and preprocessing {len(dev_scans)} dev validation volumes...")
    t1 = time.time()
    dev_patients = [load_and_preprocess_patient(os.path.join(data_root, s), s) for s in dev_scans]
    print(f"Dev volumes loaded in {time.time()-t1:.1f}s.")

    # Datasets
    # 2 tumor-centered patches per patient per epoch
    train_dataset = BraTSPatchDataset(train_patients, patch_size=(128, 128, 128), samples_per_patient=2, is_train=True)
    # 1 deterministic tumor-centered patch per dev patient
    dev_dataset = BraTSPatchDataset(dev_patients, patch_size=(128, 128, 128), samples_per_patient=1, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    epochs = 10
    print(f"\n[Training Parameters]")
    print(f"  Epochs:                    {epochs}")
    print(f"  Train Patches per Epoch:   {len(train_loader)} (Total = {len(train_loader)*epochs} iters)")
    print(f"  Dev Patches per Epoch:     {len(dev_loader)}")

    # Model & Optimization
    model = ModularMambaBackbone(in_channels=4, hidden_dim=96, num_classes=4, d_state=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = RevisedExperiment0Loss(eps=1e-5)

    best_mean_dice = -1.0
    best_ckpt_path = os.path.join(save_dir, "main_backbone_best.pt")
    best_epoch = -1
    best_metrics = {}
    history = []

    print("\n" + "=" * 80)
    print(f"STARTING MAIN BACKBONE TRAINING ({epochs} EPOCHS)")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        # 1. Train
        model.train()
        train_loss = 0.0
        torch.cuda.synchronize()
        t_tr_start = time.perf_counter()

        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device)
            y = batch["seg"].to(device).long()

            optimizer.zero_grad()
            out = model(x)
            loss, loss_dict = criterion(out, y)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] | Step [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        scheduler.step()
        torch.cuda.synchronize()
        train_time = time.perf_counter() - t_tr_start
        mean_train_loss = train_loss / len(train_loader)

        # 2. Validation on Dev Cohort (8 scans)
        model.eval()
        t_val_start = time.perf_counter()
        
        val_dices = {"exit1": [], "exit2": [], "exit3": []}
        val_briers = {"exit1": [], "exit2": [], "exit3": []}

        with torch.no_grad():
            for batch in dev_loader:
                x = batch["image"].to(device)
                y = batch["seg"].to(device).long()
                out = model(x)

                for k in ["exit1", "exit2", "exit3"]:
                    prob_k = out["probs"][k]
                    
                    # Dice: WT, TC, ET
                    dices = compute_patch_dice(prob_k.squeeze(0), y.squeeze(0))
                    val_dices[k].append([dices["Dice_WT"], dices["Dice_TC"], dices["Dice_ET"]])

                    # Voxelwise Brier
                    brier_map = compute_voxelwise_brier_map(prob_k, y)
                    val_briers[k].append(brier_map.mean().item())

        val_time = time.perf_counter() - t_val_start

        # Average metrics across dev cohort
        mean_metrics = {}
        for k in ["exit1", "exit2", "exit3"]:
            d_arr = np.array(val_dices[k])
            b_arr = np.array(val_briers[k])
            mean_metrics[k] = {
                "dice_wt": float(d_arr[:, 0].mean()),
                "dice_tc": float(d_arr[:, 1].mean()),
                "dice_et": float(d_arr[:, 2].mean()),
                "dice_mean": float(d_arr.mean()),
                "brier": float(b_arr.mean())
            }

        # Overall validation criterion: Mean Dice across all exits
        mean_dice_all_exits = float(np.mean([mean_metrics[k]["dice_mean"] for k in ["exit1", "exit2", "exit3"]]))

        print(f"\n--- Epoch {epoch}/{epochs} Complete (Train: {train_time:.1f}s, Val: {val_time:.1f}s) ---")
        print(f"  Train Loss: {mean_train_loss:.4f}")
        for k in ["exit1", "exit2", "exit3"]:
            m = mean_metrics[k]
            print(f"  {k.upper()}: Brier={m['brier']:.5f} | WT={m['dice_wt']:.4f}, TC={m['dice_tc']:.4f}, ET={m['dice_et']:.4f} | ExitMean={m['dice_mean']:.4f}")
        print(f"  All-Exit Mean Dice: {mean_dice_all_exits:.4f}")

        # Checkpoint Selection
        if mean_dice_all_exits > best_mean_dice:
            best_mean_dice = mean_dice_all_exits
            best_epoch = epoch
            best_metrics = mean_metrics
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_dice": best_mean_dice,
                "metrics": mean_metrics,
                "cohort_manifest_seed": 42
            }, best_ckpt_path)
            print(f"  >>> New Best Checkpoint Saved to {best_ckpt_path} (Epoch {epoch}, MeanDice: {best_mean_dice:.4f})")

        history.append({
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "train_time_s": train_time,
            "val_time_s": val_time,
            "mean_dice_all_exits": mean_dice_all_exits,
            "metrics": mean_metrics
        })

    total_wall_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE IN {total_wall_time/60.0:.2f} MINUTES")
    print(f"Best Checkpoint: Epoch {best_epoch} with All-Exit Mean Dice: {best_mean_dice:.4f}")
    print("=" * 80)

    # Save training log
    log_path = os.path.join(results_dir, "main_backbone_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_wall_time_s": total_wall_time,
            "best_epoch": best_epoch,
            "best_mean_dice": best_mean_dice,
            "best_metrics": best_metrics,
            "history": history
        }, f, indent=2)
    print(f"Training log saved to: {log_path}")

if __name__ == "__main__":
    main()
