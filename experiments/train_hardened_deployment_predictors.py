"""
Train Hardened Deployment Predictors on DEV Cohort Only
-------------------------------------------------------
Fits and saves the frozen deployment predictors:
1. Gate 1 Deployment: 7-D Family-B1 Huber MLP (checkpoints/gate1_deployment.pt)
2. Gate 2 Deployment: G2-D Pairwise RankNet (16-D Z2) (checkpoints/gate2_deployment.pt)
3. Gate 2 Baseline: G2-B P2-Only Ridge (7-D) (checkpoints/gate2_baseline_deployment.pt)
4. Gate 1 Complexity: 3-D Complexity Huber MLP (checkpoints/gate1_complexity_deployment.pt)

All training uses strictly the 8 scans of PREDICTOR_CASCADE_DEV.
Sealed test cohort remains 100% untouched.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV_FEATURES_PATH = os.path.join(PROJECT_ROOT, "results", "experiment2_prediction", "dev_features_16x16.csv")
OOF_G1_PATH = os.path.join(PROJECT_ROOT, "results", "experiment3c_predictor_redesign", "gate1_oof_predictions.csv")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

class PredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class LinearRankNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
    def forward(self, x):
        return self.fc(x).squeeze(-1)

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(f"TRAINING HARDENED DEPLOYMENT PREDICTORS ON {device.upper()}")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_csv(DEV_FEATURES_PATH)
    df["macro_d"] = df["coord_d"] // 4
    df["macro_h"] = df["coord_h"] // 4
    df["macro_w"] = df["coord_w"] // 4
    df["macro_tile_id"] = df["macro_d"] * 4 + df["macro_h"] * 2 + df["macro_w"]

    f1_cols = [f"F1_{i:02d}" for i in range(96)]
    f2_cols = [f"F2_{i:02d}" for i in range(96)]
    F1 = df[f1_cols].values.astype(np.float32)
    F2 = df[f2_cols].values.astype(np.float32)
    df["F1_norm"] = np.linalg.norm(F1, axis=1)
    df["F2_norm"] = np.linalg.norm(F2, axis=1)
    diff_F = F2 - F1
    df["diff_F_norm"] = np.linalg.norm(diff_F, axis=1)
    df["rho_F"] = df["diff_F_norm"] / (df["F1_norm"] + 1e-8)
    df["E_ratio"] = df["F2_norm"] / (df["F1_norm"] + 1e-8)

    p1 = df[["P1_BG", "P1_NCR", "P1_ED", "P1_ET"]].values.astype(np.float32)
    p2 = df[["P2_BG", "P2_NCR", "P2_ED", "P2_ET"]].values.astype(np.float32)
    pdiff = p2 - p1
    df["L1_disp"] = np.abs(pdiff).sum(axis=1)
    df["L2_disp"] = np.linalg.norm(pdiff, axis=1)
    df["Delta_U_entropy"] = df["U1_entropy"] - df["U2_entropy"]
    df["Delta_U_margin"] = df["U2_margin"] - df["U1_margin"]
    df["Delta_P_ET"] = pdiff[:, 3]
    df["Delta_P_TC"] = pdiff[:, 1] + pdiff[:, 3]

    COLS_B1 = ["U1_entropy", "U1_margin", "P1_BG", "P1_NCR", "P1_ED", "P1_ET", "P1_max"]
    COLS_COMPLEXITY = ["C_grad", "C_freq", "C_entropy"]
    COLS_B2 = ["U2_entropy", "U2_margin", "P2_BG", "P2_NCR", "P2_ED", "P2_ET", "P2_max"]
    COLS_Z2 = [
        "U2_entropy", "U2_margin", "P2_max", "P2_BG", "P2_NCR", "P2_ED", "P2_ET",
        "Delta_U_entropy", "Delta_U_margin", "Delta_P_ET", "Delta_P_TC", "L1_disp", "L2_disp",
        "rho_F", "E_ratio", "oof_g1_pred"
    ]

    macro_dev = df.groupby(["scan_id", "subject_id", "macro_tile_id"])[
        COLS_B1 + COLS_COMPLEXITY + ["Delta_E_12"]
    ].mean().reset_index()

    # Gate 1 OOF predictions
    oof_g1 = pd.read_csv(OOF_G1_PATH)
    oof_g1_map = dict(zip(zip(oof_g1["scan_id"], oof_g1["macro_tile_id"]), oof_g1["oof_g1_base"]))
    df["oof_g1_pred"] = [oof_g1_map[(r.scan_id, r.macro_tile_id)] for r in df[["scan_id", "macro_tile_id"]].itertuples(index=False)]

    # S1 keys: top 4 tiles per scan
    macro_full = df.groupby(["scan_id", "subject_id", "macro_tile_id"])[["oof_g1_pred", "Delta_E_12"]].mean().reset_index()
    s1_keys = set()
    for sid in macro_full["scan_id"].unique():
        sub = macro_full[macro_full["scan_id"] == sid]
        for tid in sub.nlargest(4, "oof_g1_pred")["macro_tile_id"].values:
            s1_keys.add((sid, tid))

    df["in_s1"] = [(r.scan_id, r.macro_tile_id) in s1_keys for r in df[["scan_id", "macro_tile_id"]].itertuples(index=False)]
    df_s1 = df[df["in_s1"]].copy().reset_index(drop=True)

    macro_s1 = df_s1.groupby(["scan_id", "subject_id", "macro_tile_id"])[
        list(set(COLS_B2 + COLS_Z2)) + ["Delta_E_23"]
    ].mean().reset_index()

    # --- 1. Gate 1 Deployment (Family-B1 Huber MLP) ---
    print("\n1. Training Gate-1 Deployment Predictor (Family-B1 7-D Huber MLP)...")
    X_g1 = macro_dev[COLS_B1].values.astype(np.float32)
    y_g1 = macro_dev["Delta_E_12"].values.astype(np.float32)
    mu_g1 = X_g1.mean(axis=0, keepdims=True)
    std_g1 = X_g1.std(axis=0, keepdims=True) + 1e-8
    X_g1_n = (X_g1 - mu_g1) / std_g1
    target_scale_g1 = 1000.0
    y_g1_s = y_g1 * target_scale_g1

    torch.manual_seed(42)
    model_g1 = PredictorMLP(in_dim=7, hidden_dim=32, dropout=0.10).to(device)
    optimizer_g1 = optim.AdamW(model_g1.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.SmoothL1Loss(beta=0.005)
    tx_g1 = torch.from_numpy(X_g1_n).float().to(device)
    ty_g1 = torch.from_numpy(y_g1_s).float().to(device)

    model_g1.train()
    for _ in range(100):
        optimizer_g1.zero_grad()
        loss = criterion(model_g1(tx_g1), ty_g1)
        loss.backward()
        optimizer_g1.step()
    model_g1.eval()

    ckpt_g1 = os.path.join(CKPT_DIR, "gate1_deployment.pt")
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model_g1.state_dict().items()},
        "feature_columns": COLS_B1,
        "in_dim": 7,
        "hidden_dim": 32,
        "mu": mu_g1.tolist(),
        "std": std_g1.tolist(),
        "target_scale": target_scale_g1,
        "epochs": 100,
        "lr": 1e-3,
        "model_type": "HuberMLP"
    }, ckpt_g1)
    print(f"Saved {ckpt_g1} (SHA256: {compute_sha256(ckpt_g1)[:16]}...)")

    # --- 2. Gate 1 Complexity Predictor (Ablation C) ---
    print("\n2. Training Gate-1 Complexity Predictor (3-D Huber MLP)...")
    X_comp = macro_dev[COLS_COMPLEXITY].values.astype(np.float32)
    mu_comp = X_comp.mean(axis=0, keepdims=True)
    std_comp = X_comp.std(axis=0, keepdims=True) + 1e-8
    X_comp_n = (X_comp - mu_comp) / std_comp
    torch.manual_seed(42)
    model_comp = PredictorMLP(in_dim=3, hidden_dim=32, dropout=0.10).to(device)
    optimizer_comp = optim.AdamW(model_comp.parameters(), lr=1e-3, weight_decay=1e-2)
    tx_comp = torch.from_numpy(X_comp_n).float().to(device)
    model_comp.train()
    for _ in range(100):
        optimizer_comp.zero_grad()
        loss = criterion(model_comp(tx_comp), ty_g1)
        loss.backward()
        optimizer_comp.step()
    model_comp.eval()
    ckpt_comp = os.path.join(CKPT_DIR, "gate1_complexity_deployment.pt")
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model_comp.state_dict().items()},
        "feature_columns": COLS_COMPLEXITY,
        "in_dim": 3,
        "hidden_dim": 32,
        "mu": mu_comp.tolist(),
        "std": std_comp.tolist(),
        "target_scale": target_scale_g1,
        "model_type": "HuberMLP"
    }, ckpt_comp)
    print(f"Saved {ckpt_comp} (SHA256: {compute_sha256(ckpt_comp)[:16]}...)")

    # --- 3. Gate 2 Deployment (G2-D Pairwise RankNet on Z2) ---
    print("\n3. Training Gate-2 Deployment Predictor (G2-D Linear Pairwise RankNet on Z2)...")
    X_g2 = macro_s1[COLS_Z2].values.astype(np.float32)
    y_g2 = macro_s1["Delta_E_23"].values.astype(np.float32)
    mu_g2 = X_g2.mean(axis=0, keepdims=True)
    std_g2 = X_g2.std(axis=0, keepdims=True) + 1e-8
    X_g2_n = (X_g2 - mu_g2) / std_g2

    # Form intra-scan pairs
    pairs_x1, pairs_x2, pairs_y = [], [], []
    for sid in macro_s1["scan_id"].unique():
        s_mask = (macro_s1["scan_id"] == sid).values
        s_X = X_g2_n[s_mask]
        s_y = y_g2[s_mask]
        n_s = len(s_y)
        for i in range(n_s):
            for j in range(n_s):
                if s_y[i] > s_y[j]:
                    pairs_x1.append(s_X[i])
                    pairs_x2.append(s_X[j])
                    pairs_y.append(1.0)

    px1 = torch.tensor(np.array(pairs_x1), dtype=torch.float32).to(device)
    px2 = torch.tensor(np.array(pairs_x2), dtype=torch.float32).to(device)
    py = torch.tensor(np.array(pairs_y), dtype=torch.float32).to(device)

    torch.manual_seed(42)
    ranknet = LinearRankNet(in_dim=16).to(device)
    opt_rn = optim.Adam(ranknet.parameters(), lr=0.01, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss()

    ranknet.train()
    for _ in range(300):
        opt_rn.zero_grad()
        diff = ranknet(px1) - ranknet(px2)
        loss = bce(diff, py)
        loss.backward()
        opt_rn.step()
    ranknet.eval()

    weights = ranknet.fc.weight.detach().cpu().numpy().flatten()

    ckpt_g2 = os.path.join(CKPT_DIR, "gate2_deployment.pt")
    torch.save({
        "weights": weights.tolist(),
        "model_state_dict": {k: v.cpu() for k, v in ranknet.state_dict().items()},
        "feature_columns": COLS_Z2,
        "in_dim": 16,
        "mu": mu_g2.tolist(),
        "std": std_g2.tolist(),
        "model_type": "PairwiseRankNet",
        "training_pairs": len(pairs_y),
        "selection_rationale": "Top performer in DEV subject-grouped OOF: 79.2% rank accuracy, 75.0% top-2 overlap, 87.5% scan win rate."
    }, ckpt_g2)
    print(f"Saved {ckpt_g2} (SHA256: {compute_sha256(ckpt_g2)[:16]}...)")

    # --- 4. Gate 2 Baseline (P2-Only Ridge for Ablation D) ---
    print("\n4. Training Gate-2 Baseline Predictor (P2-Only Ridge for Ablation D)...")
    X_g2_base = macro_s1[COLS_B2].values.astype(np.float32)
    mu_g2_base = X_g2_base.mean(axis=0, keepdims=True)
    std_g2_base = X_g2_base.std(axis=0, keepdims=True) + 1e-8
    X_g2_base_n = (X_g2_base - mu_g2_base) / std_g2_base

    ridge_base = Ridge(alpha=10.0)
    ridge_base.fit(X_g2_base_n, y_g2)
    ckpt_g2_base = os.path.join(CKPT_DIR, "gate2_baseline_deployment.pt")
    torch.save({
        "coef": ridge_base.coef_.tolist(),
        "intercept": float(ridge_base.intercept_),
        "alpha": 10.0,
        "feature_columns": COLS_B2,
        "in_dim": 7,
        "mu": mu_g2_base.tolist(),
        "std": std_g2_base.tolist(),
        "model_type": "Ridge"
    }, ckpt_g2_base)
    print(f"Saved {ckpt_g2_base} (SHA256: {compute_sha256(ckpt_g2_base)[:16]}...)")

    # 5. Save Manifest
    manifest = {
        "gate1_deployment": {
            "path": ckpt_g1, "sha256": compute_sha256(ckpt_g1),
            "model_type": "HuberMLP", "in_dim": 7, "features": COLS_B1
        },
        "gate1_complexity_deployment": {
            "path": ckpt_comp, "sha256": compute_sha256(ckpt_comp),
            "model_type": "HuberMLP", "in_dim": 3, "features": COLS_COMPLEXITY
        },
        "gate2_deployment": {
            "path": ckpt_g2, "sha256": compute_sha256(ckpt_g2),
            "model_type": "PairwiseRankNet", "in_dim": 16, "features": COLS_Z2
        },
        "gate2_baseline_deployment": {
            "path": ckpt_g2_base, "sha256": compute_sha256(ckpt_g2_base),
            "model_type": "Ridge", "in_dim": 7, "features": COLS_B2
        }
    }
    manifest_path = os.path.join(CKPT_DIR, "deployment_predictors_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")

if __name__ == "__main__":
    main()
