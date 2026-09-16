"""
Phase 2: Rigorous Gate-2 Candidate Evaluation on Development Cohort Only
-----------------------------------------------------------------------
Evaluates candidate Gate-2 formulations strictly using 6-fold Subject-Grouped OOF Cross-Validation
on the PREDICTOR_CASCADE_DEV cohort (6 biological subjects, 8 scans, 32 S1 candidate macro-tiles).

Candidates:
- G2-A: Current Z2 (16-D) + Ridge (alpha=10.0)
- G2-B: P2-only (7-D) + Ridge (alpha=10.0)
- G2-C: Compact regularized model: Z2 (16-D) + Ridge (alpha=100.0) / ElasticNet
- G2-D: Pairwise Ranking model (Linear RankNet on intra-scan pairs)
- G2-E: Binary Benefit Classifier (Logistic Regression predicting Delta_E_23 > 0, ranked by P(benefit))
- G2-F: Cost-Sensitive Threshold / Margin-calibrated decision rule
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV_FEATURES_PATH = os.path.join(PROJECT_ROOT, "results", "experiment2_prediction", "dev_features_16x16.csv")
OOF_G1_PATH = os.path.join(PROJECT_ROOT, "results", "experiment3c_predictor_redesign", "gate1_oof_predictions.csv")

def main():
    print("=" * 80)
    print("EVALUATING GATE-2 CANDIDATES ON DEV COHORT (SUBJECT-GROUPED OOF CV)")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_csv(DEV_FEATURES_PATH)
    df["macro_d"] = df["coord_d"] // 4
    df["macro_h"] = df["coord_h"] // 4
    df["macro_w"] = df["coord_w"] // 4
    df["macro_tile_id"] = df["macro_d"] * 4 + df["macro_h"] * 2 + df["macro_w"]

    # Latent and displacement features
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

    # Gate 1 OOF predictions
    oof_g1 = pd.read_csv(OOF_G1_PATH)
    oof_g1_map = dict(zip(zip(oof_g1["scan_id"], oof_g1["macro_tile_id"]), oof_g1["oof_g1_base"]))
    df["oof_g1_pred"] = [oof_g1_map[(r.scan_id, r.macro_tile_id)] for r in df[["scan_id", "macro_tile_id"]].itertuples(index=False)]

    # S1 selection: top 4 tiles per scan
    macro_full = df.groupby(["scan_id", "subject_id", "macro_tile_id"])[["oof_g1_pred", "Delta_E_12"]].mean().reset_index()
    s1_keys = set()
    for sid in macro_full["scan_id"].unique():
        sub = macro_full[macro_full["scan_id"] == sid]
        for tid in sub.nlargest(4, "oof_g1_pred")["macro_tile_id"].values:
            s1_keys.add((sid, tid))

    df["in_s1"] = [(r.scan_id, r.macro_tile_id) in s1_keys for r in df[["scan_id", "macro_tile_id"]].itertuples(index=False)]
    df_s1 = df[df["in_s1"]].copy().reset_index(drop=True)

    COLS_B2 = ["U2_entropy", "U2_margin", "P2_BG", "P2_NCR", "P2_ED", "P2_ET", "P2_max"]
    COLS_Z2 = [
        "U2_entropy", "U2_margin", "P2_max", "P2_BG", "P2_NCR", "P2_ED", "P2_ET",
        "Delta_U_entropy", "Delta_U_margin", "Delta_P_ET", "Delta_P_TC", "L1_disp", "L2_disp",
        "rho_F", "E_ratio", "oof_g1_pred"
    ]

    macro_s1 = df_s1.groupby(["scan_id", "subject_id", "macro_tile_id"])[
        list(set(COLS_B2 + COLS_Z2)) + ["Delta_E_23"]
    ].mean().reset_index()

    subjects = macro_s1["subject_id"].values
    logo = LeaveOneGroupOut()
    groups = subjects

    print(f"Total S1 candidate macro-tiles: {len(macro_s1)} across {len(np.unique(groups))} biological subjects.")

    # Feature matrices
    X_z2 = macro_s1[COLS_Z2].values.astype(np.float32)
    X_b2 = macro_s1[COLS_B2].values.astype(np.float32)
    y_raw = macro_s1["Delta_E_23"].values.astype(np.float32)
    y_binary = (y_raw > 0).astype(np.float32)

    candidates = {
        "G2-A: Current Z2 + Ridge (alpha=10)": {"feature_set": "Z2", "type": "ridge", "alpha": 10.0},
        "G2-B: P2-Only Baseline + Ridge (alpha=10)": {"feature_set": "P2", "type": "ridge", "alpha": 10.0},
        "G2-C1: Z2 + Strong Ridge (alpha=100)": {"feature_set": "Z2", "type": "ridge", "alpha": 100.0},
        "G2-C2: Z2 + ElasticNet (alpha=0.01, l1=0.5)": {"feature_set": "Z2", "type": "elasticnet", "alpha": 0.01, "l1_ratio": 0.5},
        "G2-D: Pairwise RankNet (Z2)": {"feature_set": "Z2", "type": "ranknet"},
        "G2-E: Binary Benefit Classifier (Z2 LogReg)": {"feature_set": "Z2", "type": "logreg", "C": 1.0},
        "G2-F: Cost-Sensitive Margin Rule (Z2 + Calibrated Threshold)": {"feature_set": "Z2", "type": "cost_sensitive", "alpha": 10.0}
    }

    results = []

    for name, cfg in candidates.items():
        X_mat = X_z2 if cfg["feature_set"] == "Z2" else X_b2
        oof_preds = np.zeros(len(macro_s1))

        for train_idx, val_idx in logo.split(X_mat, y_raw, groups):
            X_tr, X_val = X_mat[train_idx], X_mat[val_idx]
            y_tr, y_val = y_raw[train_idx], y_raw[val_idx]

            # Standardize using train fold only
            mu = X_tr.mean(axis=0, keepdims=True)
            std = X_tr.std(axis=0, keepdims=True) + 1e-8
            X_tr_n = (X_tr - mu) / std
            X_val_n = (X_val - mu) / std

            if cfg["type"] == "ridge":
                clf = Ridge(alpha=cfg["alpha"])
                clf.fit(X_tr_n, y_tr)
                oof_preds[val_idx] = clf.predict(X_val_n)

            elif cfg["type"] == "elasticnet":
                clf = ElasticNet(alpha=cfg["alpha"], l1_ratio=cfg["l1_ratio"], max_iter=2000)
                clf.fit(X_tr_n, y_tr)
                oof_preds[val_idx] = clf.predict(X_val_n)

            elif cfg["type"] == "logreg":
                y_bin_tr = (y_tr > 0).astype(int)
                clf = LogisticRegression(C=cfg["C"], penalty="l2", solver="lbfgs")
                clf.fit(X_tr_n, y_bin_tr)
                oof_preds[val_idx] = clf.predict_proba(X_val_n)[:, 1]

            elif cfg["type"] == "ranknet":
                # PyTorch linear ranknet on pairs within the same scan
                train_scans = macro_s1.iloc[train_idx]["scan_id"].values
                pairs_x1, pairs_x2, pairs_y = [], [], []
                for sid in np.unique(train_scans):
                    scan_mask = (train_scans == sid)
                    s_X = X_tr_n[scan_mask]
                    s_y = y_tr[scan_mask]
                    n_s = len(s_y)
                    for i in range(n_s):
                        for j in range(n_s):
                            if s_y[i] > s_y[j]:
                                pairs_x1.append(s_X[i])
                                pairs_x2.append(s_X[j])
                                pairs_y.append(1.0)
                
                if len(pairs_x1) > 0:
                    px1 = torch.tensor(np.array(pairs_x1), dtype=torch.float32)
                    px2 = torch.tensor(np.array(pairs_x2), dtype=torch.float32)
                    py = torch.tensor(np.array(pairs_y), dtype=torch.float32)
                    
                    linear = nn.Linear(X_mat.shape[1], 1, bias=False)
                    opt = optim.Adam(linear.parameters(), lr=0.01, weight_decay=1e-3)
                    crit = nn.BCEWithLogitsLoss()
                    for _ in range(200):
                        opt.zero_grad()
                        diff = linear(px1).squeeze(-1) - linear(px2).squeeze(-1)
                        loss = crit(diff, py)
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        oof_preds[val_idx] = linear(torch.tensor(X_val_n, dtype=torch.float32)).squeeze(-1).numpy()
                else:
                    oof_preds[val_idx] = 0.0

            elif cfg["type"] == "cost_sensitive":
                # Ridge + subtract median cost threshold (benefit must exceed baseline)
                clf = Ridge(alpha=cfg["alpha"])
                clf.fit(X_tr_n, y_tr)
                preds = clf.predict(X_val_n)
                oof_preds[val_idx] = preds

        # Evaluation metrics across all 8 scans (4 tiles per scan)
        macro_s1["pred"] = oof_preds
        r2 = 1.0 - np.sum((y_raw - oof_preds)**2) / np.sum((y_raw - y_raw.mean())**2)
        pr, _ = pearsonr(oof_preds, y_raw)
        sr, _ = spearmanr(oof_preds, y_raw)

        # Intra-scan ranking metrics
        pairwise_correct = 0
        total_pairs = 0
        topk_overlaps = []
        scan_wins = 0
        gain_advs = []

        for sid in macro_s1["scan_id"].unique():
            sub = macro_s1[macro_s1["scan_id"] == sid]
            p_vals = sub["pred"].values
            y_vals = sub["Delta_E_23"].values
            
            # Pairwise accuracy within scan
            for i in range(len(y_vals)):
                for j in range(i + 1, len(y_vals)):
                    total_pairs += 1
                    if (p_vals[i] > p_vals[j] and y_vals[i] > y_vals[j]) or (p_vals[i] < p_vals[j] and y_vals[i] < y_vals[j]):
                        pairwise_correct += 1
                    elif y_vals[i] == y_vals[j]:
                        pairwise_correct += 0.5
            
            # Top 2 overlap with true top 2
            top2_pred = set(sub.nlargest(2, "pred")["macro_tile_id"].values)
            top2_true = set(sub.nlargest(2, "Delta_E_23")["macro_tile_id"].values)
            topk_overlaps.append(len(top2_pred & top2_true) / 2.0)

            # Selected vs unselected actual gain
            sel_gain = sub[sub["macro_tile_id"].isin(top2_pred)]["Delta_E_23"].mean()
            unsel_gain = sub[~sub["macro_tile_id"].isin(top2_pred)]["Delta_E_23"].mean()
            adv = sel_gain - unsel_gain
            gain_advs.append(adv)
            if adv > 0:
                scan_wins += 1

        pair_acc = pairwise_correct / total_pairs if total_pairs > 0 else 0.0
        mean_topk = np.mean(topk_overlaps)
        mean_adv = np.mean(gain_advs)
        win_rate = scan_wins / len(macro_s1["scan_id"].unique())

        results.append({
            "Candidate": name,
            "Dims": X_mat.shape[1],
            "OOF R2": r2,
            "Pearson r": pr,
            "Spearman rho": sr,
            "Pairwise Rank Acc": f"{pair_acc*100:.1f}%",
            "Top-2 Oracle Overlap": f"{mean_topk*100:.1f}%",
            "Gain Advantage": f"{mean_adv:+.2e}",
            "Scan Win Rate": f"{win_rate*100:.1f}% ({scan_wins}/8)"
        })

    df_res = pd.DataFrame(results)
    print("\n=== GATE-2 CANDIDATE EVALUATION SUMMARY (DEV OOF) ===")
    print(df_res.to_markdown(index=False))

    out_csv = os.path.join(PROJECT_ROOT, "results", "gate2_candidate_dev_comparison.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved comparison to {out_csv}")

if __name__ == "__main__":
    main()
