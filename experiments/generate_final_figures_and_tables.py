"""
Generate Publication-Grade Figures and Tables for VaDeMamba (Hardened & Audited)
---------------------------------------------------------------------------------
Legacy retained-artifact generator for the pre-alignment table/figure set.

It remains available for historical provenance, but it is not the reporting
path for the submitted manuscript's controlled-comparison Table 6 or mechanism
validation Table 8.  Those are reported read-only from authorized retained
artifacts by ``experiments/report_final_manuscript_artifacts.py``.  In
particular, the legacy generator's latency-unavailable fields must not be used
to describe the manuscript's separately measured controlled comparison.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "final_campaign")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# 1. Load retained final-test artifacts
df_per_scan = pd.read_csv(os.path.join(RESULTS_DIR, "final_test_per_scan_results.csv"))
df_tiles = pd.read_csv(os.path.join(RESULTS_DIR, "final_test_tile_allocations.csv"))

visual_cache = None
if os.path.exists(os.path.join(RESULTS_DIR, "visual_slice_cache.pt")):
    import torch
    visual_cache = torch.load(os.path.join(RESULTS_DIR, "visual_slice_cache.pt"), map_location="cpu", weights_only=False)

# Load Gate 2 DEV candidate comparison
df_g2_dev = None
g2_dev_csv = os.path.join(PROJECT_ROOT, "results", "gate2_candidate_dev_comparison.csv")
if os.path.exists(g2_dev_csv):
    df_g2_dev = pd.read_csv(g2_dev_csv)

# End-to-end latency is unavailable / not claimed
df_latency = None

# Set publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'font.family': 'sans-serif'
})

print("=" * 80)
print("GENERATING VADEMAMBA PUBLICATION TABLES (1 TO 10)")
print("=" * 80)

def mean_std(column):
    return float(df_per_scan[column].mean()), float(df_per_scan[column].std())

def method(column, gmac, brier_column=None):
    mean, std = mean_std(column)
    return {"mean_dice": mean, "std_dice": std, "gmac": gmac,
            "brier": float(df_per_scan[brier_column].mean()) if brier_column else None}

m = {
    "k1": method("K1_Dice", 6.893),
    "k2": method("K2_Dice", 8.333),
    "k3": method("K3_Dice", 9.773, "K3_Brier"),
    "random_4_2": method("Random_Dice", 7.973),
    "vademamba": method("VaDeMamba_Dice", 7.973, "VaDeMamba_Brier"),
    "oracle": method("Oracle_Dice", 7.973),
    "gate1_only": method("Gate1_Only_Dice", 7.613),
    "g1_plus_random_g2": method("Gate1_RandG2_Dice", 7.973),
    "ablation_c_comp_g1": method("Ablation_C_Comp_Dice", 7.973),
    "ablation_d_p2only_g2": method("Ablation_D_BaseG2_Dice", 7.973),
}

h2h = {
    "vademamba_win_rate_vs_random": f"{int(df_per_scan['VaDeMamba_Wins_vs_Random'].sum())}/{len(df_per_scan)} ({df_per_scan['VaDeMamba_Wins_vs_Random'].mean()*100:.1f}%)",
    "random_to_oracle_gap_closed_percent": (m["vademamba"]["mean_dice"] - m["random_4_2"]["mean_dice"]) / (m["oracle"]["mean_dice"] - m["random_4_2"]["mean_dice"]) * 100
}

# -------------------------------------------------------------
# TABLE 1: Dataset and Subject-Disjoint Split
# -------------------------------------------------------------
tab1_data = [
    {"Cohort": "Backbone Training (BACKBONE_TRAIN)", "Subjects": 14, "Scans": 20, "Input Modalities": "4 (T1, T1ce, T2, FLAIR)", "Voxel Shape": "4 x 128 x 128 x 128", "Purpose": "Backbone weight optimization"},
    {"Cohort": "Predictor Development (PREDICTOR_CASCADE_DEV)", "Subjects": 6, "Scans": 8, "Input Modalities": "4 (T1, T1ce, T2, FLAIR)", "Voxel Shape": "4 x 128 x 128 x 128", "Purpose": "Feature selection & predictor CV"},
    {"Cohort": "Final Test (UNTOUCHED_FINAL_TEST)", "Subjects": 8, "Scans": 12, "Input Modalities": "4 (T1, T1ce, T2, FLAIR)", "Voxel Shape": "4 x 128 x 128 x 128", "Purpose": "Sealed benchmark evaluation"},
    {"Cohort": "Total Cohort", "Subjects": 28, "Scans": 40, "Input Modalities": "4 (T1, T1ce, T2, FLAIR)", "Voxel Shape": "4 x 128 x 128 x 128", "Purpose": "100% Subject-Disjoint Partition"}
]
pd.DataFrame(tab1_data).to_csv(os.path.join(TAB_DIR, "table1_dataset_split.csv"), index=False)

# -------------------------------------------------------------
# TABLE 2: Backbone Architecture & Parameter Breakdown
# -------------------------------------------------------------
tab2_data = [
    {"Stage / Module": "Input Volume", "Dimensions": "4 x 128 x 128 x 128", "Stride / Kernel": "-", "Parameters": 0, "GMAC Cost": "-"},
    {"Stage / Module": "Stem Stage 1", "Dimensions": "4 -> 48 x 64 x 64 x 64", "Stride / Kernel": "Conv3D (k=3, s=2)", "Parameters": 5232, "GMAC Cost": "1.37 GMAC"},
    {"Stage / Module": "Stem Stage 2", "Dimensions": "48 -> 96 x 32 x 32 x 32", "Stride / Kernel": "Conv3D (k=3, s=2)", "Parameters": 124512, "GMAC Cost": "4.07 GMAC"},
    {"Stage / Module": "Total Stem (F0)", "Dimensions": "96 x 32 x 32 x 32", "Stride / Kernel": "Total Stride = 4", "Parameters": 129744, "GMAC Cost": "5.44 GMAC"},
    {"Stage / Module": "Mamba Block 1 (Global)", "Dimensions": "96 x 32 x 32 x 32", "Stride / Kernel": "Bidirectional SSM (d_state=16)", "Parameters": 112032, "GMAC Cost": "1.44 GMAC"},
    {"Stage / Module": "Mamba Block 2 (Tiled)", "Dimensions": "96 x 16 x 16 x 16 per tile", "Stride / Kernel": "Bidirectional SSM (d_state=16)", "Parameters": 112032, "GMAC Cost": "1.44 GMAC (full) / 0.72 (M1=4)"},
    {"Stage / Module": "Mamba Block 3 (Tiled)", "Dimensions": "96 x 16 x 16 x 16 per tile", "Stride / Kernel": "Bidirectional SSM (d_state=16)", "Parameters": 112032, "GMAC Cost": "1.44 GMAC (full) / 0.36 (M2=2)"},
    {"Stage / Module": "Shared Readout Head (H)", "Dimensions": "96 -> 4 x 128 x 128 x 128", "Stride / Kernel": "Conv3D (1x1x1) + Trilinear x4", "Parameters": 388, "GMAC Cost": "0.013 GMAC"},
    {"Stage / Module": "Total Model (Full K3)", "Dimensions": "4 classes @ 128^3", "Stride / Kernel": "Complete 3-Stage Backbone", "Parameters": 466228, "GMAC Cost": "9.773 GMAC"},
    {"Stage / Module": "Total Model (VaDeMamba 4/2)", "Dimensions": "4 classes @ 128^3", "Stride / Kernel": "Adaptive 4/2 Cascade", "Parameters": 466228, "GMAC Cost": "7.973 GMAC (-18.4%)"}
]
pd.DataFrame(tab2_data).to_csv(os.path.join(TAB_DIR, "table2_backbone_config.csv"), index=False)

# -------------------------------------------------------------
# TABLE 3: Fixed-Depth Baseline Performance on Sealed Test Cohort
# -------------------------------------------------------------
tab3_data = [
    {"Configuration": "Fixed K1 (Stem + B1)", "Depth": "K1", "GMAC": m["k1"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "WT Dice": "Not retained", "TC Dice": "Not retained", "ET Dice": "Not retained", "Mean Dice": f"{m['k1']['mean_dice']:.4f} +/- {m['k1']['std_dice']:.4f}"},
    {"Configuration": "Fixed K2 (Stem + B1 + B2)", "Depth": "K2", "GMAC": m["k2"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "WT Dice": "Not retained", "TC Dice": "Not retained", "ET Dice": "Not retained", "Mean Dice": f"{m['k2']['mean_dice']:.4f} +/- {m['k2']['std_dice']:.4f}"},
    {"Configuration": "Fixed K3 (Stem + B1 + B2 + B3)", "Depth": "K3", "GMAC": m["k3"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['k3']['brier']:.6f}", "WT Dice": "Not retained", "TC Dice": "Not retained", "ET Dice": "Not retained", "Mean Dice": f"{m['k3']['mean_dice']:.4f} +/- {m['k3']['std_dice']:.4f}"}
]
pd.DataFrame(tab3_data).to_csv(os.path.join(TAB_DIR, "table3_fixed_depth_performance.csv"), index=False)

# -------------------------------------------------------------
# TABLE 4: Predictor Development & Out-of-Fold Performance (Dev Cohort)
# -------------------------------------------------------------
tab4_data = [
    {"Gate Stage": "Gate 1", "Candidate Formulation": "G1-A: Family-B1 7-D Huber MLP (Selected)", "Input Dims": 7, "Pearson r": 0.4731, "Spearman rho": 0.4164, "Pairwise Rank Acc": "70.9%", "Top-k Oracle Overlap": "71.9%", "Selected Gain Adv.": "+3.95e-4", "Scan Win Rate": "87.5% (7/8)", "Status": "Selected for Deployment"},
    {"Gate Stage": "Gate 1", "Candidate Formulation": "G1-B: Complexity-Only 3-D Huber MLP", "Input Dims": 3, "Pearson r": 0.2842, "Spearman rho": 0.2114, "Pairwise Rank Acc": "58.3%", "Top-k Oracle Overlap": "56.2%", "Selected Gain Adv.": "+1.12e-4", "Scan Win Rate": "50.0% (4/8)", "Status": "Evaluated for Ablation C"},
    {"Gate Stage": "Gate 1", "Candidate Formulation": "G1-C: Redesign Z1 11-D Huber MLP", "Input Dims": 11, "Pearson r": 0.4781, "Spearman rho": 0.4109, "Pairwise Rank Acc": "68.5%", "Top-k Oracle Overlap": "62.5%", "Selected Gain Adv.": "+2.77e-4", "Scan Win Rate": "75.0% (6/8)", "Status": "Exploratory Candidate"},
    {"Gate Stage": "Gate 2", "Candidate Formulation": "G2-A: Z2 16-D Ridge (alpha=10)", "Input Dims": 16, "Pearson r": 0.1640, "Spearman rho": 0.1096, "Pairwise Rank Acc": "60.4%", "Top-k Oracle Overlap": "62.5%", "Selected Gain Adv.": "+5.30e-5", "Scan Win Rate": "75.0% (6/8)", "Status": "Overfitted Across Folds"},
    {"Gate Stage": "Gate 2", "Candidate Formulation": "G2-B: P2-Only 7-D Ridge (alpha=10)", "Input Dims": 7, "Pearson r": 0.5196, "Spearman rho": 0.3864, "Pairwise Rank Acc": "72.9%", "Top-k Oracle Overlap": "75.0%", "Selected Gain Adv.": "+6.58e-5", "Scan Win Rate": "75.0% (6/8)", "Status": "Selected for Ablation D"},
    {"Gate Stage": "Gate 2", "Candidate Formulation": "G2-C: Z2 16-D Strong Ridge (alpha=100)", "Input Dims": 16, "Pearson r": 0.2289, "Spearman rho": 0.2093, "Pairwise Rank Acc": "66.7%", "Top-k Oracle Overlap": "62.5%", "Selected Gain Adv.": "+5.30e-5", "Scan Win Rate": "75.0% (6/8)", "Status": "Exploratory Candidate"},
    {"Gate Stage": "Gate 2", "Candidate Formulation": "G2-D: Z2 16-D Linear Pairwise RankNet", "Input Dims": 16, "Pearson r": 0.6537, "Spearman rho": 0.3464, "Pairwise Rank Acc": "79.2%", "Top-k Oracle Overlap": "75.0%", "Selected Gain Adv.": "+6.87e-5", "Scan Win Rate": "87.5% (7/8)", "Status": "Selected for Deployment"},
    {"Gate Stage": "Gate 2", "Candidate Formulation": "G2-E: Z2 16-D Binary Logistic Regression", "Input Dims": 16, "Pearson r": -0.5847, "Spearman rho": -0.4058, "Pairwise Rank Acc": "41.7%", "Top-k Oracle Overlap": "50.0%", "Selected Gain Adv.": "-3.46e-5", "Scan Win Rate": "25.0% (2/8)", "Status": "Rejected (Inappropriate Target)"}
]
pd.DataFrame(tab4_data).to_csv(os.path.join(TAB_DIR, "table4_predictor_dev_performance.csv"), index=False)

# -------------------------------------------------------------
# TABLE 5: Adaptive Routing vs Matched Random & Retrospective Oracle
# -------------------------------------------------------------
tab5_data = [
    {"Model / Pipeline": "Fixed Full-Depth K3", "Budget (M1/M2)": "8 / 8", "GMAC": m["k3"]["gmac"], "Savings vs K3": "0.0%", "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['k3']['brier']:.6f}", "Mean Dice": f"{m['k3']['mean_dice']:.4f} +/- {m['k3']['std_dice']:.4f}", "Win Rate vs Random": "-", "Gap Closed": "-"},
    {"Model / Pipeline": "Matched Random Routing (5 seeds)", "Budget (M1/M2)": "4 / 2", "GMAC": m["random_4_2"]["gmac"], "Savings vs K3": "18.4%", "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['random_4_2']['mean_dice']:.4f} +/- {m['random_4_2']['std_dice']:.4f}", "Win Rate vs Random": "Reference", "Gap Closed": "0.0%"},
    {"Model / Pipeline": "Proposed VaDeMamba", "Budget (M1/M2)": "4 / 2", "GMAC": m["vademamba"]["gmac"], "Savings vs K3": "18.4%", "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['vademamba']['brier']:.6f}", "Mean Dice": f"{m['vademamba']['mean_dice']:.4f} +/- {m['vademamba']['std_dice']:.4f}", "Win Rate vs Random": h2h["vademamba_win_rate_vs_random"], "Gap Closed": f"{h2h['random_to_oracle_gap_closed_percent']:.1f}%"},
    {"Model / Pipeline": "Retrospective oracle allocation (same 4/2 budget; ground-truth-derived)", "Budget (M1/M2)": "4 / 2", "GMAC": m["oracle"]["gmac"], "Savings vs K3": "18.4%", "Latency (ms)": "Not benchmarked (non-deployable)", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['oracle']['mean_dice']:.4f} +/- {m['oracle']['std_dice']:.4f}", "Win Rate vs Random": "100%", "Gap Closed": "100.0%"}
]
pd.DataFrame(tab5_data).to_csv(os.path.join(TAB_DIR, "table5_adaptive_vs_random_oracle.csv"), index=False)

# -------------------------------------------------------------
# TABLE 6: Comprehensive Ablation Study
# -------------------------------------------------------------
tab6_data = [
    {"Ablation": "A. Allocation Strategy", "Variant": "Fixed Depth K1 everywhere", "Budget": "0 / 0", "GMAC": m["k1"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['k1']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "A. Allocation Strategy", "Variant": "Fixed Depth K2 everywhere", "Budget": "8 / 0", "GMAC": m["k2"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['k2']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "A. Allocation Strategy", "Variant": "Fixed Depth K3 everywhere", "Budget": "8 / 8", "GMAC": m["k3"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['k3']['brier']:.6f}", "Mean Dice": f"{m['k3']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "A. Allocation Strategy", "Variant": "Matched Random Routing (5 seeds)", "Budget": "4 / 2", "GMAC": m["random_4_2"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['random_4_2']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "A. Allocation Strategy", "Variant": "Proposed VaDeMamba (4/2)", "Budget": "4 / 2", "GMAC": m["vademamba"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['vademamba']['brier']:.6f}", "Mean Dice": f"{m['vademamba']['mean_dice']:.4f}", "WT": f"{df_per_scan['VaDeMamba_WT'].mean():.4f}", "TC": f"{df_per_scan['VaDeMamba_TC'].mean():.4f}", "ET": f"{df_per_scan['VaDeMamba_ET'].mean():.4f}"},
    {"Ablation": "A. Allocation Strategy", "Variant": "Retrospective oracle allocation (same 4/2 budget)", "Budget": "4 / 2", "GMAC": m["oracle"]["gmac"], "Latency (ms)": "Not benchmarked (non-deployable)", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['oracle']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},

    # B. Cascade Depth
    {"Ablation": "B. Cascade Depth", "Variant": "Gate-1 Only (4 K2, 0 K3)", "Budget": "4 / 0", "GMAC": m["gate1_only"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['gate1_only']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "B. Cascade Depth", "Variant": "Gate-1 + Random Gate-2", "Budget": "4 / 2", "GMAC": m["g1_plus_random_g2"]["gmac"], "Latency (ms)": "Not benchmarked", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['g1_plus_random_g2']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "B. Cascade Depth", "Variant": "Gate-1 + learned Z2 RankNet (Proposed)", "Budget": "4 / 2", "GMAC": m["vademamba"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['vademamba']['brier']:.6f}", "Mean Dice": f"{m['vademamba']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},

    # C. Gate-1 Features
    {"Ablation": "C. Gate-1 Features", "Variant": "Complexity-Only Gate 1 (3 dims)", "Budget": "4 / 2", "GMAC": m["ablation_c_comp_g1"]["gmac"], "Latency (ms)": "Not benchmarked", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['ablation_c_comp_g1']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "C. Gate-1 Features", "Variant": "Predictive Family-B1 Gate 1 (7 dims; Proposed)", "Budget": "4 / 2", "GMAC": m["vademamba"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['vademamba']['brier']:.6f}", "Mean Dice": f"{m['vademamba']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},

    # D. Gate-2 Information
    {"Ablation": "D. Gate-2 Formulation", "Variant": "P2-only Ridge (7 dims; ablation)", "Budget": "4 / 2", "GMAC": m["ablation_d_p2only_g2"]["gmac"], "Latency (ms)": "Not benchmarked", "Multiclass Brier error": "Not retained", "Mean Dice": f"{m['ablation_d_p2only_g2']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"},
    {"Ablation": "D. Gate-2 Formulation", "Variant": "Z2 Linear Pairwise RankNet (16 dims; Proposed)", "Budget": "4 / 2", "GMAC": m["vademamba"]["gmac"], "Latency (ms)": "Not benchmarked in this environment", "Multiclass Brier error": f"{m['vademamba']['brier']:.6f}", "Mean Dice": f"{m['vademamba']['mean_dice']:.4f}", "WT": "Not retained", "TC": "Not retained", "ET": "Not retained"}
]
pd.DataFrame(tab6_data).to_csv(os.path.join(TAB_DIR, "table6_ablation_study.csv"), index=False)

# -------------------------------------------------------------
# TABLE 7: Computational Efficiency Breakdown by Stage (Analytical GMAC)
# -------------------------------------------------------------
tab7_data = [
    {"Model Stage / Module": "Patch Stem (2-Stage Conv3D)", "Active Spatial Extent": "Global (100%)", "Parameters": 129744, "GMAC Cost": 5.440, "Stage Savings": "0.0%", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Block 1 (Global Bidirectional Mamba)", "Active Spatial Extent": "8 of 8 Macro-tiles (100%)", "Parameters": 112032, "GMAC Cost": 1.440, "Stage Savings": "0.0%", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Shared Readout Head (H)", "Active Spatial Extent": "Global (100%)", "Parameters": 388, "GMAC Cost": 0.013, "Stage Savings": "0.0%", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Gate 1 Value Predictor (7-D Huber MLP)", "Active Spatial Extent": "8 Macro-tiles", "Parameters": 273, "GMAC Cost": "< 0.0001", "Stage Savings": "-", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Block 2 (Group-Batched Mamba, M1=4)", "Active Spatial Extent": "4 of 8 Macro-tiles (50%)", "Parameters": 112032, "GMAC Cost": 0.720, "Stage Savings": "50.0% stage bypass", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Gate 2 Value Predictor (16-D Z2 RankNet)", "Active Spatial Extent": "4 Candidate Macro-tiles", "Parameters": 16, "GMAC Cost": "< 0.0001", "Stage Savings": "-", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Block 3 (Group-Batched Mamba, M2=2)", "Active Spatial Extent": "2 of 8 Macro-tiles (25%)", "Parameters": 112032, "GMAC Cost": 0.360, "Stage Savings": "75.0% stage bypass", "Latency Note": "Analytical GMAC only"},
    {"Model Stage / Module": "Total VaDeMamba Cascade (4/2)", "Active Spatial Extent": "4 K1 + 2 K2 + 2 K3", "Parameters": 466228, "GMAC Cost": 7.973, "Stage Savings": "18.4% whole-model GMAC reduction", "Latency Note": "End-to-end latency not claimed (benchmark stalled)"},
    {"Model Stage / Module": "Total Fixed Full-Depth K3 Baseline", "Active Spatial Extent": "8 K3 Everywhere (100%)", "Parameters": 466228, "GMAC Cost": 9.773, "Stage Savings": "Reference baseline (0.0%)", "Latency Note": "End-to-end latency not claimed"}
]
pd.DataFrame(tab7_data).to_csv(os.path.join(TAB_DIR, "table7_computational_efficiency.csv"), index=False)

# -------------------------------------------------------------
# TABLE 8: Per-Scan Final-Test Results
# -------------------------------------------------------------
df_tab8 = df_per_scan[[
    "scan_id", "subject_id", "K1_Dice", "K2_Dice", "K3_Dice",
    "VaDeMamba_Dice", "Random_Dice", "Oracle_Dice",
    "VaDeMamba_WT", "VaDeMamba_TC", "VaDeMamba_ET", "VaDeMamba_Wins_vs_Random"
]].copy()
df_tab8.rename(columns={
    "VaDeMamba_Dice": "VaDeMamba_Dice",
    "VaDeMamba_WT": "VaDeMamba_WT",
    "VaDeMamba_TC": "VaDeMamba_TC",
    "VaDeMamba_ET": "VaDeMamba_ET",
    "VaDeMamba_Wins_vs_Random": "VaDeMamba_Wins_vs_Random"
}, inplace=True)
df_tab8.to_csv(os.path.join(TAB_DIR, "table8_per_scan_results.csv"), index=False)

# -------------------------------------------------------------
# TABLE 9: Predictor / Oracle Agreement on Test Cohort
# -------------------------------------------------------------
g1_ovlps = df_tiles.groupby("scan_id")["g1_oracle_overlap"].mean().values
g2_ovlps = df_tiles.groupby("scan_id")["g2_oracle_overlap"].mean().values
g1_adv = df_tiles.groupby("scan_id").apply(lambda d: d.loc[d.in_s1_proposed, "actual_delta_e12"].mean() - d.loc[~d.in_s1_proposed, "actual_delta_e12"].mean()).mean()
g2_adv = df_tiles.groupby("scan_id").apply(lambda d: d.loc[d.in_s2_proposed, "actual_delta_e23"].mean() - d.loc[d.in_s1_proposed & ~d.in_s2_proposed, "actual_delta_e23"].mean()).mean()
tab9_data = [
    {"Gate Stage": "Gate 1 (Delta_E_12)", "Budget": "M1 = 4 of 8", "Mean Oracle Top-k Overlap": f"{np.mean(g1_ovlps)*100:.1f}%", "Mean selected-minus-unselected benefit": f"{g1_adv:+.2e}", "Scope": "Final-test alignment; not used for model selection"},
    {"Gate Stage": "Gate 2 (Delta_E_23 on S1)", "Budget": "M2 = 2 of 4", "Mean Oracle Top-k Overlap": f"{np.mean(g2_ovlps)*100:.1f}%", "Mean selected-minus-unselected benefit": f"{g2_adv:+.2e}", "Scope": "Final-test alignment; not used for model selection"}
]
pd.DataFrame(tab9_data).to_csv(os.path.join(TAB_DIR, "table9_predictor_oracle_agreement.csv"), index=False)

# -------------------------------------------------------------
# TABLE 10: Campaign Summary Dashboard
# -------------------------------------------------------------
tab10_data = [
    {"Question": "Average marginal depth returns", "Evidence": "Retained tile-level multiclass-Brier improvements", "Finding": "The mean marginal Brier improvement from K1->K2 exceeds that from K2->K3, indicating diminishing average marginal returns with depth."},
    {"Question": "Spatial heterogeneity", "Evidence": "Macro-tile Delta_E distributions across final-test scans", "Finding": "Tile-to-tile marginal-benefit heterogeneity is observed across 3D anatomical quadrants."},
    {"Question": "Gate 1 selection", "Evidence": "DEV subject-grouped OOF cross-validation", "Finding": "The locked Family-B1 Huber MLP (7-D) was selected before final testing based on OOF ranking accuracy."},
    {"Question": "Gate 2 selection", "Evidence": "DEV subject-grouped OOF cross-validation", "Finding": "The locked 16-D Z2 Linear Pairwise RankNet was selected before final testing; Ridge remains an ablation."},
    {"Question": "Matched-random comparison", "Evidence": "Retained per-scan final-test Dice (N=12)", "Finding": f"VaDeMamba achieves higher Dice on {h2h['vademamba_win_rate_vs_random']} scans; held-out evidence, not a population-level claim."},
    {"Question": "Analytical compute complexity", "Evidence": "Analytical GMAC accounting", "Finding": "VaDeMamba operates at 7.973 GMAC vs 9.773 GMAC for K3 (18.4% whole-model GMAC reduction)."},
    {"Question": "Wall-clock latency", "Evidence": "Dedicated timing harness status", "Finding": "No wall-clock latency claim is made (CUDA initialization stalled during isolated timing pass)."}
]
pd.DataFrame(tab10_data).to_csv(os.path.join(TAB_DIR, "table10_campaign_summary.csv"), index=False)
print("All 10 tables successfully saved to results/final_campaign/tables/.")

# =============================================================
# GENERATING PUBLICATION FIGURES (1 TO 10) AT 300 DPI
# =============================================================
print("\n" + "=" * 80)
print("GENERATING VADEMAMBA PUBLICATION FIGURES (1 TO 10) AT 300 DPI")
print("=" * 80)

# FIGURE 1: Architecture Diagram
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
ax.axis('off')
boxes = [
    ("Input Volume\n4x128x128x128", 0.03, 0.40, 0.13, 0.25, "#e2e8f0"),
    ("2-Stage Stem\nConv3D (s=4)\nF0: 96x32^3", 0.19, 0.40, 0.14, 0.25, "#cbd5e1"),
    ("Global Block 1\nFull-Volume SSM\nF1: 96x32^3", 0.36, 0.40, 0.14, 0.25, "#bfdbfe"),
    ("Gate 1 Predictor\n7-D Huber MLP\nRank 8 Tiles", 0.36, 0.08, 0.14, 0.22, "#fde68a"),
    ("Block 2 (Tiled)\nGroup-Batched SSM\n4 Tiles (S1)", 0.53, 0.55, 0.14, 0.25, "#93c5fd"),
    ("Bypass Block 2\n4 tiles retain F1\n(Term: K1)", 0.53, 0.23, 0.14, 0.22, "#e2e8f0"),
    ("Gate 2 Predictor\n16-D RankNet\nRank S1 Tiles", 0.53, 0.02, 0.14, 0.18, "#fed7aa"),
    ("Block 3 (Tiled)\nGroup-Batched SSM\n2 Tiles (S2)", 0.70, 0.65, 0.14, 0.25, "#60a5fa"),
    ("Bypass Block 3\n2 tiles retain F2\n(Term: K2)", 0.70, 0.35, 0.14, 0.22, "#cbd5e1"),
    ("Shared Head H\nConv3D 1x1x1\n+ Trilinear x4", 0.87, 0.40, 0.11, 0.25, "#a7f3d0"),
]
for text, x0, y0, w, h, col in boxes:
    rect = patches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02", ec="#334155", fc=col, lw=1.5)
    ax.add_patch(rect)
    ax.text(x0 + w/2, y0 + h/2, text, ha='center', va='center', fontsize=8, fontweight='bold', color="#0f172a")

ax.set_title("Figure 1: VaDeMamba Architecture & Spatially Adaptive Tiled Routing Pipeline", fontsize=12, pad=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure1_architecture_pipeline.png"), dpi=300)
plt.close()

# FIGURE 2: Depth-Response Curve
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
depths = ["K1 -> K2", "K2 -> K3"]
improvements = [df_tiles["actual_delta_e12"].mean(), df_tiles["actual_delta_e23"].mean()]
ax.bar(depths, improvements, color=['#1e3a8a', '#60a5fa'], width=0.45)
ax.set_ylabel('Mean Tile Multiclass Brier Improvement', color='#1e3a8a', fontweight='bold')
ax.tick_params(axis='y', labelcolor='#1e3a8a')
ax.grid(True, alpha=0.3)
ax2 = ax.twinx()
ax2.plot(["K1", "K2", "K3"], [m["k1"]["mean_dice"], m["k2"]["mean_dice"], m["k3"]["mean_dice"]], marker='s', color='#047857', lw=2.2, ls='--', label='Mean Dice Score')
ax2.set_ylabel('Cohort Mean Dice Score (Higher is Better)', color='#047857', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#047857')
ax.set_title("Figure 2: Empirical Depth-Response Curves (Final Test Cohort, N=12 Scans / 8 Subjects)", fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure2_depth_response_curve.png"), dpi=300)
plt.close()

# FIGURE 3: Spatial Heterogeneity
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
tile_ids = sorted(df_tiles["tile_idx"].unique())
d12_by_tile = [df_tiles[df_tiles["tile_idx"] == t]["actual_delta_e12"].values * 1e4 for t in tile_ids]
d23_by_tile = [df_tiles[df_tiles["tile_idx"] == t]["actual_delta_e23"].values * 1e5 for t in tile_ids]
axes[0].boxplot(d12_by_tile, tick_labels=[f"T{t}" for t in tile_ids], patch_artist=True,
                boxprops=dict(facecolor='#93c5fd', color='#1e40af'), medianprops=dict(color='#b91c1c', lw=1.5))
axes[0].set_title("Spatial Heterogeneity: Stage 1->2 Gain (Delta_E12)")
axes[0].set_xlabel("Macro-Tile ID (16^3 Latent Sub-volume)")
axes[0].set_ylabel("Actual Delta_E12 (x10^-4)")
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(d23_by_tile, tick_labels=[f"T{t}" for t in tile_ids], patch_artist=True,
                boxprops=dict(facecolor='#fed7aa', color='#c2410c'), medianprops=dict(color='#b91c1c', lw=1.5))
axes[1].set_title("Spatial Heterogeneity: Stage 2->3 Residual Gain (Delta_E23)")
axes[1].set_xlabel("Macro-Tile ID (16^3 Latent Sub-volume)")
axes[1].set_ylabel("Actual Delta_E23 (x10^-5)")
axes[1].grid(True, alpha=0.3)
fig.suptitle("Figure 3: Spatial Heterogeneity of Marginal Mamba Depth Benefits (Test Cohort)", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure3_spatial_heterogeneity.png"), dpi=300)
plt.close()

# FIGURE 4: Gate 1 Predicted vs Actual
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
scans = df_tiles["scan_id"].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(scans)))
for ci, sid in enumerate(scans):
    st = df_tiles[df_tiles["scan_id"] == sid]
    ax.scatter(st["actual_delta_e12"] * 1e4, st["pred_delta_e12"] * 1e4, color=colors[ci], s=35, alpha=0.8, label=sid[-7:])
mn = min(df_tiles["actual_delta_e12"].min(), df_tiles["pred_delta_e12"].min()) * 1e4
mx = max(df_tiles["actual_delta_e12"].max(), df_tiles["pred_delta_e12"].max()) * 1e4
ax.plot([mn, mx], [mn, mx], 'k--', lw=1, label='Identity')
pr_g1 = float(np.corrcoef(df_tiles["actual_delta_e12"], df_tiles["pred_delta_e12"])[0, 1])
ax.set_title(f"Figure 4: Gate 1 Final-Test Alignment (Held-Out Evaluation)\nPearson r = {pr_g1:.3f} | Top-4 Oracle Overlap = {np.mean(g1_ovlps)*100:.1f}%", fontweight='bold')
ax.set_xlabel("Actual Measured Delta_E12 (x10^-4)")
ax.set_ylabel("Predicted Delta_E12 (x10^-4)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure4_gate1_predicted_vs_actual.png"), dpi=300)
plt.close()

# FIGURE 5: Gate 2 Candidate Development Evaluation
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
if df_g2_dev is not None:
    cands = [c.split(":")[0] for c in df_g2_dev["Candidate"]]
    pr_vals = df_g2_dev["Pearson r"].values
    acc_vals = [float(v.replace("%", "")) / 100.0 for v in df_g2_dev["Pairwise Rank Acc"]]
    win_vals = [float(v.split("%")[0]) / 100.0 for v in df_g2_dev["Scan Win Rate"]]
    
    x_idx = np.arange(len(cands))
    w = 0.26
    ax.bar(x_idx - w, pr_vals, width=w, label='Pearson r', color='#3b82f6')
    ax.bar(x_idx, acc_vals, width=w, label='Pairwise Rank Acc', color='#10b981')
    ax.bar(x_idx + w, win_vals, width=w, label='Scan Win Rate', color='#f59e0b')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(cands, rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax.set_ylabel("Metric Score / Proportion")
    ax.set_title("Figure 5: Gate-2 Formulation Comparison (DEV Subject-Grouped OOF CV)", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure5_gate2_predicted_vs_actual.png"), dpi=300)
plt.close()

# FIGURE 6: Depth Allocation Histogram & Frequency Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
counts = [48, 24, 24] # Canonical 4 K1, 2 K2, 2 K3 across 12 scans = 48, 24, 24
bars = axes[0].bar(['K1 (Block 1)', 'K2 (Block 1+2)', 'K3 (Block 1+2+3)'], counts,
                   color=['#94a3b8', '#60a5fa', '#1d4ed8'], width=0.55)
for b in bars:
    yval = b.get_height()
    pct = yval / 96.0 * 100.0
    axes[0].text(b.get_x() + b.get_width()/2.0, yval + 1.5, f"{yval} tiles\n({pct:.1f}%)", ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_ylim(0, 60)
axes[0].set_ylabel("Total Allocated Macro-Tiles (96 Total across 12 Scans)")
axes[0].set_title("Cohort Depth Allocation Histogram")
axes[0].grid(True, alpha=0.3, axis='y')

tile_k3_counts = [df_tiles[(df_tiles["tile_idx"] == t) & df_tiles["in_s2_proposed"]]["scan_id"].count() for t in range(8)]
tile_k2_counts = [df_tiles[(df_tiles["tile_idx"] == t) & df_tiles["in_s1_proposed"] & (~df_tiles["in_s2_proposed"])]["scan_id"].count() for t in range(8)]
tile_k1_counts = [df_tiles[(df_tiles["tile_idx"] == t) & (~df_tiles["in_s1_proposed"])]["scan_id"].count() for t in range(8)]

x_t = np.arange(8)
axes[1].bar(x_t, tile_k1_counts, label='K1 (Exited Early)', color='#94a3b8', width=0.6)
axes[1].bar(x_t, tile_k2_counts, bottom=tile_k1_counts, label='K2 (Terminated)', color='#60a5fa', width=0.6)
axes[1].bar(x_t, tile_k3_counts, bottom=np.array(tile_k1_counts) + np.array(tile_k2_counts), label='K3 (Full Depth)', color='#1d4ed8', width=0.6)
axes[1].set_xticks(x_t)
axes[1].set_xticklabels([f"Tile {t}" for t in range(8)], fontsize=9)
axes[1].set_ylabel("Number of Scans (out of 12)")
axes[1].set_title("Regional Depth Assignment Across Macro-Tiles")
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(True, alpha=0.3, axis='y')

fig.suptitle("Figure 6: Macro-Tile Depth Allocation Histogram & Regional Assignment Distribution", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure6_final_spatial_allocation.png"), dpi=300)
plt.close()

# FIGURE 7: Qualitative Segmentation Comparison
if visual_cache is not None and len(visual_cache) > 0:
    sid = list(visual_cache.keys())[0]
    cd = visual_cache[sid]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
    
    axes[0, 0].imshow(cd["t1ce"], cmap='gray')
    axes[0, 0].set_title(f"T1ce Input (z={cd['slice_z']})")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cd["flair"], cmap='gray')
    axes[0, 1].set_title("FLAIR Input")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cd["t1ce"], cmap='gray')
    axes[0, 2].imshow(cd["gt"], cmap='jet', alpha=0.5)
    axes[0, 2].set_title("Ground Truth Segmentation")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(cd["t1ce"], cmap='gray')
    axes[0, 3].imshow(cd["pred_k1"], cmap='jet', alpha=0.5)
    axes[0, 3].set_title("Fixed K1 Prediction")
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(cd["t1ce"], cmap='gray')
    axes[1, 0].imshow(cd["pred_random"], cmap='jet', alpha=0.5)
    axes[1, 0].set_title("Matched Random Prediction")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cd["t1ce"], cmap='gray')
    axes[1, 1].imshow(cd["pred_k3"], cmap='jet', alpha=0.5)
    axes[1, 1].set_title("Fixed K3 Prediction (Full)")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cd["t1ce"], cmap='gray')
    axes[1, 2].imshow(cd["pred_vademamba"], cmap='jet', alpha=0.5)
    axes[1, 2].set_title("VaDeMamba (Proposed)")
    axes[1, 2].axis('off')
    
    # Active tiles diagram
    axes[1, 3].imshow(cd["t1ce"], cmap='gray')
    axes[1, 3].set_title(f"Allocated Depth Regions\nS1={cd['s1_tiles']}, S2={cd['s2_tiles']}")
    axes[1, 3].axis('off')
    
    fig.suptitle(f"Figure 7: Qualitative Brain Tumor Segmentation on Final Test Scan {sid}", fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "figure7_qualitative_segmentation.png"), dpi=300)
    plt.close()

# FIGURE 8: Quality vs Compute (Analytical GMAC)
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
pts = [
    ("Fixed K1", m["k1"]["gmac"], m["k1"]["mean_dice"], "#64748b", "o"),
    ("Fixed K2", m["k2"]["gmac"], m["k2"]["mean_dice"], "#0284c7", "s"),
    ("Fixed K3", m["k3"]["gmac"], m["k3"]["mean_dice"], "#1e3a8a", "D"),
    ("Random 4/2", m["random_4_2"]["gmac"], m["random_4_2"]["mean_dice"], "#d97706", "^"),
    ("VaDeMamba 4/2", m["vademamba"]["gmac"], m["vademamba"]["mean_dice"], "#16a34a", "*"),
    ("Retrospective 4/2 Oracle", m["oracle"]["gmac"], m["oracle"]["mean_dice"], "#9333ea", "P")
]
for name, gmac, dice, col, mark in pts:
    size = 140 if mark in ["*", "P"] else 80
    ax.scatter(gmac, dice, color=col, marker=mark, s=size, label=name, zorder=5)
    ax.annotate(name, (gmac, dice), textcoords="offset points", xytext=(8, -3), fontsize=8, fontweight='bold', color=col)
ax.set_xlabel("Computational Complexity (Analytical GMAC per Volume)")
ax.set_ylabel("Cohort Mean Dice Score")
ax.set_title("Figure 8: Segmentation Quality vs Analytical GMAC Complexity", fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure8_quality_compute_pareto.png"), dpi=300)
plt.close()

# FIGURE 9: Gate 2 Ablation (Development and Final-Test Evidence Kept Separate)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
if df_g2_dev is not None:
    dev = df_g2_dev[df_g2_dev["Candidate"].str.contains("G2-A|G2-B|G2-D", regex=True)].copy()
    labels = [x.split(":")[0] for x in dev["Candidate"]]
    axes[0].bar(labels, [float(x.replace("%", ""))/100 for x in dev["Pairwise Rank Acc"]], color=['#94a3b8', '#60a5fa', '#1d4ed8'])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("DEV OOF Pairwise Rank Accuracy")
    axes[0].set_title("Predictor-Development Evidence")
    for i, value in enumerate(dev["Top-2 Oracle Overlap"]):
        axes[0].text(i, .03, f"top-2: {value}", ha='center', rotation=90, color='white', fontsize=8)

axes[1].bar(["P2-only Ridge\n(ablation)", "Z2 Pairwise RankNet\n(deployed)"], [m["ablation_d_p2only_g2"]["mean_dice"], m["vademamba"]["mean_dice"]], color=['#60a5fa', '#1d4ed8'])
axes[1].set_ylabel("Final-Test Mean Dice")
axes[1].set_title("Retained Final-Test Segmentation Evidence")
axes[1].set_ylim(min(m["ablation_d_p2only_g2"]["mean_dice"], m["vademamba"]["mean_dice"]) - .01, max(m["ablation_d_p2only_g2"]["mean_dice"], m["vademamba"]["mean_dice"]) + .01)
fig.suptitle("Figure 9: Gate-2 Ablation — Development and Final-Test Evidence Kept Separate", fontweight='bold')
for a in axes:
    a.grid(True, alpha=.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure9_gate2_information_ablation.png"), dpi=300)
plt.close()

# FIGURE 10: Per-Scan Head-to-Head Win Margins
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
scan_labels = [s[-7:] for s in df_per_scan["scan_id"]]
dice_diff = df_per_scan["VaDeMamba_Dice"] - df_per_scan["Random_Dice"]
colors_diff = ['#16a34a' if d > 0 else '#dc2626' for d in dice_diff]
bars = ax.bar(scan_labels, dice_diff * 100.0, color=colors_diff, width=0.6)
ax.axhline(0, color='black', lw=1)
ax.set_ylabel("Dice Advantage vs Matched Random (x10^-2)")
ax.set_xlabel("Final Test Scan Identifier")
ax.set_title(f"Figure 10: Per-Scan Dice Advantage of VaDeMamba vs Matched Random Routing\n(Win Rate: {h2h['vademamba_win_rate_vs_random']})", fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figure10_perscan_consistency.png"), dpi=300)
plt.close()

print("All 10 figures saved successfully to results/final_campaign/figures/.")
print("=" * 80)

# Generate Markdown companions from CSVs
for csv_name in sorted(name for name in os.listdir(TAB_DIR) if name.endswith(".csv")):
    table = pd.read_csv(os.path.join(TAB_DIR, csv_name))
    md_name = os.path.splitext(csv_name)[0] + ".md"
    with open(os.path.join(TAB_DIR, md_name), "w", encoding="utf-8") as handle:
        handle.write(table.to_markdown(index=False) + "\n")
print("Markdown companions generated from all CSV tables.")
