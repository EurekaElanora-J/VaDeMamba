"""Schema validation and statistics for retained final-manuscript artifacts.

These utilities deliberately do not run inference, load checkpoints, or write
artifacts.  They summarize separately retained evaluation files when an
authorized user supplies them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json

import numpy as np
import pandas as pd


TABLE6_METHODS = (
    "Full-Depth Mamba (K3)",
    "Early-Exit Mamba (Entropy)",
    "Early-Exit Mamba (MSP)",
    "VaDeMamba (Proposed)",
)
TABLE6_COLUMNS = (
    "Method",
    "Strategy",
    "Params",
    "FLOPs/MACs",
    "Inference Time",
    "Avg Executed Depth",
    "Computation Saved (%)",
    "WT Dice",
    "TC Dice",
    "ET Dice",
    "Mean Dice",
    "Peak GPU Mem (MB)",
)


def normalize_historical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with historical ``AdaDepth_*`` fields renamed in memory.

    The retained files are sealed.  This adapter never writes them and refuses
    ambiguous input that already contains both historical and current names.
    """
    rename = {
        column: "VaDeMamba_" + column.removeprefix("AdaDepth_")
        for column in frame.columns
        if column.startswith("AdaDepth_")
    }
    clashes = set(rename.values()).intersection(frame.columns)
    if clashes:
        raise ValueError(f"Historical/current column collision: {sorted(clashes)}")
    return frame.rename(columns=rename).copy()


def load_controlled_comparison(path: str | Path) -> pd.DataFrame:
    """Load and validate Table 6 rows from a retained reviewer metrics JSON."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        rows = payload["experiment1_controlled_comparison"]["table1"]
    except (KeyError, TypeError) as error:
        raise ValueError("Missing experiment1_controlled_comparison.table1") from error
    table = pd.DataFrame(rows)
    missing = set(TABLE6_COLUMNS).difference(table.columns)
    if missing:
        raise ValueError(f"Table 6 is missing columns: {sorted(missing)}")
    if tuple(table["Method"]) != TABLE6_METHODS:
        raise ValueError("Table 6 methods or ordering do not match the controlled comparison")
    return table.loc[:, TABLE6_COLUMNS].copy()


def _pearson(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return _pearson(x.rank(method="average"), y.rank(method="average"))


def _require_columns(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _pairwise_accuracy(scores: pd.Series, benefits: pd.Series) -> float:
    score = scores.to_numpy(dtype=float)
    benefit = benefits.to_numpy(dtype=float)
    correct, comparable = 0, 0
    for i in range(len(score)):
        for j in range(i + 1, len(score)):
            direction = (score[i] - score[j]) * (benefit[i] - benefit[j])
            if direction != 0:
                comparable += 1
                correct += direction > 0
    return float(correct / comparable) if comparable else float("nan")


def summarize_mechanism(
    gate1: pd.DataFrame,
    gate2: pd.DataFrame,
) -> dict[str, Mapping[str, float | int | None]]:
    """Compute Table 8 descriptive quantities from retained tile records.

    Gate 1 is evaluated as a continuous predicted-versus-observed ``Delta E``
    association.  Gate 2 is evaluated only as an ordinal score ranking.
    The retained files do not encode the original permutation protocol, so this
    function intentionally does not recompute or infer permutation p-values.
    """
    gate1 = normalize_historical_columns(gate1)
    gate2 = normalize_historical_columns(gate2)
    _require_columns(gate1, {"scan_id", "pred_g1_value", "actual_delta_e12"}, "Gate 1 artifact")
    _require_columns(gate2, {"scan_id", "g2_rank_score", "actual_delta_e23"}, "Gate 2 artifact")

    g1_per_scan = [_pearson(group["pred_g1_value"], group["actual_delta_e12"])
                   for _, group in gate1.groupby("scan_id", sort=False)]
    g2_per_scan = [_spearman(group["g2_rank_score"], group["actual_delta_e23"])
                   for _, group in gate2.groupby("scan_id", sort=False)]
    g2_pairwise = [_pairwise_accuracy(group["g2_rank_score"], group["actual_delta_e23"])
                   for _, group in gate2.groupby("scan_id", sort=False)]

    result: dict[str, Mapping[str, float | int | None]] = {
        "gate1_continuous_delta_e": {
            "tiles": len(gate1),
            "pooled_pearson_r": _pearson(gate1["pred_g1_value"], gate1["actual_delta_e12"]),
            "pooled_spearman_rho": _spearman(gate1["pred_g1_value"], gate1["actual_delta_e12"]),
            "mean_scan_pearson_r": float(np.mean(g1_per_scan)),
            "std_scan_pearson_r": float(np.std(g1_per_scan)),
            "positive_scans": int(sum(value > 0 for value in g1_per_scan)),
            "scans": len(g1_per_scan),
            "permutation_pvalue": None,
        },
        "gate2_ordinal_ranknet": {
            "candidate_tiles": len(gate2),
            "pooled_spearman_rho": _spearman(gate2["g2_rank_score"], gate2["actual_delta_e23"]),
            "mean_scan_spearman_rho": float(np.mean(g2_per_scan)),
            "std_scan_spearman_rho": float(np.std(g2_per_scan)),
            "mean_pairwise_accuracy": float(np.mean(g2_pairwise)),
            "std_pairwise_accuracy": float(np.std(g2_pairwise)),
            "permutation_pvalue": None,
        },
    }
    return result
