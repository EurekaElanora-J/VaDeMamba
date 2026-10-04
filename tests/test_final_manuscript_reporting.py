"""Synthetic tests for read-only final-manuscript reporting helpers."""

from pathlib import Path
import json
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.final_manuscript import (
    TABLE6_METHODS,
    load_controlled_comparison,
    normalize_historical_columns,
    summarize_mechanism,
)


def _table6_rows():
    columns = [
        "Method", "Strategy", "Params", "FLOPs/MACs", "Inference Time",
        "Avg Executed Depth", "Computation Saved (%)", "WT Dice", "TC Dice",
        "ET Dice", "Mean Dice", "Peak GPU Mem (MB)",
    ]
    return [dict(zip(columns, [method, "strategy", "params", "gmac", "latency", "depth", "saved", "wt", "tc", "et", "dice", "memory"]))
            for method in TABLE6_METHODS]


def test_controlled_table_schema_is_loaded_from_artifact(tmp_path):
    path = tmp_path / "reviewer_raw_metrics.json"
    path.write_text(json.dumps({"experiment1_controlled_comparison": {"table1": _table6_rows()}}), encoding="utf-8")
    table = load_controlled_comparison(path)
    assert list(table["Method"]) == list(TABLE6_METHODS)
    assert table.shape == (4, 12)


def test_historical_adadepth_fields_are_renamed_in_memory_only():
    original = pd.DataFrame({"AdaDepth_Dice": [0.7], "scan_id": ["s1"]})
    normalized = normalize_historical_columns(original)
    assert "VaDeMamba_Dice" in normalized
    assert "AdaDepth_Dice" in original
    with pytest.raises(ValueError, match="collision"):
        normalize_historical_columns(pd.DataFrame({"AdaDepth_Dice": [0.7], "VaDeMamba_Dice": [0.7]}))


def test_mechanism_statistics_keep_gate2_ordinal():
    gate1 = pd.DataFrame({
        "scan_id": ["a"] * 4 + ["b"] * 4,
        "pred_g1_value": [1, 2, 3, 4, 1, 2, 3, 4],
        "actual_delta_e12": [1, 2, 3, 4, 4, 3, 2, 1],
    })
    gate2 = pd.DataFrame({
        "scan_id": ["a"] * 4 + ["b"] * 4,
        "g2_rank_score": [1, 2, 3, 4, 1, 2, 3, 4],
        "actual_delta_e23": [1, 2, 3, 4, 4, 3, 2, 1],
    })
    summary = summarize_mechanism(gate1, gate2)
    assert summary["gate1_continuous_delta_e"]["tiles"] == 8
    assert summary["gate1_continuous_delta_e"]["positive_scans"] == 1
    assert summary["gate2_ordinal_ranknet"]["candidate_tiles"] == 8
    assert summary["gate2_ordinal_ranknet"]["mean_pairwise_accuracy"] == pytest.approx(0.5)
    assert summary["gate2_ordinal_ranknet"]["permutation_pvalue"] is None
