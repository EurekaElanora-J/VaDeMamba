"""Read-only reporting path for final-manuscript Tables 6 and 8.

This command consumes separately authorized, sealed retained artifacts.  It
does not run model inference, load checkpoints, create figures, or modify an
input/output artifact.  It prints the controlled-comparison table and mechanism
statistics to stdout only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.final_manuscript import load_controlled_comparison, summarize_mechanism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled-json", type=Path, required=True, help="Retained reviewer_raw_metrics.json")
    parser.add_argument("--gate1-csv", type=Path, required=True, help="Retained K1-to-K2 tile CSV")
    parser.add_argument("--gate2-csv", type=Path, required=True, help="Retained K2-to-K3 tile CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table6 = load_controlled_comparison(args.controlled_json)
    table8 = summarize_mechanism(
        pd.read_csv(args.gate1_csv),
        pd.read_csv(args.gate2_csv),
    )
    print("# Final-manuscript retained-artifact report")
    print("\n## Table 6: controlled comparison")
    print(table6.to_csv(index=False).rstrip())
    print("\nAnalytical GMAC, measured synchronized wall-clock latency, peak GPU memory, and segmentation metrics are distinct fields above.")
    print("\n## Table 8: mechanism validation")
    print(json.dumps(table8, indent=2))
    print("\nGate 1 is a continuous Delta E predictor. Gate 2 scores are ordinal RankNet values, not calibrated Delta E estimates.")


if __name__ == "__main__":
    main()
