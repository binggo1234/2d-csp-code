"""Aggregate I1-I5 outputs into Table-2/3 friendly CSVs.

This helper reads per-case totals files created by `run_stepF(cfg)`:
  outputs/<case>/totals_baselineA.csv
  outputs/<case>/totals_baselineB.csv
  outputs/<case>/totals_proposed_shared.csv

It produces:
  outputs/_summary/table2_i1_i5_mean_std.csv
  outputs/_summary/table2_i1_i5_by_case.csv

Usage:
  python -m experiments.aggregate_i1_i5_table2 --out_root outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="outputs")
    ap.add_argument("--cases", type=str, default="I1,I2,I3,I4,I5")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repo = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo / out_root).resolve()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    variants = {
        "baselineA": "totals_baselineA.csv",
        "baselineB": "totals_baselineB.csv",
        "proposed_shared": "totals_proposed_shared.csv",
    }

    rows = []
    for case in cases:
        for v, fn in variants.items():
            p = out_root / case / fn
            if not p.exists():
                logging.warning(f"missing: {p}")
                continue
            df = _read(p)
            df["case"] = case
            df["variant"] = v
            rows.append(df)

    if not rows:
        raise SystemExit("No totals files found. Run experiments first.")

    all_df = pd.concat(rows, ignore_index=True)

    # pick a stable set of columns used in the paper
    col_map = {
        "U_avg": "U",
        "L_shared_sum": "L_shared",
        "L_cut_sum": "L_cut",
        "L_air_sum": "L_air",
        "N_lift_sum": "N_lift",
        "T_est_sum": "T_total",
    }
    for src, dst in col_map.items():
        if src in all_df.columns:
            all_df[dst] = all_df[src]

    keep_cols = ["case", "variant", "seed"] + [c for c in col_map.values() if c in all_df.columns]
    slim = all_df[keep_cols].copy()

    # --- per-case mean/std (useful for ablation or case-wise plots) ---
    by_case = slim.groupby(["case", "variant"]).agg(["mean", "std"])
    by_case.columns = [f"{a}_{b}" for a, b in by_case.columns]
    by_case = by_case.reset_index()

    # --- pooled mean/std over all cases+seeds (Table 2: mean±std) ---
    pooled = slim.groupby(["variant"]).agg(["mean", "std"])
    pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
    pooled = pooled.reset_index()

    # TSR_A computed on pooled means (time saving vs baselineA)
    if "T_total_mean" in pooled.columns:
        tA = float(pooled.loc[pooled["variant"] == "baselineA", "T_total_mean"].iloc[0])
        for v in ("baselineB", "proposed_shared"):
            if (pooled["variant"] == v).any():
                tv = float(pooled.loc[pooled["variant"] == v, "T_total_mean"].iloc[0])
                pooled.loc[pooled["variant"] == v, "TSR_A_mean"] = (tA - tv) / tA * 100.0
        pooled.loc[pooled["variant"] == "baselineA", "TSR_A_mean"] = 0.0

    out_dir = out_root / "_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "table2_i1_i5_by_case.csv"
    by_case.to_csv(p1, index=False, encoding="utf-8-sig")
    logging.info(f"wrote {p1}")

    p2 = out_dir / "table2_i1_i5_mean_std.csv"
    pooled.to_csv(p2, index=False, encoding="utf-8-sig")
    logging.info(f"wrote {p2}")


if __name__ == "__main__":
    main()
