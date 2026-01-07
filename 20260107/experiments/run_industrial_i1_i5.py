"""Industrial pipeline: split a large batch CSV into I1-I5 by order and run StepF.

This script is designed for the CIE paper experiments with industrial data.

What it does
------------
1) Split a single batch file by `Customernumber` (order id) into K sub-cases (default K=5).
   - Each order is kept intact.
   - We balance *total order area* across sub-cases using a greedy bin-packing heuristic.
2) Expand `Quantity` (if present) into a per-part list so the core INRP code can read it.
3) Run `inrp.runner.run_stepF(cfg)` for each case, producing:
   - totals_baselineA.csv / totals_baselineB.csv / totals_proposed_shared.csv
   - summary_stats.csv
   - seed0 plots (layout + toolpath overlay)

Usage (from repo root)
----------------------
  python -m experiments.run_industrial_i1_i5 --src "path/to/BATCH.csv" --out_root outputs

Recommended workflow
--------------------
  1) First do a short dry-run to validate everything:
       python -m experiments.run_industrial_i1_i5 --src BATCH.csv --n_seeds 5 --plot
  2) Then run the full experiment:
       python -m experiments.run_industrial_i1_i5 --src BATCH.csv --n_seeds 100 --plot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running without "pip install -e ." by ensuring ./src is on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from inrp.cfg import CFG
from inrp.runner import run_stepF


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    """Read CSV with robust encoding fallback (utf-8-sig / gbk / gb18030)."""
    p = Path(path)
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "gb18030", "latin1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    # last resort (let pandas raise)
    return pd.read_csv(p)


def split_by_order_balance_area(df: pd.DataFrame, k: int = 5) -> Tuple[List[List[str]], pd.DataFrame]:
    """Split orders into k bins balancing total area.

    Returns:
      - bins: list of order-id lists
      - order_stat: dataframe with n_parts and area_sum per order
    """
    if "Customernumber" not in df.columns:
        raise ValueError("Missing column 'Customernumber' for order-level splitting.")
    if "Fleng" not in df.columns or "Fwidth" not in df.columns:
        raise ValueError("Missing columns 'Fleng'/'Fwidth'.")

    dfa = df.copy()
    dfa["_area"] = dfa["Fleng"].astype(float) * dfa["Fwidth"].astype(float)

    order_stat = dfa.groupby("Customernumber").agg(
        n_parts=("Customernumber", "size"),
        area_sum=("_area", "sum"),
    ).reset_index()

    # greedy bin packing: largest order first
    order_stat = order_stat.sort_values("area_sum", ascending=False)
    bins = [{"area": 0.0, "orders": []} for _ in range(k)]
    for _, row in order_stat.iterrows():
        j = min(range(k), key=lambda t: bins[t]["area"])
        bins[j]["area"] += float(row["area_sum"])
        bins[j]["orders"].append(str(row["Customernumber"]))

    return [b["orders"] for b in bins], order_stat


def explode_quantity_to_parts_csv(df: pd.DataFrame, dst_csv: Path) -> None:
    """Expand Quantity (if present) into per-part rows and write a minimal parts CSV.

    Output columns are compatible with `inrp.dataio.read_sample_parts`:
      uid, Upi, Fleng, Fwidth
    """
    if "Fleng" not in df.columns or "Fwidth" not in df.columns:
        raise ValueError("DataFrame must contain 'Fleng' and 'Fwidth'.")

    if "Quantity" in df.columns:
        q = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1).astype(int)
        q = q.clip(lower=1)
        df2 = df.loc[df.index.repeat(q)].copy()
    else:
        df2 = df.copy()

    uid = range(1, len(df2) + 1)
    if "Upi" in df2.columns:
        pid = df2["Upi"].astype(str)
    elif "ID" in df2.columns:
        pid = df2["ID"].astype(str)
    else:
        pid = [str(i) for i in uid]

    out = pd.DataFrame({
        "uid": uid,
        "Upi": pid,
        "Fleng": pd.to_numeric(df2["Fleng"], errors="coerce"),
        "Fwidth": pd.to_numeric(df2["Fwidth"], errors="coerce"),
    }).dropna(subset=["Fleng", "Fwidth"])

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst_csv, index=False, encoding="utf-8-sig")


def build_cfg(case_name: str, parts_csv: Path, out_root: Path, args) -> CFG:
    cfg = CFG()
    cfg.CASE_NAME = case_name
    cfg.SAMPLE_CSV = str(parts_csv.resolve())
    cfg.OUT_ROOT = str(out_root.resolve())

    # paper parameters
    cfg.BOARD_W = 2440.0
    cfg.BOARD_H = 1220.0
    cfg.TRIM = 5.0
    cfg.GAP = 0.0
    cfg.TOOL_D = 6.0

    cfg.BINARY_SPACING = True
    cfg.SAFE_GAP = cfg.TOOL_D
    cfg.TOUCH_TOL = 1e-6

    cfg.FEED_CUT = 12000.0
    cfg.FEED_AIR = 30000.0
    cfg.T_LIFT = 0.8

    cfg.KERF_MODE = "tool_center"
    cfg.LEAD_IN = 10.0
    cfg.LEAD_OUT = 10.0

    # seeds & runtime
    cfg.SEED0 = args.seed0
    cfg.N_JOBS = args.n_jobs
    cfg.PLOT = bool(args.plot)

    # Adaptive stop (CI convergence) — if enabled, we treat N_SEEDS as MAX_SEEDS
    if getattr(args, "adaptive", False):
        cfg.ADAPTIVE_STOP = True
        cfg.ADAPTIVE_MIN_SEEDS = int(args.min_seeds)
        cfg.ADAPTIVE_STEP_SEEDS = int(args.step_seeds)
        cfg.ADAPTIVE_MAX_SEEDS = int(args.max_seeds)
        cfg.ADAPTIVE_CI_ALPHA = float(args.ci_alpha)
        cfg.ADAPTIVE_CI_REL = float(args.ci_rel)
        cfg.ADAPTIVE_CI_METRIC = str(args.ci_metric)
        cfg.ADAPTIVE_CI_VARIANT = str(args.ci_variant)
        cfg.N_SEEDS = int(args.max_seeds)
    else:
        cfg.N_SEEDS = args.n_seeds

    # ---------------------------------------------------------
    # Algorithm parameters
    # - preset="paper" matches Section 4 defaults
    # - preset="fast" is a practical budget preset (same logic, smaller budgets)
    # ---------------------------------------------------------
    preset = str(getattr(args, "preset", "paper")).lower().strip()

    if preset == "fast":
        cfg.STAGE1_METHOD = "sa"
        cfg.STAGE1_SA_ITERS = 400
        cfg.RESTARTS = 15

        cfg.INNER_REPACK_METHOD = "sa"
        cfg.INNER_REPACK_SA_ITERS = 60
        cfg.INNER_REPACK_SA_T0 = 5000.0
        cfg.INNER_REPACK_SA_ALPHA = 0.995
        cfg.INNER_REPACK_SA_MAX_STARTS = 1
        cfg.INNER_REPACK_SA_PATIENCE = 40
        cfg.INNER_REPACK_SA_AUTO = True
        cfg.INNER_REPACK_SA_BASE_N = 200
        cfg.INNER_REPACK_SA_MIN_ITERS = 8
        cfg.REPACK_BORDER_W = 0.05
        cfg.REPACK_TRAIL_PENALTY = 0.02
    else:
        # Section 4 defaults
        cfg.STAGE1_METHOD = "sa"
        cfg.STAGE1_SA_ITERS = 2000
        cfg.RESTARTS = 50

        cfg.INNER_REPACK_METHOD = "sa"
        cfg.INNER_REPACK_SA_ITERS = 600
        cfg.INNER_REPACK_SA_T0 = 5000.0
        cfg.INNER_REPACK_SA_ALPHA = 0.995
        cfg.INNER_REPACK_SA_MAX_STARTS = 5
        cfg.INNER_REPACK_SA_PATIENCE = 200
        cfg.INNER_REPACK_SA_AUTO = False
        cfg.INNER_REPACK_SA_BASE_N = 200
        cfg.INNER_REPACK_SA_MIN_ITERS = 10
        cfg.REPACK_BORDER_W = 0.05
        cfg.REPACK_TRAIL_PENALTY = 0.02

    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Industrial batch CSV (contains Customernumber, Fleng, Fwidth, Quantity(optional)).")
    ap.add_argument("--k", type=int, default=5, help="Number of sub-cases (default 5 -> I1..I5).")
    ap.add_argument("--out_root", type=str, default="outputs", help="Output root for experiment results.")
    ap.add_argument("--work_dir", type=str, default="data/industrial_cases", help="Where to write generated I1..Ik CSVs.")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--plot", action="store_true", help="Enable seed0 plots")

    ap.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["fast", "paper"],
        help="Budget preset: fast (recommended for many seeds) or paper (Section 4 defaults).",
    )

    # Adaptive stop (CI convergence)
    ap.add_argument("--adaptive", action="store_true", help="Enable adaptive stopping based on CI convergence.")
    ap.add_argument("--min_seeds", type=int, default=10, help="Minimum seeds before checking CI (default: 10)")
    ap.add_argument("--step_seeds", type=int, default=5, help="Add this many seeds per round (default: 5)")
    ap.add_argument("--max_seeds", type=int, default=100, help="Maximum seeds cap (default: 100)")
    ap.add_argument("--ci_alpha", type=float, default=0.05, help="CI alpha (default: 0.05 => 95%% CI)")
    ap.add_argument("--ci_rel", type=float, default=0.02, help="Target relative CI half-width (default: 0.02 => 2%%)")
    ap.add_argument("--ci_metric", type=str, default="T_est_sum", help="Metric field name used for CI (default: T_est_sum)")
    ap.add_argument("--ci_variant", type=str, default="proposed_shared", help="Variant name used for CI (default: proposed_shared)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repo = Path(__file__).resolve().parents[1]
    src = Path(args.src)
    if not src.is_absolute():
        src = (repo / src).resolve()

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (repo / out_root).resolve()

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = (repo / work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    df = _read_csv_with_fallback(str(src))
    logging.info(f"[LOAD] {src.name} rows={len(df)} cols={len(df.columns)}")

    # split
    bins, order_stat = split_by_order_balance_area(df, k=args.k)
    manifest: Dict[str, Dict[str, object]] = {}

    for i, orders in enumerate(bins, start=1):
        case = f"I{i}"
        sub = df[df["Customernumber"].astype(str).isin(set(orders))].copy()
        raw_csv = work_dir / f"{case}_raw.csv"
        sub.to_csv(raw_csv, index=False, encoding="utf-8-sig")

        parts_csv = work_dir / f"{case}_parts.csv"
        explode_quantity_to_parts_csv(sub, parts_csv)

        manifest[case] = {
            "orders": orders,
            "n_orders": len(orders),
            "n_rows_raw": int(len(sub)),
            "n_parts_expanded": int(pd.read_csv(parts_csv).shape[0]),
            "raw_csv": str(raw_csv),
            "parts_csv": str(parts_csv),
        }

        logging.info(f"[CASE] {case}: orders={len(orders)} raw_rows={len(sub)} parts={manifest[case]['n_parts_expanded']}")

    # save manifest
    manifest_path = work_dir / "I1_I5_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info(f"[MANIFEST] wrote {manifest_path}")

    # run stepF per case
    for case in sorted(manifest.keys()):
        cfg = build_cfg(case, Path(manifest[case]["parts_csv"]), out_root, args)
        run_stepF(cfg)


if __name__ == "__main__":
    main()
