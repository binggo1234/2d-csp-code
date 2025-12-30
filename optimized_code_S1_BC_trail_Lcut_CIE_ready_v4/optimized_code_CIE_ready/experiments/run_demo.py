"""Run a small end-to-end demo (synthetic data) for reviewers.

This script is intentionally lightweight and produces:
- totals_*.csv
- summary_stats.csv
- boards_metrics.csv
- example nesting/path plots (if matplotlib is installed)

Usage (from repo root):
  python -m experiments.run_demo

Optional:
  python -m experiments.run_demo --n_seeds 5 --seed0 1000 --trim 5 --tool_d 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from inrp.cfg import CFG
from inrp.runner import run_stepF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, default="demo_case")
    ap.add_argument("--parts_csv", type=str, default=None, help="Path to parts CSV. If relative, resolved from repo root.")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--trim", type=float, default=5.0)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--tool_d", type=float, default=6.0)
    ap.add_argument("--plot", action="store_true", help="force enable plots")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repo = Path(__file__).resolve().parents[1]

    cfg = CFG()
    cfg.CASE_NAME = args.case
    cfg.SEED0 = args.seed0
    cfg.N_SEEDS = args.n_seeds
    cfg.TRIM = args.trim
    cfg.GAP = args.gap
    cfg.TOOL_D = args.tool_d

    if args.parts_csv:
        cfg.SAMPLE_CSV = args.parts_csv

    # resolve relative paths to absolute so the demo can run from any cwd
    # resolve relative paths to absolute so the demo can run from any cwd
    p = Path(cfg.SAMPLE_CSV)
    if not p.is_absolute():
        p = repo / p
    cfg.SAMPLE_CSV = str(p.resolve())
    cfg.OUT_ROOT = str((repo / cfg.OUT_ROOT).resolve())

    if args.plot:
        cfg.PLOT = True

    run_stepF(cfg)


if __name__ == "__main__":
    main()
