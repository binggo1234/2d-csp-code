"""Generate synthetic 2D rectangular part instances for reproducibility checks.

This script is only for *dummy* (non-confidential) data. It produces CSV files compatible
with `inrp.dataio.read_sample_parts()`.

Usage:
  python instances/generate_dummy_instances.py --out data/sample_parts.csv --n 100 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="output CSV path")
    ap.add_argument("--n", type=int, default=100, help="number of parts")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--maxL", type=int, default=1200, help="max length (mm)")
    ap.add_argument("--maxW", type=int, default=600, help="max width (mm)")
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for uid in range(1, args.n + 1):
        L = random.randint(80, args.maxL)
        W = random.randint(60, args.maxW)
        rows.append({"uid": uid, "pid_raw": f"P{uid:04d}", "Fleng_mm": L, "Fwidth_mm": W})

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "pid_raw", "Fleng_mm", "Fwidth_mm"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] wrote {len(rows)} parts -> {out}")


if __name__ == "__main__":
    main()
