"""Convenience wrapper to run industrial I1-I5 experiments.

This wrapper makes the project runnable even without `pip install -e .`.

Example:
  python run_industrial_i1_i5.py --src data/ZJ_ZN241102A1-1_ZN_GSTX.csv --k 5 --n_seeds 1 --plot
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import after sys.path patch
from experiments.run_industrial_i1_i5 import main


if __name__ == "__main__":
    main()
