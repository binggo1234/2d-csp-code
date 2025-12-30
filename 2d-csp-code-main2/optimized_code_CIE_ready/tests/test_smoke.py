from __future__ import annotations

from pathlib import Path

from inrp.cfg import CFG
from inrp.runner import run_one_variant
from inrp.dataio import read_sample_parts


def test_run_one_variant_smoke(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    cfg = CFG()
    cfg.SAMPLE_CSV = str((repo / cfg.SAMPLE_CSV).resolve())
    cfg.OUT_ROOT = str(tmp_path)
    cfg.CASE_NAME = "pytest_case"
    cfg.N_SEEDS = 1
    cfg.SEED0 = 1000
    cfg.PLOT = False  # keep tests fast

    parts = read_sample_parts(cfg.SAMPLE_CSV, cfg.TRIM, cfg.GAP, tool_d=cfg.TOOL_D)
    out_case_dir = tmp_path / cfg.CASE_NAME
    out_case_dir.mkdir(parents=True, exist_ok=True)

    row = run_one_variant(parts, cfg, seed=cfg.SEED0, variant="baselineA", out_case_dir=out_case_dir)
    assert "N_board" in row
    assert row["N_board"] >= 1
