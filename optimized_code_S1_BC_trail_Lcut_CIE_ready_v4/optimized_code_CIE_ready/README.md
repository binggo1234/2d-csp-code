# Integrated Nesting & Routing (INRP) — Anonymous Code Package

This repository contains the Python implementation used in the paper submission.
It targets **panel-based furniture** cutting on standard boards (default: 2440×1220 mm),
and evaluates multiple variants including a **shared-edge (common-cut) aware** method.

## Quick start (demo)

Create a clean environment and install dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run an end-to-end demo on **synthetic** parts (no confidential data):

```bash
python -m experiments.run_demo --n_seeds 3 --seed0 1000
```

Outputs will be written to `outputs/<case_name>/`.

## Key process parameters

Two manufacturing parameters are explicit in the configuration and are used consistently
in geometry feasibility and metric accounting:

- `TRIM` (mm): trimming margin on each side of the board (default **5.0 mm**)
- `TOOL_D` (mm): tool diameter / kerf proxy (default **6.0 mm**)

You can override them via CLI in the demo script, or by editing `src/inrp/cfg.py`.

## Project layout

- `src/inrp/` — core algorithms (nesting + routing + metrics)
- `data/` — synthetic demo data (`sample_parts.csv`)
- `instances/` — dummy instance generator
- `experiments/` — scripts that reproduce paper-style outputs
- `tests/` — minimal unit/integration tests
- `docs/` — module-to-paper mapping notes

## Data availability

This code package ships with **synthetic demo instances** in `data/sample_parts.csv`
to allow reviewers to run the pipeline without accessing any proprietary production data.
If you have real instances, replace the CSV (same schema; see `data/README.md`).

## Expected outputs

For each variant and each seed, the runner writes:

- `totals_<variant>.csv` — per-seed totals (utilization, shared length, air-move, lift count, time estimate, etc.)
- `boards_metrics.csv` — per-board detailed metrics
- `summary_stats.csv` — mean/std summary across seeds
- optional plots (nesting + toolpath) if matplotlib is available

## Anonymity

This package is prepared for double-blind review:
- no hard-coded personal paths
- no IDE metadata / git history
- synthetic data included for execution checks


## Using your own parts table

By default, the demo reads `data/sample_parts.csv`.

- Easiest: replace that file with your own CSV (keep the header compatible).
- Or: keep your file in `data/` and pass it explicitly:

```bash
python -m experiments.run_demo --parts_csv data/my_parts.csv --trim 5 --tool_d 6
```

On Windows one-click runner, you can also do:

```bat
set PARTS_CSV=data\my_parts.csv
run_all.bat
```

Expected columns (any one of the following header styles is OK):
- `uid, pid_raw, Fleng_mm, Fwidth_mm`
- `Fleng, Fwidth` (optional id column)
