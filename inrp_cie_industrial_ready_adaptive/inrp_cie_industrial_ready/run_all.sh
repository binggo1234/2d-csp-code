#!/usr/bin/env bash
set -euo pipefail

# Default run parameters (override via env vars)
: "${N_SEEDS:=20}"
: "${N_JOBS:=1}"
: "${SEED0:=1000}"
: "${TRIM:=5}"
: "${TOOL_D:=6}"
: "${CASE:=demo_case_$$}"


# ============================================================================
# One-click reproducible runner for Linux/macOS.
# - Creates a local venv in .venv (if missing)
# - Installs dependencies via "python -m pip"
# - Runs the demo pipeline with TRIM=5mm and TOOL_D=6mm by default
#
# Usage:
#   bash run_all.sh
#
# Optional:
#   PARTS_CSV=data/my_parts.csv bash run_all.sh
# ============================================================================
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -x ".venv/bin/python" ]; then
  echo "[SETUP] Creating venv in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi

PY=".venv/bin/python"

echo "[SETUP] Python: $PY"
$PY -m pip install -U pip
$PY -m pip install -r requirements.txt
$PY -m pip install -e .

if [ -n "${PARTS_CSV:-}" ]; then
  echo "[RUN] Demo with PARTS_CSV=${PARTS_CSV}"
  $PY -m experiments.run_demo --case "${CASE}" --n_seeds ${N_SEEDS} --n_jobs ${N_JOBS} --seed0 ${SEED0} --trim ${TRIM} --tool_d ${TOOL_D} --parts_csv "${PARTS_CSV}"
else
  echo "[RUN] Demo with default sample CSV (data/sample_parts.csv)"
  $PY -m experiments.run_demo --case "${CASE}" --n_seeds ${N_SEEDS} --n_jobs ${N_JOBS} --seed0 ${SEED0} --trim ${TRIM} --tool_d ${TOOL_D}
fi

echo "[DONE] Outputs are under: outputs/demo_case/"