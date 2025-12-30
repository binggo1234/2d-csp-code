@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  One-click reproducible runner for Windows (cmd / PowerShell).
REM  - Creates a local venv in .venv (if missing)
REM  - Installs dependencies via "python -m pip" (no PATH pip.exe required)
REM  - Runs the demo pipeline with TRIM=5mm and TOOL_D=6mm by default
REM
REM  Usage:
REM    1) Open terminal at repo root
REM    2) run_all.bat
REM
REM  Optional:
REM    set PARTS_CSV=data\my_parts.csv
REM    run_all.bat
REM ============================================================================
cd /d "%~dp0"

REM ---- Default run parameters (override by setting env vars before running) ----
if not defined N_SEEDS set "N_SEEDS=20"
if not defined N_JOBS set "N_JOBS=1"
if not defined SEED0 set "SEED0=1000"
if not defined TRIM set "TRIM=5"
if not defined TOOL_D set "TOOL_D=6"
if not defined CASE set "CASE=demo_case_%RANDOM%"
REM Optional: set PARTS_CSV=path\to\your_parts.csv
REM ---------------------------------------------------------------------------


REM --------------------------------------------------------------------------
REM  Resolve a Python executable for bootstrapping venv creation.
REM  Priority:
REM    1) User provided PYTHON_EXE (absolute path)
REM    2) Windows Python Launcher: py -3.12 (fallback: py -3)
REM    3) python on PATH
REM
REM  If your IDE runs this .bat without inheriting PATH, set PYTHON_EXE first:
REM    set PYTHON_EXE=C:\Path\to\python.exe
REM --------------------------------------------------------------------------
set "PY_BOOT="

set "PY_BOOT_IS_PATH="

if defined PYTHON_EXE (
  set "PY_BOOT=%PYTHON_EXE%"
  set "PY_BOOT_IS_PATH=1"
)

if not defined PY_BOOT (
  where py >nul 2>nul && (
    set "PY_BOOT=py -3.12"
    %PY_BOOT% -c "import sys;print(sys.version_info[:2])" >nul 2>&1 || set "PY_BOOT=py -3"
  )
)

if not defined PY_BOOT (
  where python >nul 2>nul && set "PY_BOOT=python"
)

if not defined PY_BOOT (
  echo [ERROR] Python not found for creating venv.
  echo        Try opening a normal cmd window and run: where.exe python
  echo        Or set PYTHON_EXE to your python.exe full path and rerun.
  exit /b 1
)

REM Create venv if missing
if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating venv in .venv ...
  echo [SETUP] Boot python: %PY_BOOT%
  if defined PY_BOOT_IS_PATH (

    "%PY_BOOT%" -m venv .venv

  ) else (

    %PY_BOOT% -m venv .venv

  )
  if errorlevel 1 (
    echo [ERROR] Failed to create venv. Make sure python is available on PATH.
    exit /b 1
  )
)

set "PY=%CD%\.venv\Scripts\python.exe"

echo [SETUP] Python: %PY%
"%PY%" -m ensurepip --upgrade >nul 2>&1

echo [SETUP] Upgrading pip ...
"%PY%" -m pip install -U pip

echo [SETUP] Installing requirements ...
"%PY%" -m pip install -r requirements.txt

echo [SETUP] Installing package (editable) ...
"%PY%" -m pip install -e .

REM Run demo
echo [RUNCFG] case=!CASE! n_seeds=!N_SEEDS! n_jobs=!N_JOBS! seed0=!SEED0! trim=!TRIM! tool_d=!TOOL_D!
if defined PARTS_CSV (
  echo [RUN] Demo with PARTS_CSV=!PARTS_CSV!
  "%PY%" -m experiments.run_demo --case "!CASE!" --n_seeds !N_SEEDS! --n_jobs !N_JOBS! --seed0 !SEED0! --trim !TRIM! --tool_d !TOOL_D! --parts_csv "!PARTS_CSV!"
) else (
  echo [RUN] Demo with default sample CSV (data\sample_parts.csv)
  "%PY%" -m experiments.run_demo --case "!CASE!" --n_seeds !N_SEEDS! --n_jobs !N_JOBS! --seed0 !SEED0! --trim !TRIM! --tool_d !TOOL_D!
)

echo.
echo [DONE] Outputs are under: outputs\!CASE!\
endlocal