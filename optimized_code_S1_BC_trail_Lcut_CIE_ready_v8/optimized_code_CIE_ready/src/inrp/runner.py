# runner.py
from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import logging
logger = logging.getLogger(__name__)

from .dataio import read_sample_parts
from .packer import pack_baselineA, pack_baselineB, pack_proposed_shared
from .validate import validate_solution
from .metrics import compute_board_metrics, aggregate_totals
from .viz import plot_one_seed_outputs
from .repro import set_global_seed, dump_config


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Robust CSV writer:
    - fieldnames is union of all keys across rows (avoid "dict contains fields not in fieldnames")
    - write tmp then os.replace to avoid Windows PermissionError
    """
    _ensure_dir(path.parent)
    if not rows:
        return

    preferred = [
        "case", "variant", "seed", "n",
        "N_input", "N_board",
        "runtime_s",
        # common totals fields (keep if exist)
        "U_avg", "U_mean",
        "L_shared_sum", "N_lift_sum", "L_air_sum",
        "L_cut_sum", "L_cut_base_sum", "L_kerf_extra_sum", "L_lead_sum",
        "T_est_sum",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    fieldnames = []
    for k in preferred:
        if k in all_keys:
            fieldnames.append(k)
            all_keys.remove(k)
    fieldnames += sorted(all_keys)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows: file may be locked by Excel/WPS/preview.
        # Fall back to a new filename to avoid crashing long experiments.
        import time
        ts = int(time.time())
        alt = path.with_name(f"{path.stem}_locked_{ts}{path.suffix}")
        os.replace(tmp, alt)
        logger.warning(f"[WARN] {path.name} is locked (likely opened by Excel). Wrote: {alt.name}. Close it and rerun to overwrite.")


def run_one_variant(parts, cfg, seed: int, variant: str, out_case_dir: Path) -> Dict[str, Any]:
    """
    Run one (seed, variant):
      out_case_dir/variant/seed_xxx/boards_metrics.csv + plots (only seed==SEED0)
    Return totals row for this run.
    """
    # Reproducibility: set global RNG seeds
    set_global_seed(seed)

    rng = random.Random(seed)
    parts_shuf = parts[:]
    rng.shuffle(parts_shuf)

    out_variant_dir = out_case_dir / variant
    out_seed_dir = out_variant_dir / f"seed_{seed}"
    _ensure_dir(out_seed_dir)

    t0 = time.perf_counter()

    # ---- PACK ----
    if variant == "baselineA":
        boards = pack_baselineA(parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT)
        share_mode = "none"
    elif variant == "baselineB":
        boards = pack_baselineB(
            parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT,
            restarts=getattr(cfg, "RESTARTS", 50),
            seed=seed,
            rand_place=getattr(cfg, "BASELINEB_RAND_PLACE", True),
            rand_topk=getattr(cfg, "BASELINEB_RAND_TOPK", 3),
        )
        share_mode = "none"
    elif variant == "proposed_shared":
        boards = pack_proposed_shared(
            parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT,
            restarts=getattr(cfg, "RESTARTS", 50),
            seed=seed,
            inner_restarts=getattr(cfg, "INNER_REPACK_RESTARTS", 30),
            border_w=getattr(cfg, "REPACK_BORDER_W", 0.05),
            trail_penalty=getattr(cfg, "REPACK_TRAIL_PENALTY", 0.02),
            cut_mode=getattr(cfg, "CUT_MODE", "trail"),
            tab_enable=getattr(cfg, "TAB_ENABLE", False),
            tab_per_part=getattr(cfg, "TAB_PER_PART", 0),
            tab_len=getattr(cfg, "TAB_LEN", 0.0),
            tab_corner_clear=getattr(cfg, "TAB_CORNER_CLEAR", 0.0),
            line_snap_eps=getattr(cfg, "LINE_SNAP_EPS", 0.0),
            min_shared_len=getattr(cfg, "MIN_SHARED_LEN", 0.0),
            nd_coord=getattr(cfg, "ND_COORD", 6),
            stage2_method=getattr(cfg, "INNER_REPACK_METHOD", "restarts"),
            stage2_sa_iters=getattr(cfg, "INNER_REPACK_SA_ITERS", 600),
            stage2_sa_T0=getattr(cfg, "INNER_REPACK_SA_T0", 5000.0),
            stage2_sa_alpha=getattr(cfg, "INNER_REPACK_SA_ALPHA", 0.995),
            stage2_sa_max_starts=getattr(cfg, "INNER_REPACK_SA_MAX_STARTS", 5),
        )
        share_mode = "union"
    else:
        raise ValueError(f"unknown variant: {variant}")

    # ---- VALIDATE ----
    validate_solution(parts_shuf, boards, cfg.BOARD_W, cfg.BOARD_H)

    # ---- METRICS ----
    boards_rows: List[Dict[str, Any]] = []
    for b in boards:
        row = compute_board_metrics(
            b,
            board_W=cfg.BOARD_W,
            board_H=cfg.BOARD_H,
            feed_cut=cfg.FEED_CUT,
            feed_air=cfg.FEED_AIR,
            t_lift=cfg.T_LIFT,
            share_mode=share_mode,
            cut_mode=getattr(cfg, "CUT_MODE", "cpp"),
            tool_d=getattr(cfg, "TOOL_D", 0.0),
            kerf_mode=getattr(cfg, "KERF_MODE", "none"),
            lead_in=getattr(cfg, "LEAD_IN", 0.0),
            lead_out=getattr(cfg, "LEAD_OUT", 0.0),
            tab_enable=getattr(cfg, "TAB_ENABLE", False),
            tab_per_part=getattr(cfg, "TAB_PER_PART", 0),
            tab_len=getattr(cfg, "TAB_LEN", 0.0),
            tab_corner_clear=getattr(cfg, "TAB_CORNER_CLEAR", 0.0),
            line_snap_eps=getattr(cfg, "LINE_SNAP_EPS", 0.0),
            min_shared_len=getattr(cfg, "MIN_SHARED_LEN", 0.0),
            nd_coord=getattr(cfg, "ND_COORD", 6),
        )
        boards_rows.append(row)

    _write_rows_csv(out_seed_dir / "boards_metrics.csv", boards_rows)

    # ---- PLOTS (only SEED0) ----
    if seed == cfg.SEED0 and getattr(cfg, "PLOT", True):
        plot_one_seed_outputs(
            boards,
            boards_rows,
            out_seed_dir,
            title=f"{cfg.CASE_NAME} | {variant} | seed={seed}",
            max_boards=getattr(cfg, "PLOT_MAX_BOARDS", 6),
            trim=getattr(cfg, "TRIM", 0.0),
            share_mode=share_mode,
            plot_toolpath=getattr(cfg, "PLOT_TOOLPATH", True),
            show_ids=getattr(cfg, "PLOT_SHOW_IDS", False),
            tab_enable=getattr(cfg, "TAB_ENABLE", False),
            tab_per_part=getattr(cfg, "TAB_PER_PART", 0),
            tab_len=getattr(cfg, "TAB_LEN", 0.0),
            tab_corner_clear=getattr(cfg, "TAB_CORNER_CLEAR", 0.0),
            line_snap_eps=getattr(cfg, "LINE_SNAP_EPS", 0.0),
            min_shared_len=getattr(cfg, "MIN_SHARED_LEN", 0.0),
            nd_coord=getattr(cfg, "ND_COORD", 6),
        )

    runtime_s = time.perf_counter() - t0

    # ---- TOTALS ----
    totals_row = aggregate_totals(boards_rows)
    totals_row.update({
        "case": cfg.CASE_NAME,
        "variant": variant,
        "seed": seed,
        "N_input": len(parts_shuf),
        "N_board": len(boards),
        "runtime_s": runtime_s,
    })
    return totals_row


def _run_one_seed_worker(cfg_dict: Dict[str, Any], seed: int, variants: List[str], out_case_dir: str) -> List[Dict[str, Any]]:
    """Worker function for parallel runs (pickle-friendly).

    Notes:
    - We reconstruct a lightweight cfg object from a plain dict to avoid pickling issues.
    - Each worker re-reads the parts CSV (cheap vs. pickling large objects).
    - File outputs are isolated under: outputs/<case>/<variant>/seed_<seed>/...
    """
    cfg = SimpleNamespace(**cfg_dict)
    parts = read_sample_parts(cfg.SAMPLE_CSV, cfg.TRIM, cfg.GAP, tool_d=getattr(cfg, "TOOL_D", None))
    rows: List[Dict[str, Any]] = []
    out_case_path = Path(out_case_dir)
    for v in variants:
        rows.append(run_one_variant(parts, cfg, seed, v, out_case_path))
    return rows


def run_stepF(cfg) -> None:
    """
    StepF: fixed sample_parts.csv, run N_SEEDS * variants.
    Output:
      totals_baselineA.csv / totals_baselineB.csv / totals_proposed_shared.csv
      summary_stats.csv
    """
    out_root = Path(cfg.OUT_ROOT)
    out_case_dir = out_root / cfg.CASE_NAME
    _ensure_dir(out_case_dir)

    # Dump configuration snapshot for reproducibility (paper-friendly)
    if getattr(cfg, "DUMP_CONFIG", True):
        dump_config(cfg, out_case_dir, extra={"script": "main_stepF.py"})

    logger.info(f"[CASE] {cfg.CASE_NAME}")
    logger.info(
        f"[CFG] board={cfg.BOARD_W}x{cfg.BOARD_H} trim={cfg.TRIM} gap={cfg.GAP} "
        f"tool_d={getattr(cfg,'TOOL_D',None)} kerf_mode={getattr(cfg,'KERF_MODE','none')} "
        f"lead_in={getattr(cfg,'LEAD_IN',0.0)} lead_out={getattr(cfg,'LEAD_OUT',0.0)} "
        f"seeds={cfg.N_SEEDS} seed0={cfg.SEED0} cut_mode={getattr(cfg,'CUT_MODE','cpp')}"
    )
    logger.info(f"[OUT_ROOT] {str(out_root)}")

    variants = ["baselineA", "baselineB", "proposed_shared"]
    totals_all: List[Dict[str, Any]] = []

    def _log_row(row: Dict[str, Any]) -> None:
        seed = row.get("seed")
        v = row.get("variant")
        Uv = row.get("U_avg", row.get("U_mean", None))
        msg = f"[OK] {v} seed={seed} N_board={row.get('N_board')}"
        if isinstance(Uv, (int, float)):
            msg += f" U={Uv:.4f}"
        if "L_shared_sum" in row:
            msg += f" L_shared={row['L_shared_sum']:.1f}"
        if "N_lift_sum" in row:
            msg += f" N_lift={int(row['N_lift_sum'])}"
        if "T_est_sum" in row:
            msg += f" T_est={row['T_est_sum']:.2f}"
        msg += f" runtime={row.get('runtime_s', 0.0):.3f}s"
        logger.info(msg)

    n_jobs = int(getattr(cfg, "N_JOBS", 1) or 1)
    if n_jobs > 1 and cfg.N_SEEDS > 1:
        # Parallel across seeds (each worker runs all variants for one seed)
        logger.info(f"[PAR] Running seeds in parallel: N_JOBS={n_jobs}")
        cfg_dict = dict(getattr(cfg, "__dict__", {}))
        if not cfg_dict:
            # Fallback for non-dataclass configs
            cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {}
            for i in range(cfg.N_SEEDS):
                seed = cfg.SEED0 + i
                fut = ex.submit(_run_one_seed_worker, cfg_dict, seed, variants, str(out_case_dir))
                futures[fut] = seed
            for fut in as_completed(futures):
                rows = fut.result()
                for row in rows:
                    totals_all.append(row)
                    _log_row(row)
    else:
        # Serial
        # IMPORTANT: read_sample_parts should handle encoding fallback (utf-8-sig/gbk/gb18030)
        parts = read_sample_parts(cfg.SAMPLE_CSV, cfg.TRIM, cfg.GAP, tool_d=getattr(cfg, "TOOL_D", None))
        for i in range(cfg.N_SEEDS):
            seed = cfg.SEED0 + i
            for v in variants:
                logger.info(f"=== [{v}] seed={seed} ===")
                row = run_one_variant(parts, cfg, seed, v, out_case_dir)
                totals_all.append(row)
                _log_row(row)


    # totals per variant
    for v in variants:
        rows_v = [r for r in totals_all if r["variant"] == v]
        _write_rows_csv(out_case_dir / f"totals_{v}.csv", rows_v)

    # summary
    from .stats_test import build_summary_stats
    summary_rows = build_summary_stats(totals_all, variants)
    _write_rows_csv(out_case_dir / "summary_stats.csv", summary_rows)

    logger.info("StepF finished.")
    logger.info(f"Output dir: {str(out_case_dir)}")
    logger.info("Generated: totals_*.csv, summary_stats.csv, boards_metrics.csv, and plots (if enabled).")
