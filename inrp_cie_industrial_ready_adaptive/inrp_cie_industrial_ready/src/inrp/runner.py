# src/inrp/runner.py
from __future__ import annotations

import csv
import os
import random
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

from statistics import NormalDist

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
    trim = float(getattr(cfg, "TRIM", 0.0) or 0.0)
    # Binary spacing: allow touch/share OR keep distance >= SAFE_GAP (default TOOL_D)
    if bool(getattr(cfg, "BINARY_SPACING", False)):
        safe_gap = float(getattr(cfg, "SAFE_GAP", getattr(cfg, "TOOL_D", 0.0)) or 0.0)
    else:
        safe_gap = 0.0
    touch_tol = float(getattr(cfg, "TOUCH_TOL", 1e-6) or 0.0)

    if variant == "baselineA":
        boards = pack_baselineA(parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT, trim=trim, safe_gap=safe_gap, touch_tol=touch_tol)
        share_mode = "none"
    elif variant == "baselineB":
        boards = pack_baselineB(
            parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT,
            trim=trim, safe_gap=safe_gap, touch_tol=touch_tol,
            restarts=getattr(cfg, "RESTARTS", 50),
            seed=seed,
            rand_place=getattr(cfg, "BASELINEB_RAND_PLACE", True),
            rand_topk=getattr(cfg, "BASELINEB_RAND_TOPK", 3),
        )
        share_mode = "none"
    elif variant == "proposed_shared":
        boards = pack_proposed_shared(
            parts_shuf, cfg.BOARD_W, cfg.BOARD_H, cfg.ALLOW_ROT,
            trim=trim, safe_gap=safe_gap, touch_tol=touch_tol,
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
            stage1_method=getattr(cfg, "STAGE1_METHOD", "sa"),
            stage1_sa_iters=getattr(cfg, "STAGE1_SA_ITERS", 2000),
        )
        share_mode = "union"
    else:
        raise ValueError(f"unknown variant: {variant}")

    # ---- VALIDATE ----
    validate_solution(parts_shuf, boards, cfg.BOARD_W, cfg.BOARD_H, trim=trim, safe_gap=safe_gap, touch_tol=touch_tol)

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


def _normal_ci_rel_halfwidth(values: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
    """Compute a normal-approx confidence interval half-width.

    Returns
    -------
    mean : float
        Sample mean.
    half_width : float
        CI half-width.
    rel_half_width : float
        half_width / |mean| (0 if mean is 0).

    Notes
    -----
    We intentionally use a normal approximation (z) to avoid introducing extra
    dependencies (e.g., SciPy). With n\u226510, this is typically acceptable for
    the adaptive stopping heuristic.
    """
    n = len(values)
    if n == 0:
        return 0.0, float("inf"), float("inf")
    if n == 1:
        m = float(values[0])
        return m, float("inf"), float("inf")

    m = sum(values) / n
    # sample std
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    s = math.sqrt(max(var, 0.0))
    if s == 0.0:
        return m, 0.0, 0.0

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    half = z * s / math.sqrt(n)
    rel = half / abs(m) if abs(m) > 0 else float("inf")
    return m, half, rel


def _run_one_seed_worker(cfg_dict: Dict[str, Any], seed: int, variants: List[str], out_case_dir: str) -> List[Dict[str, Any]]:
    """Worker function for parallel runs (pickle-friendly)."""
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

    # Dump configuration snapshot
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

    adaptive = bool(getattr(cfg, "ADAPTIVE_STOP", False))
    n_jobs = int(getattr(cfg, "N_JOBS", 1) or 1)

    if adaptive:
        # --------------------------------------------------------------
        # Plan-B: Adaptive stopping based on CI convergence.
        # We run in batches and stop once the 95% CI relative half-width
        # of a key metric (e.g., T_est_sum) is below the target.
        # --------------------------------------------------------------
        if n_jobs > 1:
            logger.warning("[ADAPT] Adaptive mode currently runs sequentially; forcing N_JOBS=1.")
            n_jobs = 1

        min_seeds = int(getattr(cfg, "ADAPTIVE_MIN_SEEDS", 10))
        step_seeds = int(getattr(cfg, "ADAPTIVE_STEP", 5))
        max_seeds = int(getattr(cfg, "ADAPTIVE_MAX_SEEDS", int(cfg.N_SEEDS)))
        ci_alpha = float(getattr(cfg, "ADAPTIVE_CI_ALPHA", 0.05))
        ci_rel = float(getattr(cfg, "ADAPTIVE_CI_REL", 0.02))
        ci_variant = str(getattr(cfg, "ADAPTIVE_CI_VARIANT", "proposed_shared"))
        ci_metric = str(getattr(cfg, "ADAPTIVE_CI_METRIC", "T_est_sum"))

        logger.info(
            f"[ADAPT] min_seeds={min_seeds} step={step_seeds} max_seeds={max_seeds} "
            f"ci_alpha={ci_alpha} ci_rel={ci_rel} metric={ci_variant}.{ci_metric}"
        )

        parts = read_sample_parts(cfg.SAMPLE_CSV, cfg.TRIM, cfg.GAP, tool_d=getattr(cfg, "TOOL_D", None))
        used_seeds: List[int] = []
        last_ci = {"n": 0, "mean": None, "half": None, "rel": None}

        n_done = 0
        while n_done < max_seeds:
            if n_done < min_seeds:
                batch = min_seeds - n_done
            else:
                batch = step_seeds
            batch = min(batch, max_seeds - n_done)
            if batch <= 0:
                break

            for j in range(batch):
                seed = cfg.SEED0 + n_done + j
                used_seeds.append(seed)
                for v in variants:
                    logger.info(f"=== [{v}] seed={seed} ===")
                    row = run_one_variant(parts, cfg, seed, v, out_case_dir)
                    totals_all.append(row)
                    _log_row(row)

            n_done = len(used_seeds)

            # CI check (based on the chosen variant/metric)
            vals = [
                float(r[ci_metric])
                for r in totals_all
                if r.get("variant") == ci_variant and ci_metric in r
            ]
            if len(vals) >= min_seeds:
                m, half, rel = _normal_ci_rel_halfwidth(vals, alpha=ci_alpha)
                last_ci = {"n": len(vals), "mean": m, "half": half, "rel": rel}
                logger.info(
                    f"[ADAPT] n={len(vals)} {ci_variant}.{ci_metric}: mean={m:.3f}, "
                    f"CI_half={half:.3f}, rel_half={rel*100:.2f}% (target<{ci_rel*100:.2f}%)"
                )
                if rel <= ci_rel:
                    logger.info(f"[ADAPT] Stop criterion met at n_seeds={n_done}.")
                    break

        # Record the actual number of seeds used
        try:
            cfg.N_SEEDS = int(n_done)
        except Exception:
            pass

        # Save adaptive report for paper reproducibility
        _write_rows_csv(
            out_case_dir / "adaptive_report.csv",
            [
                {
                    "case": cfg.CASE_NAME,
                    "metric": f"{ci_variant}.{ci_metric}",
                    "ci_alpha": ci_alpha,
                    "target_rel_half": ci_rel,
                    "min_seeds": min_seeds,
                    "step_seeds": step_seeds,
                    "max_seeds": max_seeds,
                    "used_seeds": n_done,
                    "ci_n": last_ci.get("n"),
                    "ci_mean": last_ci.get("mean"),
                    "ci_half": last_ci.get("half"),
                    "ci_rel_half": last_ci.get("rel"),
                }
            ],
        )

    elif n_jobs > 1 and cfg.N_SEEDS > 1:
        logger.info(f"[PAR] Running seeds in parallel: N_JOBS={n_jobs}")
        cfg_dict = dict(getattr(cfg, "__dict__", {}))
        if not cfg_dict:
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

    # summary (With Error Handling!)
    try:
        from .stats_test import build_summary_stats
        summary_rows = build_summary_stats(totals_all, variants)
        _write_rows_csv(out_case_dir / "summary_stats.csv", summary_rows)
    except ImportError:
        logger.warning("[WARN] Could not import 'build_summary_stats'. Skipping summary_stats.csv generation.")
    except Exception as e:
        logger.warning(f"[WARN] Failed to generate summary stats: {e}")

    logger.info("StepF finished.")
    logger.info(f"Output dir: {str(out_case_dir)}")
    logger.info("Generated: totals_*.csv, summary_stats.csv, boards_metrics.csv, and plots (if enabled).")