# metrics.py
from __future__ import annotations

from typing import Dict, Any, List
import statistics
import math

from .routing import (
    build_segments_from_board,
    estimate_cut_and_strokes,
    air_length_by_strokes,
    trails_lower_bound_from_segments,
)


def compute_board_metrics(
    board,
    board_W: float,
    board_H: float,
    feed_cut: float,
    feed_air: float,
    t_lift: float,
    share_mode: str,
    cut_mode: str,
    # S1: kerf/lead-in/out
    tool_d: float = 0.0,
    kerf_mode: str = "none",
    lead_in: float = 0.0,
    lead_out: float = 0.0,
    # tabs
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    # shared-edge detection robustness
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
) -> Dict[str, Any]:
    """Compute per-board metrics.

    share_mode:
      - "none"  : baseline (no common-cut union)
      - "union" : union common-cut (shared edges merged; connected components used)

    cut_mode:
      - "trail": base segment sum
      - "cpp"  : base + approximate CPP extra (odd-degree matching)

    Tabs (optional): remove short uncut bridges from segments, which affects
    L_cut (slightly down) and may change connectivity & air length.
    """
    n_parts = len(board.placed)

    # utilization uses original sizes (no inflation)
    used = sum(pp.w0 * pp.h0 for pp in board.placed)
    U = used / (board_W * board_H) if board_W * board_H > 0 else 0.0

    segs, L_shared, n_tabs = build_segments_from_board(
        board,
        share_mode=share_mode,
        tab_enable=tab_enable,
        tab_per_part=tab_per_part,
        tab_len=tab_len,
        tab_corner_clear=tab_corner_clear,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        nd=nd_coord,
    )

    L_cut_base, n_comp, n_stroke, stroke_reps, comps = estimate_cut_and_strokes(segs, cut_mode=cut_mode)

    # Validation-friendly stats: Euler-trail lower bound based on odd-degree nodes
    odd_total, min_trails, _per = trails_lower_bound_from_segments(segs)

    # ---- S1: kerf口径 + lead-in/out（简化实现） ----
    # 口径说明：
    # - L_cut_base：几何边界/共边合并后的线段长度（不含任何引入/引出，也不含刀具偏置增量）
    # - kerf_mode == "tool_center"：用 2*pi*r * N_stroke 近似补偿“中心线偏置/圆角”带来的长度增量
    #   （简化假设：每条 stroke 约等价于一圈圆弧的角点增量；用于论文/对比口径一致性）
    r = max(float(tool_d), 0.0) * 0.5
    L_kerf_extra = 0.0
    if kerf_mode == "tool_center" and r > 0 and n_stroke > 0:
        L_kerf_extra = 2.0 * math.pi * r * float(n_stroke)

    # lead-in/out：每条 stroke 追加固定直线长度（更接近 CNC 真实起刀/收刀）
    L_lead = 0.0
    if (lead_in > 0 or lead_out > 0) and n_stroke > 0:
        L_lead = (max(float(lead_in), 0.0) + max(float(lead_out), 0.0)) * float(n_stroke)

    L_cut = L_cut_base + L_kerf_extra + L_lead

    # P2 (refined): Lifts based on explicit stroke decomposition (edge-disjoint trails)
    #   - N_lift: number of strokes (more faithful than component count)
    #   - L_air : TSP over stroke entry points (NN + 2-opt)
    L_air = air_length_by_strokes(stroke_reps, start=(0.0, 0.0))
    N_lift = n_stroke

    # time estimate: cut(mm/min) + air(mm/min) + lift(s)
    T_cut = (L_cut / feed_cut) * 60.0 if feed_cut > 0 else 0.0
    T_air = (L_air / feed_air) * 60.0 if feed_air > 0 else 0.0
    T_est = T_cut + T_air + N_lift * t_lift

    return dict(
        board=board.bid,
        n_parts=n_parts,
        U=U,
        L_shared=L_shared if share_mode == "union" else 0.0,
        n_comp=n_comp,
        n_stroke=n_stroke,
        n_tabs=n_tabs,
        odd_total=odd_total,
        min_trails_lb=min_trails,
        stroke_over_lb=(float(n_stroke) / float(min_trails) if min_trails > 0 else 0.0),
        N_lift=N_lift,
        L_air=L_air,
        # cut length (mm)
        L_cut=L_cut,
        L_cut_base=L_cut_base,
        L_kerf_extra=L_kerf_extra,
        L_lead=L_lead,
        kerf_mode=kerf_mode,
        tool_d=tool_d,
        lead_in=lead_in,
        lead_out=lead_out,
        T_est=T_est,
    )


def aggregate_totals(boards_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not boards_rows:
        return dict(
            N_board=0,
            U_avg=0.0,
            L_shared_sum=0.0,
            n_comp_sum=0,
            n_stroke_sum=0,
            n_tabs_sum=0,
            odd_total_sum=0,
            min_trails_lb_sum=0,
            N_lift_sum=0,
            L_air_sum=0.0,
            L_cut_sum=0.0,
            L_cut_base_sum=0.0,
            L_kerf_extra_sum=0.0,
            L_lead_sum=0.0,
            T_est_sum=0.0,
        )

    N_board = len(boards_rows)
    U_avg = statistics.mean([r["U"] for r in boards_rows])
    return dict(
        N_board=N_board,
        U_avg=U_avg,
        L_shared_sum=sum(r.get("L_shared", 0.0) for r in boards_rows),
        n_comp_sum=int(sum(r.get("n_comp", 0) for r in boards_rows)),
        n_stroke_sum=int(sum(r.get("n_stroke", 0) for r in boards_rows)),
        n_tabs_sum=int(sum(r.get("n_tabs", 0) for r in boards_rows)),
        odd_total_sum=int(sum(r.get("odd_total", 0) for r in boards_rows)),
        min_trails_lb_sum=int(sum(r.get("min_trails_lb", 0) for r in boards_rows)),
        N_lift_sum=int(sum(r.get("N_lift", 0) for r in boards_rows)),
        L_air_sum=sum(r.get("L_air", 0.0) for r in boards_rows),
        L_cut_sum=sum(r.get("L_cut", 0.0) for r in boards_rows),
        # S1 decomposition (paper-friendly)
        L_cut_base_sum=sum(r.get("L_cut_base", 0.0) for r in boards_rows),
        L_kerf_extra_sum=sum(r.get("L_kerf_extra", 0.0) for r in boards_rows),
        L_lead_sum=sum(r.get("L_lead", 0.0) for r in boards_rows),
        T_est_sum=sum(r.get("T_est", 0.0) for r in boards_rows),
    )
