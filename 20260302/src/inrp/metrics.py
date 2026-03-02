from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from .routing import (
    air_length_by_strokes,
    build_segments_from_board,
    estimate_cut_and_strokes,
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
    tab_skip_trim_edge: bool = False,
    tab_adaptive: bool = False,
    tab_slender_ratio: float = 6.0,
    tab_slender_extra: int = 0,
    tab_small_area_extra: int = 0,
    # shared-edge detection robustness and eligibility
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    shared_enable_edgeband: bool = False,
    shared_min_len_edgeband: float = 0.0,
    shared_max_continuous_cut: float = 0.0,
    shared_hold_bridge_len: float = 0.0,
    shared_require_dual_on_edgeband: bool = True,
    shared_pass_mode: str = "none",
    shared_pass_mult: float = 1.0,
    shared_risk_w: float = 0.0,
    # route heuristics
    route_start_policy: str = "none",
    route_priority: str = "none",
    route_ccw: bool = False,
    route_local_window: int = 10,
    route_local_backtrack: int = 2,
    route_small_first_area_mm2: float = 0.0,
    route_entry_junction_penalty_mm: float = 0.0,
    # CNC process-rule routing
    route_hierarchical: bool = False,
    route_large_frac: float = 0.20,
    route_ramp_enable: bool = False,
    route_ramp_len: float = 0.0,
    anti_shift_enable: bool = False,
    anti_shift_area_m2: float = 0.05,
    anti_shift_ar: float = 5.0,
    anti_shift_two_pass: bool = False,
    nd_coord: int = 6,
) -> Dict[str, Any]:
    n_parts = len(board.placed)

    used = sum(pp.w0 * pp.h0 for pp in board.placed)
    U = used / (board_W * board_H) if board_W * board_H > 0 else 0.0

    segs, L_shared, n_tabs, shared_info = build_segments_from_board(
        board,
        share_mode=share_mode,
        tab_enable=tab_enable,
        tab_per_part=tab_per_part,
        tab_len=tab_len,
        tab_corner_clear=tab_corner_clear,
        tab_skip_trim_edge=tab_skip_trim_edge,
        tab_adaptive=tab_adaptive,
        tab_slender_ratio=tab_slender_ratio,
        tab_slender_extra=tab_slender_extra,
        tab_small_area_extra=tab_small_area_extra,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        shared_enable_edgeband=shared_enable_edgeband,
        shared_min_len_edgeband=shared_min_len_edgeband,
        shared_max_continuous_cut=shared_max_continuous_cut,
        shared_hold_bridge_len=shared_hold_bridge_len,
        nd=nd_coord,
    )

    route_start_eff = route_start_policy if share_mode == "union" else "none"
    route_prio_eff = route_priority if share_mode == "union" else "none"
    route_ccw_eff = route_ccw if share_mode == "union" else False
    route_win_eff = route_local_window if share_mode == "union" else 10
    route_back_eff = route_local_backtrack if share_mode == "union" else 2

    # Part bounding boxes for hierarchical routing / anti-shifting rules
    part_boxes = []
    try:
        for pp in getattr(board, "placed", []) or []:
            r = pp.rect
            xmin, ymin = float(r.x), float(r.y)
            xmax, ymax = float(r.x + r.w), float(r.y + r.h)
            area_mm2 = float(r.w) * float(r.h)
            ar = (max(float(r.w), float(r.h)) / max(1e-9, min(float(r.w), float(r.h))))
            part_boxes.append({
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                "area_mm2": area_mm2,
                "ar": float(ar),
            })
    except Exception:
        part_boxes = []

    L_cut_base, n_comp, n_stroke, stroke_reps, comps, route_metrics = estimate_cut_and_strokes(
        segs,
        cut_mode=cut_mode,
        origin=(0.0, 0.0),
        nd=nd_coord,
        route_start_policy=route_start_eff,
        route_priority=route_prio_eff,
        route_ccw=route_ccw_eff,
        route_small_first_area_mm2=route_small_first_area_mm2,
        route_entry_junction_penalty_mm=route_entry_junction_penalty_mm,
        part_boxes=part_boxes,
        route_hierarchical=bool(route_hierarchical),
        route_large_frac=float(route_large_frac),
        ramp_enable=bool(route_ramp_enable),
        ramp_len=float(route_ramp_len),
        anti_shift_enable=bool(anti_shift_enable),
        anti_shift_area_m2=float(anti_shift_area_m2),
        anti_shift_ar=float(anti_shift_ar),
        two_pass_enable=bool(anti_shift_two_pass),
        feed_air=feed_air,
        t_lift=t_lift,
        route_local_window=route_win_eff,
        route_local_backtrack=route_back_eff,
        board_bounds=(0.0, float(board_W), 0.0, float(board_H)),
        return_route_metrics=True,
    )

    odd_total, min_trails, _per = trails_lower_bound_from_segments(segs)

    r = max(float(tool_d), 0.0) * 0.5
    L_kerf_extra = 0.0
    if kerf_mode == "tool_center" and r > 0 and n_stroke > 0:
        L_kerf_extra = 2.0 * math.pi * r * float(n_stroke)

    L_lead = 0.0
    if (lead_in > 0 or lead_out > 0) and n_stroke > 0:
        L_lead = (max(float(lead_in), 0.0) + max(float(lead_out), 0.0)) * float(n_stroke)

    L_shared_edgeband = float(shared_info.get("L_shared_edgeband", 0.0))
    pass_mode = str(shared_pass_mode or "none").strip().lower()
    pass_mult = max(1.0, float(shared_pass_mult))
    L_dual_pass_extra = 0.0
    if share_mode == "union":
        if pass_mode == "global":
            L_dual_pass_extra = (pass_mult - 1.0) * max(float(L_shared), 0.0)
            if shared_require_dual_on_edgeband and pass_mult < 2.0:
                # Ensure edgeband shared segments are at least dual-pass.
                L_dual_pass_extra += (2.0 - pass_mult) * max(L_shared_edgeband, 0.0)
        elif pass_mode in {"edgeband", "edgeband_only"}:
            mult_eff = max(pass_mult, 2.0) if shared_require_dual_on_edgeband else pass_mult
            L_dual_pass_extra = (mult_eff - 1.0) * max(L_shared_edgeband, 0.0)
        elif pass_mode == "none" and shared_require_dual_on_edgeband:
            # Conservative fallback: force dual pass only on edgeband shared segments.
            L_dual_pass_extra = max(L_shared_edgeband, 0.0)

    L_cut = L_cut_base + L_dual_pass_extra + L_kerf_extra + L_lead

    L_air = float(route_metrics.get("L_air", air_length_by_strokes(stroke_reps, start=(0.0, 0.0))))
    N_lift = n_stroke

    T_cut = (L_cut / feed_cut) * 60.0 if feed_cut > 0 else 0.0
    T_air = (L_air / feed_air) * 60.0 if feed_air > 0 else 0.0
    T_lift = N_lift * t_lift
    T_est = T_cut + T_air + T_lift

    risk_shared_raw = 0.0
    if L_shared > 1e-9:
        risk_shared_raw = max(0.0, min(1.0, L_shared_edgeband / float(L_shared)))
    if shared_risk_w > 0:
        risk_shared = risk_shared_raw * float(shared_risk_w)
    else:
        risk_shared = risk_shared_raw

    n_slender = 0
    for pp in board.placed:
        a = max(float(pp.w0), float(pp.h0))
        b = max(1e-9, min(float(pp.w0), float(pp.h0)))
        if (a / b) >= float(tab_slender_ratio):
            n_slender += 1
    risk_slender = (float(n_slender) / float(n_parts)) if n_parts > 0 else 0.0

    return dict(
        board=board.bid,
        n_parts=n_parts,
        U=U,
        L_shared=L_shared if share_mode == "union" else 0.0,
        L_shared_edgeband=L_shared_edgeband if share_mode == "union" else 0.0,
        L_shared_candidate=float(shared_info.get("L_shared_candidate", 0.0)),
        L_shared_candidate_edgeband=float(shared_info.get("L_shared_candidate_edgeband", 0.0)),
        n_shared=int(shared_info.get("N_shared", 0.0)),
        n_shared_edgeband=int(shared_info.get("N_shared_edgeband", 0.0)),
        n_comp=n_comp,
        n_stroke=n_stroke,
        n_tabs=n_tabs,
        odd_total=odd_total,
        min_trails_lb=min_trails,
        stroke_over_lb=(float(n_stroke) / float(min_trails) if min_trails > 0 else 0.0),
        N_lift=N_lift,
        L_air=L_air,
        L_cut=L_cut,
        L_cut_base=L_cut_base,
        L_dual_pass_extra=L_dual_pass_extra,
        L_kerf_extra=L_kerf_extra,
        L_lead=L_lead,
        kerf_mode=kerf_mode,
        tool_d=tool_d,
        lead_in=lead_in,
        lead_out=lead_out,
        Risk_shared=risk_shared,
        Risk_shared_raw=risk_shared_raw,
        Risk_slender=risk_slender,
        n_slender=n_slender,
        T_cut=T_cut,
        T_air=T_air,
        T_lift=T_lift,
        T_est=T_est,
    )


def aggregate_totals(boards_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not boards_rows:
        return dict(
            N_board=0,
            U_avg=0.0,
            U_avg_excl_last=0.0,
            L_shared_sum=0.0,
            L_shared_edgeband_sum=0.0,
            n_comp_sum=0,
            n_stroke_sum=0,
            n_tabs_sum=0,
            odd_total_sum=0,
            min_trails_lb_sum=0,
            N_lift_sum=0,
            L_air_sum=0.0,
            L_cut_sum=0.0,
            L_cut_base_sum=0.0,
            L_dual_pass_extra_sum=0.0,
            L_kerf_extra_sum=0.0,
            L_lead_sum=0.0,
            Risk_shared_sum=0.0,
            Risk_shared_avg=0.0,
            Risk_slender_sum=0.0,
            Risk_slender_avg=0.0,
            n_slender_sum=0,
            T_cut_sum=0.0,
            T_air_sum=0.0,
            T_lift_sum=0.0,
            T_est_sum=0.0,
        )

    N_board = len(boards_rows)
    U_avg = statistics.mean([r["U"] for r in boards_rows])
    if N_board > 1:
        U_avg_excl_last = statistics.mean([r["U"] for r in boards_rows[:-1]])
    else:
        U_avg_excl_last = U_avg
    return dict(
        N_board=N_board,
        U_avg=U_avg,
        U_avg_excl_last=U_avg_excl_last,
        L_shared_sum=sum(r.get("L_shared", 0.0) for r in boards_rows),
        L_shared_edgeband_sum=sum(r.get("L_shared_edgeband", 0.0) for r in boards_rows),
        n_comp_sum=int(sum(r.get("n_comp", 0) for r in boards_rows)),
        n_stroke_sum=int(sum(r.get("n_stroke", 0) for r in boards_rows)),
        n_tabs_sum=int(sum(r.get("n_tabs", 0) for r in boards_rows)),
        odd_total_sum=int(sum(r.get("odd_total", 0) for r in boards_rows)),
        min_trails_lb_sum=int(sum(r.get("min_trails_lb", 0) for r in boards_rows)),
        N_lift_sum=int(sum(r.get("N_lift", 0) for r in boards_rows)),
        L_air_sum=sum(r.get("L_air", 0.0) for r in boards_rows),
        L_cut_sum=sum(r.get("L_cut", 0.0) for r in boards_rows),
        L_cut_base_sum=sum(r.get("L_cut_base", 0.0) for r in boards_rows),
        L_dual_pass_extra_sum=sum(r.get("L_dual_pass_extra", 0.0) for r in boards_rows),
        L_kerf_extra_sum=sum(r.get("L_kerf_extra", 0.0) for r in boards_rows),
        L_lead_sum=sum(r.get("L_lead", 0.0) for r in boards_rows),
        Risk_shared_sum=sum(r.get("Risk_shared", 0.0) for r in boards_rows),
        Risk_shared_avg=statistics.mean([r.get("Risk_shared", 0.0) for r in boards_rows]),
        Risk_slender_sum=sum(r.get("Risk_slender", 0.0) for r in boards_rows),
        Risk_slender_avg=statistics.mean([r.get("Risk_slender", 0.0) for r in boards_rows]),
        n_slender_sum=int(sum(r.get("n_slender", 0) for r in boards_rows)),
        T_cut_sum=sum(r.get("T_cut", 0.0) for r in boards_rows),
        T_air_sum=sum(r.get("T_air", 0.0) for r in boards_rows),
        T_lift_sum=sum(r.get("T_lift", 0.0) for r in boards_rows),
        T_est_sum=sum(r.get("T_est", 0.0) for r in boards_rows),
    )
