from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .packer import Board
from .routing import build_segments_from_board, estimate_cut_and_strokes

Point = Tuple[float, float]

STYLE = {
    "cut": "black",
    "shared": "#D62728",
    "air": "#B0B0B0",
    "start_marker": "#2CA02C",
}


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _save_png_and_svg(fig, out_path: str, dpi: int = 300) -> Tuple[str, str]:
    root, ext = os.path.splitext(str(out_path))
    if ext.lower() not in {".png", ".svg"}:
        out_path = str(out_path) + ".png"
    png_path = os.path.splitext(str(out_path))[0] + ".png"
    svg_path = os.path.splitext(str(out_path))[0] + ".svg"
    _ensure_dir(os.path.dirname(os.path.abspath(png_path)))
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def plot_board_toolpath(
    board: Board,
    out_path: Optional[str] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    *,
    share_mode: str = "union",
    cut_mode: str = "trail",
    origin: Point = (0.0, 0.0),
    show_sequence: bool = True,
    show_trim: bool = True,
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    tab_skip_trim_edge: bool = False,
    tab_adaptive: bool = False,
    tab_slender_ratio: float = 6.0,
    tab_slender_extra: int = 0,
    tab_small_area_extra: int = 0,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    shared_enable_edgeband: bool = False,
    shared_min_len_edgeband: float = 0.0,
    shared_max_continuous_cut: float = 0.0,
    shared_hold_bridge_len: float = 0.0,
    nd_coord: int = 6,
    trim: float = 0.0,
    route_start_policy: str = "none",
    route_priority: str = "none",
    route_ccw: bool = False,
    route_local_window: int = 10,
    route_local_backtrack: int = 2,
    route_small_first_area_mm2: float = 0.0,
    route_entry_junction_penalty_mm: float = 0.0,
    route_hierarchical: bool = False,
    route_large_frac: float = 0.20,
    route_ramp_enable: bool = False,
    route_ramp_len: float = 0.0,
    anti_shift_enable: bool = False,
    anti_shift_area_m2: float = 0.05,
    anti_shift_ar: float = 5.0,
    anti_shift_two_pass: bool = False,
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    use_curved_air: Optional[bool] = False,
) -> Optional[Tuple[str, str]]:
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    ax.add_patch(Rectangle((0, 0), board.W, board.H, facecolor="none", edgecolor="black", lw=0.6))
    trim_val = trim if trim > 0 else float(getattr(board, "trim", 0.0) or 0.0)
    if show_trim and trim_val > 0:
        ax.add_patch(
            Rectangle(
                (trim_val, trim_val),
                board.W - 2 * trim_val,
                board.H - 2 * trim_val,
                facecolor="none",
                edgecolor="#cccccc",
                lw=0.6,
                ls=":",
            )
        )

    segs, _, _, _ = build_segments_from_board(
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

    part_boxes = []
    try:
        for pp in getattr(board, "placed", []) or []:
            r = pp.rect
            xmin, ymin = float(r.x), float(r.y)
            xmax, ymax = float(r.x + r.w), float(r.y + r.h)
            area_mm2 = float(r.w) * float(r.h)
            ar = max(float(r.w), float(r.h)) / max(1e-9, min(float(r.w), float(r.h)))
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

    _, _, _, _, trails, _ = estimate_cut_and_strokes(
        segs,
        cut_mode=cut_mode,
        return_trails=True,
        origin=origin,
        nd=nd_coord,
        route_start_policy=route_start_policy,
        route_priority=route_priority,
        route_ccw=route_ccw,
        route_small_first_area_mm2=route_small_first_area_mm2,
        route_entry_junction_penalty_mm=route_entry_junction_penalty_mm,
        part_boxes=part_boxes,
        route_hierarchical=route_hierarchical,
        route_large_frac=route_large_frac,
        ramp_enable=route_ramp_enable,
        ramp_len=route_ramp_len,
        anti_shift_enable=anti_shift_enable,
        anti_shift_area_m2=anti_shift_area_m2,
        anti_shift_ar=anti_shift_ar,
        two_pass_enable=anti_shift_two_pass,
        feed_air=feed_air,
        t_lift=t_lift,
        route_local_window=route_local_window,
        route_local_backtrack=route_local_backtrack,
        board_bounds=(0.0, float(board.W), 0.0, float(board.H)),
    )

    for seg in segs:
        color = STYLE["shared"] if seg.shared else STYLE["cut"]
        width = 2.0 if seg.shared else 0.8
        ax.plot(
            [seg.a[0], seg.b[0]],
            [seg.a[1], seg.b[1]],
            color=color,
            lw=width,
            solid_capstyle="round",
            zorder=2 if seg.shared else 1,
        )

    use_curved = False if use_curved_air is None else bool(use_curved_air)
    cur = origin
    ax.scatter([origin[0]], [origin[1]], marker="s", s=25, c="black", zorder=10)
    for i, tr in enumerate(trails):
        if not tr:
            continue
        start = tr[0]
        dist = math.hypot(start[0] - cur[0], start[1] - cur[1])
        if dist > 1.0:
            if use_curved:
                arr = patches.FancyArrowPatch(
                    cur,
                    start,
                    arrowstyle="->,head_length=3,head_width=2",
                    connectionstyle="arc3,rad=0.15",
                    color=STYLE["air"],
                    lw=0.6,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(arr)
            else:
                ax.plot([cur[0], start[0]], [cur[1], start[1]], color=STYLE["air"], lw=0.6, ls="--", zorder=0)
            if show_sequence:
                mx = (cur[0] + start[0]) * 0.5
                my = (cur[1] + start[1]) * 0.5
                ax.text(
                    mx,
                    my,
                    str(i + 1),
                    fontsize=5,
                    color="#666666",
                    ha="center",
                    va="center",
                    zorder=11,
                )
        ax.scatter([start[0]], [start[1]], s=12, c=STYLE["start_marker"], zorder=6)
        if show_sequence:
            ax.text(
                start[0],
                start[1],
                str(i + 1),
                color="white",
                fontsize=5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc="black", ec="none", alpha=0.8),
                zorder=12,
            )
        cur = tr[-1]

    if created_fig and out_path:
        paths = _save_png_and_svg(fig, str(out_path))
        plt.close(fig)
        return paths
    if created_fig:
        plt.close(fig)
    return None


def plot_paper_comparison_4panel(board_baseline: Board, board_proposed: Board, out_path: str) -> None:
    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plot_board_toolpath(board_baseline, ax=ax1, share_mode="none", show_sequence=False)
    ax1.set_title("(a) Baseline Layout", y=-0.05, fontsize=11)
    plot_board_toolpath(board_proposed, ax=ax2, share_mode="union", show_sequence=False)
    ax2.set_title("(b) Proposed Layout (Shared)", y=-0.05, fontsize=11)
    plot_board_toolpath(board_baseline, ax=ax3, share_mode="none", show_sequence=True)
    ax3.set_title("(c) Baseline Sequence", y=-0.05, fontsize=11)
    plot_board_toolpath(board_proposed, ax=ax4, share_mode="union", show_sequence=True)
    ax4.set_title("(d) Proposed Sequence (Optimized)", y=-0.05, fontsize=11)

    _save_png_and_svg(fig, out_path)
    plt.close(fig)


def plot_board_layout(
    board,
    out_path,
    title: str = "",
    *,
    trim: float = 0.0,
    show_clearance: bool = True,
    show_ids: bool = False,
    show_dims: bool = True,
    vis_margin: float = 0.5,
    highlight_shared: bool = False,
    shared_overlay_mode: str = "union",
    share_mode: Optional[str] = None,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[Tuple[str, str]]:
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    W, H = float(board.W), float(board.H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="none", edgecolor="black", lw=0.6))

    trim_val = trim if trim > 0 else float(getattr(board, "trim", 0.0) or 0.0)
    if trim_val > 0:
        ax.add_patch(
            Rectangle(
                (trim_val, trim_val),
                W - 2 * trim_val,
                H - 2 * trim_val,
                facecolor="none",
                edgecolor="#cccccc",
                lw=0.6,
                ls=":",
            )
        )

    for pp in getattr(board, "placed", []):
        r = pp.rect
        if show_clearance:
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, facecolor="none", edgecolor="#bbbbbb", lw=0.6, ls="--", zorder=1))

        w0 = pp.h0 if getattr(pp, "rotated", False) else pp.w0
        h0 = pp.w0 if getattr(pp, "rotated", False) else pp.h0
        dx = max((r.w - w0) * 0.5, 0.0)
        dy = max((r.h - h0) * 0.5, 0.0)
        nominal_x = r.x + dx
        nominal_y = r.y + dy

        draw_x = nominal_x + vis_margin
        draw_y = nominal_y + vis_margin
        draw_w = max(0.1, w0 - 2 * vis_margin)
        draw_h = max(0.1, h0 - 2 * vis_margin)

        ax.add_patch(
            Rectangle(
                (draw_x, draw_y),
                draw_w,
                draw_h,
                facecolor="#f2f2f2",
                edgecolor="#444444",
                linewidth=0.8,
                zorder=2,
            )
        )

        labels: List[str] = []
        if show_ids:
            labels.append(f"#{pp.uid}")
        if show_dims:
            if float(w0).is_integer() and float(h0).is_integer():
                labels.append(f"{w0:.0f}x{h0:.0f}")
            else:
                labels.append(f"{w0:.1f}x{h0:.1f}")
        if labels:
            ax.text(
                nominal_x + w0 / 2,
                nominal_y + h0 / 2,
                "\n".join(labels),
                ha="center",
                va="center",
                fontsize=6,
                color="#111111",
                zorder=3,
            )

    overlay_mode = shared_overlay_mode if highlight_shared else (share_mode if share_mode is not None else None)
    if overlay_mode and overlay_mode != "none":
        try:
            segs_overlay, _, _, _ = build_segments_from_board(
                board,
                share_mode=overlay_mode,
                line_snap_eps=line_snap_eps,
                min_shared_len=min_shared_len,
                nd=nd_coord,
            )
            for s in segs_overlay:
                if s.shared:
                    ax.plot(
                        [s.a[0], s.b[0]],
                        [s.a[1], s.b[1]],
                        color=STYLE["shared"],
                        linewidth=2.0,
                        solid_capstyle="round",
                        zorder=5,
                    )
        except Exception:
            pass

    if created_fig and out_path:
        paths = _save_png_and_svg(fig, str(out_path))
        plt.close(fig)
        return paths
    if created_fig:
        plt.close(fig)
    return None


def plot_one_seed_outputs(
    boards,
    boards_rows: List[dict],
    out_dir,
    title: str,
    *,
    max_boards: int = 6,
    trim: float = 0.0,
    share_mode: str = "none",
    plot_toolpath: bool = True,
    show_clearance: bool = True,
    vis_margin: float = 0.5,
    show_ids: bool = False,
    show_dims: bool = True,
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    tab_skip_trim_edge: bool = False,
    tab_adaptive: bool = False,
    tab_slender_ratio: float = 6.0,
    tab_slender_extra: int = 0,
    tab_small_area_extra: int = 0,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    shared_enable_edgeband: bool = False,
    shared_min_len_edgeband: float = 0.0,
    shared_max_continuous_cut: float = 0.0,
    shared_hold_bridge_len: float = 0.0,
    nd_coord: int = 6,
    route_start_policy: str = "none",
    route_priority: str = "none",
    route_ccw: bool = False,
    route_local_window: int = 10,
    route_local_backtrack: int = 2,
    route_small_first_area_mm2: float = 0.0,
    route_entry_junction_penalty_mm: float = 0.0,
    route_hierarchical: bool = False,
    route_large_frac: float = 0.20,
    route_ramp_enable: bool = False,
    route_ramp_len: float = 0.0,
    anti_shift_enable: bool = False,
    anti_shift_area_m2: float = 0.05,
    anti_shift_ar: float = 5.0,
    anti_shift_two_pass: bool = False,
    feed_air: float = 30000.0,
    t_lift: float = 0.8,
    cut_mode: str = "trail",
) -> None:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir.as_posix())

    for b in boards[:max_boards]:
        plot_board_layout(
            b,
            out_dir / f"board_{b.bid:03d}_layout",
            title=f"{title} | board={b.bid} layout",
            trim=trim,
            show_clearance=show_clearance,
            show_ids=show_ids,
            show_dims=show_dims,
            vis_margin=vis_margin,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
        )
        if plot_toolpath:
            plot_board_toolpath(
                b,
                out_dir / f"board_{b.bid:03d}_toolpath",
                title=f"{title} | board={b.bid} toolpath",
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
                nd_coord=nd_coord,
                trim=trim,
                route_start_policy=route_start_policy,
                route_priority=route_priority,
                route_ccw=route_ccw,
                route_local_window=route_local_window,
                route_local_backtrack=route_local_backtrack,
                route_small_first_area_mm2=route_small_first_area_mm2,
                route_entry_junction_penalty_mm=route_entry_junction_penalty_mm,
                route_hierarchical=route_hierarchical,
                route_large_frac=route_large_frac,
                route_ramp_enable=route_ramp_enable,
                route_ramp_len=route_ramp_len,
                anti_shift_enable=anti_shift_enable,
                anti_shift_area_m2=anti_shift_area_m2,
                anti_shift_ar=anti_shift_ar,
                anti_shift_two_pass=anti_shift_two_pass,
                feed_air=feed_air,
                t_lift=t_lift,
                cut_mode=cut_mode,
            )

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [title, "", "Metrics Summary:"]
    for row in boards_rows:
        bid = int(row.get("board", row.get("bid", -1)))
        n_parts = int(row.get("n_parts", row.get("N_parts", 0)))
        util = float(row.get("U", 0.0) or 0.0) * 100.0
        L_cut = float(row.get("L_cut", 0.0) or 0.0)
        L_air = float(row.get("L_air", 0.0) or 0.0)
        risk_shared = float(row.get("Risk_shared", 0.0) or 0.0)
        lines.append(
            f"Board {bid:02d}: Parts={n_parts} Util={util:.1f}% L_cut={L_cut:.0f}mm "
            f"L_air={L_air:.0f}mm Risk_shared={risk_shared:.3f}"
        )

    ax.text(0.05, 0.9, "\n".join(lines), va="top", ha="left", fontsize=10, fontfamily="monospace")
    _save_png_and_svg(fig, out_dir / "boards_metrics_preview")
    plt.close(fig)
