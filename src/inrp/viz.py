# src/inrp/viz.py
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# NOTE:
# This viz module expects routing.py to provide the endpoint-aware functions:
# - endpoint_nn_order
# - endpoint_two_opt
# - air_moves_from_order_and_dirs
from .routing import (
    build_segments_from_board,
    estimate_cut_and_strokes,
    endpoint_nn_order,
    endpoint_two_opt,
    air_moves_from_order_and_dirs,
)

Point = Tuple[float, float]


def _save_png_and_svg(fig, out_path: Path, *, dpi: int = 300) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        return plt, patches
    except Exception as e:
        logger.warning("[PLOT] skip (matplotlib not available): %s", e)
        return None, None


def _is_int_like(x, eps: float = 1e-6) -> bool:
    try:
        xf = float(x)
        return abs(xf - round(xf)) <= eps
    except Exception:
        return False


def _fmt_dims(w, h) -> str:
    if _is_int_like(w) and _is_int_like(h):
        return f"{int(round(float(w)))}x{int(round(float(h)))}"
    return f"{float(w):.1f}x{float(h):.1f}"


def plot_board_layout(
        board,
        out_path: Path,
        title: str,
        *,
        trim: float = 0.0,
        show_clearance: bool = True,
        show_ids: bool = False,
        show_dims: bool = True,
        vis_margin: float = 0.5,  # purely for visualization
        pad: float = 0.0,
        dpi: int = 300,
) -> None:
    """
    Draw board layout:
    - Grey background board
    - Optional trim rectangle
    - Optional clearance rectangles (packing rect)
    - Nominal part rectangles with visual shrink (vis_margin) so gaps are visible
    """
    plt, mpatches = _safe_import_matplotlib()
    if plt is None:
        return

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-pad, board.W + pad)
    ax.set_ylim(-pad, board.H + pad)

    # Board background
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0), board.W, board.H,
            fill=True, facecolor="#eeeeee",
            edgecolor="black", linewidth=1.5
        )
    )

    # Trim boundary
    if trim > 0:
        inner_w = board.W - 2 * trim
        inner_h = board.H - 2 * trim
        if inner_w > 0 and inner_h > 0:
            ax.add_patch(
                mpatches.Rectangle(
                    (trim, trim), inner_w, inner_h,
                    fill=False, linestyle="--",
                    linewidth=1.0, edgecolor="#888888"
                )
            )
        else:
            logger.warning("[PLOT] trim too large for board: trim=%s W=%s H=%s", trim, board.W, board.H)

    for pp in getattr(board, "placed", []):
        r = pp.rect  # expected: x,y,w,h

        # Clearance / packing rect boundary
        if show_clearance:
            ax.add_patch(
                mpatches.Rectangle(
                    (r.x, r.y), r.w, r.h,
                    fill=False, linewidth=0.5,
                    linestyle=":", edgecolor="blue", alpha=0.35
                )
            )

        rotated = bool(getattr(pp, "rotated", False))
        w0 = pp.h0 if rotated else pp.w0
        h0 = pp.w0 if rotated else pp.h0

        # Center nominal part in its packing rect
        dx = max((float(r.w) - float(w0)) * 0.5, 0.0)
        dy = max((float(r.h) - float(h0)) * 0.5, 0.0)
        nominal_x = float(r.x) + dx
        nominal_y = float(r.y) + dy

        # Visual shrink for gap
        draw_x = nominal_x + vis_margin
        draw_y = nominal_y + vis_margin
        draw_w = max(0.1, float(w0) - 2 * vis_margin)
        draw_h = max(0.1, float(h0) - 2 * vis_margin)

        ax.add_patch(
            mpatches.Rectangle(
                (draw_x, draw_y), draw_w, draw_h,
                fill=True, facecolor="#aaddff",
                edgecolor="black", linewidth=0.8, alpha=0.95
            )
        )

        # Labels
        lines = []
        if show_ids:
            lines.append(f"#{getattr(pp, 'uid', '?')}")
        if show_dims:
            lines.append(_fmt_dims(w0, h0))

        if lines:
            sz = min(float(w0), float(h0))
            fs = 8 if sz > 200 else (6 if sz > 100 else 4)
            ax.text(
                nominal_x + float(w0) / 2.0,
                nominal_y + float(h0) / 2.0,
                "\n".join(lines),
                ha="center", va="center",
                fontsize=fs, color="#333333"
            )

    _save_png_and_svg(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_board_toolpath(
        board,
        out_path: Path,
        title: str,
        *,
        share_mode: str,
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        trim: float = 0.0,
        cut_mode: str = "trail",
        origin: Point = (0.0, 0.0),
        pad: float = 10.0,
        dpi: int = 300,
        # baseline jitter (only for share_mode == "none")
        jitter_seed: Optional[int] = None,
        jitter_mm: float = 0.6,
        # stroke labels
        stroke_label_n: int = 30,
        stroke_label_fontsize: int = 7,
        # air move batching
        air_min_len: float = 2.0,
        air_arrow_len: float = 18.0,
        air_arrow_width: float = 6.0,
        # direction arrow
        draw_direction: bool = True,
        dir_arrow_len: float = 14.0,
        dir_arrow_width: float = 5.0,
        # IMPORTANT for paper: whether air tour returns to origin
        return_to_origin: bool = False,
        # optional: cap number of strokes drawn (visual clutter control)
        max_strokes_draw: Optional[int] = None,
) -> None:
    """
    Paper-grade aligned visualization:
    - Air-moves (orange) are generated by endpoint-aware order + per-stroke direction,
      and L_air computed from the SAME air-move segments.
    - Cut segments: normal blue below, shared red above (clear layering)
    - Stroke numbering: first N strokes in chosen order
    - Direction arrows: consistent with chosen per-stroke direction
    """
    plt, mpatches = _safe_import_matplotlib()
    if plt is None:
        return

    try:
        from matplotlib.collections import LineCollection
    except Exception as e:
        logger.warning("[PLOT] skip (LineCollection not available): %s", e)
        return

    # 1) Build segments
    segs, _, _ = build_segments_from_board(
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

    # 2) Build strokes/trails
    try:
        _, _, n_strokes, _, stroke_trails, _ = estimate_cut_and_strokes(
            segs, cut_mode=cut_mode, return_trails=True
        )
        if n_strokes <= 0 or not stroke_trails:
            stroke_trails = []
    except Exception as e:
        logger.warning("Failed to compute strokes for viz: %s", e)
        stroke_trails = []

    # 3) Endpoint-aware: order + dirs + air segments (THIS aligns numbers with the plot)
    order: list[int] = []
    dirs: list[int] = []
    air_segs: list[tuple[Point, Point]] = []
    try:
        if stroke_trails:
            order0, dirs0, _ = endpoint_nn_order(stroke_trails, start=origin, return_to_start=return_to_origin)
            order, dirs, L_air = endpoint_two_opt(
                order0, stroke_trails, start=origin, iters=120, return_to_start=return_to_origin
            )
            air_segs, L_air_check = air_moves_from_order_and_dirs(
                order, dirs, stroke_trails, start=origin, return_to_start=return_to_origin
            )
            # They should match (floating tolerance)
            if abs(float(L_air) - float(L_air_check)) > 1e-6:
                logger.warning("[PLOT] L_air mismatch: two_opt=%s vs built=%s", L_air, L_air_check)
    except Exception as e:
        logger.warning("Failed endpoint-aware tour for viz: %s", e)
        order, dirs, air_segs = [], [], []

    # Optional visual cap
    if max_strokes_draw is not None and order:
        k = int(max_strokes_draw)
        k = max(0, min(k, len(order)))
        # Keep only first k strokes; rebuild air segs accordingly (paper uses full metrics elsewhere)
        order = order[:k]
        dirs = dirs[:k]
        try:
            air_segs, _ = air_moves_from_order_and_dirs(order, dirs, stroke_trails, start=origin, return_to_start=False)
        except Exception:
            air_segs = []

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-pad, board.W + pad)
    ax.set_ylim(-pad, board.H + pad)

    # Board outline (bottom layer)
    ax.add_patch(
        mpatches.Rectangle((0, 0), board.W, board.H, fill=False, linewidth=1.0, edgecolor="black", zorder=1)
    )
    if trim > 0:
        inner_w = board.W - 2 * trim
        inner_h = board.H - 2 * trim
        if inner_w > 0 and inner_h > 0:
            ax.add_patch(
                mpatches.Rectangle(
                    (trim, trim), inner_w, inner_h,
                    fill=False, linestyle="--", linewidth=0.6, edgecolor="gray", zorder=1
                )
            )

    # 4) Cuts (batch render + layering)
    apply_jitter = (share_mode == "none")
    seed = jitter_seed if jitter_seed is not None else int(getattr(board, "bid", 0))
    rng = random.Random(seed)

    lines_cut = []
    lines_shared = []
    for s in segs:
        (x1, y1), (x2, y2) = s.a, s.b
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        if apply_jitter:
            dx = rng.uniform(-jitter_mm, jitter_mm)
            dy = rng.uniform(-jitter_mm, jitter_mm)
            x1 += dx; x2 += dx; y1 += dy; y2 += dy

        seg_line = [(x1, y1), (x2, y2)]
        if bool(getattr(s, "shared", False)):
            lines_shared.append(seg_line)
        else:
            lines_cut.append(seg_line)

    if lines_cut:
        lc_cut = LineCollection(
            lines_cut,
            colors="#1f77b4",
            linewidths=1.35,
            alpha=0.60 if apply_jitter else 0.75,
            zorder=2,
            capstyle="round",
        )
        ax.add_collection(lc_cut)

    if lines_shared:
        lc_shared = LineCollection(
            lines_shared,
            colors="#d62728",
            linewidths=2.6,
            alpha=0.92,
            zorder=3,
            capstyle="round",
        )
        ax.add_collection(lc_shared)

    # 5) Air moves (batch) + arrowheads (batch)
    if air_segs:
        # filter short air moves for readability (still keeps metric integrity elsewhere)
        air_lines = []
        bases_x, bases_y, vec_x, vec_y = [], [], [], []
        Lh = float(air_arrow_len)
        for (p, q) in air_segs:
            p = (float(p[0]), float(p[1]))
            q = (float(q[0]), float(q[1]))
            dx, dy = q[0] - p[0], q[1] - p[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= float(air_min_len):
                continue
            air_lines.append([p, q])

            # arrowhead near END: base point = q - u*Lh
            ux, uy = dx / dist, dy / dist
            bx = q[0] - ux * Lh
            by = q[1] - uy * Lh
            bases_x.append(bx); bases_y.append(by)
            vec_x.append(ux * Lh); vec_y.append(uy * Lh)

        if air_lines:
            lc_air = LineCollection(
                air_lines,
                colors="#ff7f0e",
                linewidths=1.0,
                linestyles="--",
                alpha=0.55,
                zorder=4,
                capstyle="round",
            )
            ax.add_collection(lc_air)

            # arrowheads
            ax.quiver(
                bases_x, bases_y, vec_x, vec_y,
                angles="xy", scale_units="xy", scale=1,
                width=0.003,
                headwidth=air_arrow_width,
                headlength=air_arrow_width,
                headaxislength=air_arrow_width,
                color="#ff7f0e",
                alpha=0.55,
                zorder=5,
            )

    # Origin marker
    ox, oy = float(origin[0]), float(origin[1])
    ax.plot(ox, oy, "ko", markersize=6, zorder=6)

    # 6) Stroke numbering + direction arrows (consistent with dirs)
    if stroke_trails and order and dirs:
        label_n = min(int(stroke_label_n), len(order))

        for i in range(label_n):
            idx = order[i]
            dflag = dirs[i]
            if idx < 0 or idx >= len(stroke_trails):
                continue
            trail = stroke_trails[idx]
            if not trail:
                continue

            # apply direction (0: normal, 1: reversed)
            if int(dflag) == 1:
                t = list(reversed(trail))
            else:
                t = trail

            # label near start of the stroke (consistent with chosen entry endpoint)
            spt = (float(t[0][0]), float(t[0][1]))
            ax.text(
                spt[0] + 3.0, spt[1] + 3.0, str(i + 1),
                fontsize=stroke_label_fontsize,
                color="#111111",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                )

            # direction arrow on the stroke middle (consistent with chosen direction)
            if draw_direction and len(t) >= 2:
                mid_idx = len(t) // 2
                p0 = t[mid_idx - 1] if mid_idx > 0 else t[0]
                p1 = t[mid_idx]
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])
                vx, vy = x1 - x0, y1 - y0
                vlen = (vx * vx + vy * vy) ** 0.5
                if vlen > 1.0:
                    ux, uy = vx / vlen, vy / vlen
                    ax.quiver(
                        [x1 - ux * (dir_arrow_len * 0.5)],
                        [y1 - uy * (dir_arrow_len * 0.5)],
                        [ux * dir_arrow_len],
                        [uy * dir_arrow_len],
                        angles="xy", scale_units="xy", scale=1,
                        width=0.003,
                        headwidth=dir_arrow_width,
                        headlength=dir_arrow_width,
                        headaxislength=dir_arrow_width,
                        color="black",
                        alpha=0.9,
                        zorder=7,
                    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1f77b4", lw=2, alpha=0.75, label="Cut (Blue)"),
        Line2D([0], [0], color="#d62728", lw=2.8, alpha=0.92, label="Shared (Red)"),
        Line2D([0], [0], color="#ff7f0e", lw=1, linestyle="--", alpha=0.55, label="Air Move"),
        Line2D([0], [0], marker=r"$\rightarrow$", color="k", label="Direction", markersize=10, linestyle="None"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    _save_png_and_svg(fig, out_path, dpi=dpi)
    plt.close(fig)


def plot_one_seed_outputs(
        boards,
        boards_rows,
        out_dir: Path,
        title: str,
        *,
        max_boards: int = 6,
        trim: float = 0.0,
        share_mode: str = "none",
        plot_toolpath: bool = True,
        show_ids: bool = False,
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        # layout controls
        vis_margin: float = 0.5,
        pad_layout: float = 0.0,
        # toolpath controls
        pad_toolpath: float = 10.0,
        origin: Point = (0.0, 0.0),
        jitter_seed: Optional[int] = None,
        jitter_mm: float = 0.6,
        stroke_label_n: int = 30,
        return_to_origin: bool = False,
        dpi: int = 300,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in boards[:max_boards]:
        plot_board_layout(
            b,
            out_dir / f"board_{b.bid:03d}_layout.png",
            title=f"{title} | board={b.bid} layout",
            trim=trim,
            show_clearance=True,
            show_ids=show_ids,
            show_dims=True,
            vis_margin=vis_margin,
            pad=pad_layout,
            dpi=dpi,
            )

        if plot_toolpath:
            plot_board_toolpath(
                b,
                out_dir / f"board_{b.bid:03d}_toolpath.png",
                title=f"{title} | board={b.bid} toolpath",
                share_mode=share_mode,
                tab_enable=tab_enable,
                tab_per_part=tab_per_part,
                tab_len=tab_len,
                tab_corner_clear=tab_corner_clear,
                line_snap_eps=line_snap_eps,
                min_shared_len=min_shared_len,
                nd_coord=nd_coord,
                trim=trim,
                cut_mode="trail",
                origin=origin,
                pad=pad_toolpath,
                jitter_seed=jitter_seed,
                jitter_mm=jitter_mm,
                stroke_label_n=stroke_label_n,
                return_to_origin=return_to_origin,
                dpi=dpi,
                )

    # Metrics preview (does not compute metrics; just prints provided rows)
    plt, _ = _safe_import_matplotlib()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [title, "", "Metrics Summary:"]
    for row in boards_rows:
        lines.append(
            f"Board {int(row['board']):02d}: "
            f"Parts={int(row['n_parts'])} "
            f"Util={row['U'] * 100:.1f}% "
            f"L_cut={row['L_cut']:.0f}mm "
            f"L_air={row['L_air']:.0f}mm"
        )

    ax.text(
        0.05, 0.9, "\n".join(lines),
        va="top", ha="left",
        fontsize=10, fontfamily="monospace"
    )

    _save_png_and_svg(fig, out_dir / "boards_metrics_preview.png", dpi=dpi)
    plt.close(fig)
