"""Visualization for INRP (Integrated Nesting & Routing) experiments.

This version is tuned for CIE-style figures:
- Layout figure: grey board background, thin part outlines, optional trim boundary.
  Optional clearance rectangles (packing rectangles) and visual shrink margin to
  make gaps visible even when parts touch.
- Toolpath figure: cut segments (blue), shared segments (red & thicker), air
  moves (orange dashed arrows), direction arrows (black) and optional deterministic
  jitter for baseline overlays.
- One-click export: per-board layout/toolpath in PNG+SVG, plus a metrics preview.

All functions are safe to call in headless environments (matplotlib 'Agg').
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Headless-safe backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

from .packer import Board, PlacedPart
from .routing import (
    Segment,
    build_segments_from_board,
    estimate_cut_and_strokes,
    nn_tour,
    two_opt,
)

Point = Tuple[float, float]


# ---------------------------
# Small utilities
# ---------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_png_and_svg(fig, out_path: str, dpi: int = 220) -> Tuple[str, str]:
    """Save figure to out_path (png) and same-name svg.

    If out_path endswith .png/.svg it will be respected; the sibling format is
    generated with the same stem.
    """
    root, ext = os.path.splitext(out_path)
    if ext.lower() not in {".png", ".svg"}:
        # default to png
        png_path = out_path + ".png"
        svg_path = out_path + ".svg"
    else:
        png_path = root + ".png"
        svg_path = root + ".svg"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


def _fmt_dim(v) -> str:
    """Format int/float dimensions robustly."""
    try:
        fv = float(v)
        if math.isfinite(fv) and abs(fv - round(fv)) < 1e-9:
            return str(int(round(fv)))
        return f"{fv:.1f}".rstrip("0").rstrip(".")
    except Exception:
        return str(v)


def _orig_rect_for_display(pp: PlacedPart) -> Tuple[float, float, float, float]:
    """Compute the *original* rectangle position (undo symmetric inflation).

    pp.rect is the placement rectangle used by the packer (may include inflation
    for clearance/gap). pp.w0/pp.h0 are the original dims (pre-inflation).

    Returns (x, y, w, h) for the original part rectangle.
    """
    x, y, w, h = pp.rect
    if pp.rot:
        ow, oh = pp.h0, pp.w0
    else:
        ow, oh = pp.w0, pp.h0

    dx = (w - ow) / 2.0
    dy = (h - oh) / 2.0
    return x + dx, y + dy, ow, oh


def _shrink_rect(x: float, y: float, w: float, h: float, margin: float) -> Tuple[float, float, float, float]:
    if margin <= 0:
        return x, y, w, h
    return x + margin, y + margin, max(0.0, w - 2 * margin), max(0.0, h - 2 * margin)


def _stroke_direction_arrow(ax, trail: List[Point], color: str = "k", scale: float = 1.0) -> None:
    """Draw a small direction arrow near the middle of a polyline."""
    if len(trail) < 2:
        return
    mid = len(trail) // 2
    i0 = max(0, mid - 1)
    i1 = min(len(trail) - 1, mid)
    p0 = trail[i0]
    p1 = trail[i1]
    dx, dy = (p1[0] - p0[0]), (p1[1] - p0[1])
    L = math.hypot(dx, dy)
    if L <= 1e-9:
        return
    # Arrow length around ~25mm in model coords
    alen = 25.0 * scale
    ux, uy = dx / L, dy / L
    sx, sy = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    ex, ey = sx + ux * alen, sy + uy * alen
    ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="->", mutation_scale=8, lw=0.8, color=color))


@dataclass
class _OrientedStroke:
    trail: List[Point]

    @property
    def start(self) -> Point:
        return self.trail[0]

    @property
    def end(self) -> Point:
        return self.trail[-1]

    def reversed(self) -> "_OrientedStroke":
        return _OrientedStroke(list(reversed(self.trail)))


def _orient_strokes_by_order(stroke_trails: List[List[Point]], order: List[int], origin: Point) -> List[_OrientedStroke]:
    """Given an order, choose direction per stroke to reduce air moves."""
    oriented: List[_OrientedStroke] = []
    cur = origin
    for idx in order:
        tr = stroke_trails[idx]
        s = _OrientedStroke(tr)
        d_fwd = math.hypot(s.start[0] - cur[0], s.start[1] - cur[1])
        d_rev = math.hypot(s.end[0] - cur[0], s.end[1] - cur[1])
        if d_rev < d_fwd:
            s = s.reversed()
        oriented.append(s)
        cur = s.end
    return oriented


def _compute_air_length(oriented: List[_OrientedStroke], origin: Point, close_to_origin: bool = False) -> float:
    L = 0.0
    cur = origin
    for s in oriented:
        L += math.hypot(s.start[0] - cur[0], s.start[1] - cur[1])
        cur = s.end
    if close_to_origin and oriented:
        L += math.hypot(cur[0] - origin[0], cur[1] - origin[1])
    return L


# ---------------------------
# Main plotting APIs
# ---------------------------

def plot_board_layout(
        board: Board,
        out_path: str,
        *,
        show_trim: bool = True,
        show_clearance: bool = True,
        vis_margin: float = 0.5,
        show_ids: bool = False,
        show_dims: bool = False,
        title: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot a board layout (PNG + SVG).

    Parameters
    ----------
    board:
        Board instance.
    out_path:
        Output path (either with extension or without). Both PNG and SVG will be saved.
    show_clearance:
        If True, draw packing rectangles (inflated/clearance boundary) as blue dotted lines.
    vis_margin:
        Visual-only shrink margin (mm). Applied to the *real* part outline so that
        even touching parts show a visible gap.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Board background
    ax.add_patch(Rectangle((0, 0), board.W, board.H, facecolor="#dddddd", edgecolor="black", lw=1.0, zorder=0))

    # Trim boundary
    if show_trim and board.trim and board.trim > 0:
        ax.add_patch(
            Rectangle(
                (board.trim, board.trim),
                board.W - 2 * board.trim,
                board.H - 2 * board.trim,
                facecolor="none",
                edgecolor="#777777",
                lw=0.9,
                ls=(0, (2, 2)),
                zorder=1,
            )
        )

    # Parts
    for pp in board.placed:
        # clearance rect
        if show_clearance:
            x, y, w, h = pp.rect
            ax.add_patch(
                Rectangle(
                    (x, y), w, h,
                    facecolor="none",
                    edgecolor="#1f77b4",
                    lw=0.8,
                    ls=(0, (2, 2)),
                    zorder=2,
                )
            )

        # real part rect (undo inflation) + visual shrink
        ox, oy, ow, oh = _orig_rect_for_display(pp)
        sx, sy, sw, sh = _shrink_rect(ox, oy, ow, oh, vis_margin)
        ax.add_patch(
            Rectangle(
                (sx, sy), sw, sh,
                facecolor="white",
                edgecolor="#333333",
                lw=0.7,
                zorder=3,
            )
        )

        if show_ids or show_dims:
            cx, cy = ox + ow / 2.0, oy + oh / 2.0
            lines = []
            if show_ids:
                lines.append(str(pp.uid))
            if show_dims:
                lines.append(f"{_fmt_dim(ow)}×{_fmt_dim(oh)}")
            ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=7, color="#222222", zorder=4)

    if title:
        ax.set_title(title, fontsize=10)

    # Legend (proxy artists)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#333333", lw=0.7, label="Part outline"),
        Line2D([0], [0], color="#1f77b4", lw=0.8, ls=(0, (2, 2)), label="Packing rect"),
    ]
    if show_trim and board.trim and board.trim > 0:
        legend_items.append(Line2D([0], [0], color="#777777", lw=0.9, ls=(0, (2, 2)), label="Trim boundary"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True)

    _ensure_dir(os.path.dirname(out_path) or ".")
    png, svg = _save_png_and_svg(fig, out_path)
    plt.close(fig)
    return png, svg


def plot_board_toolpath(
        board: Board,
        out_path: str,
        *,
        show_trim: bool = True,
        share_mode: str = "shared",
        cut_mode: str = "trail",
        origin: Point = (0.0, 0.0),
        jitter: float = 0.6,
        jitter_seed: Optional[int] = None,
        do_two_opt: bool = True,
        two_opt_max_iter: int = 1200,
        title: Optional[str] = None,
) -> Tuple[str, str]:
    """Plot toolpath-related visualization (segments + strokes) as PNG+SVG."""

    # 1) Build segments
    segs, L_shared, _ = build_segments_from_board(board, share_mode=share_mode)

    # Optional deterministic jitter for baseline (share_mode == 'none')
    if share_mode == "none" and jitter and jitter > 0:
        import numpy as np

        seed = jitter_seed if jitter_seed is not None else int(board.bid)
        rng = np.random.RandomState(seed)
        jittered: List[Segment] = []
        for (x1, y1, x2, y2, shared) in segs:
            dx = float(rng.uniform(-jitter, jitter))
            dy = float(rng.uniform(-jitter, jitter))
            jittered.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, shared))
        segs = jittered

    # 2) Convert segments into strokes
    L_cut, n_components, n_strokes, stroke_reps, comps, stroke_trails = estimate_cut_and_strokes(
        segs, cut_mode=cut_mode, return_trails=True
    )

    # 3) Order strokes (NN + 2-opt)
    if n_strokes == 0:
        order = []
    else:
        order, _ = nn_tour(stroke_reps, start=origin)
        if do_two_opt and len(order) >= 4:
            order, _ = two_opt(stroke_reps, order, start=origin, max_iter=two_opt_max_iter)

    oriented = _orient_strokes_by_order(stroke_trails, order, origin)

    # 4) Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    # Board outline
    ax.add_patch(Rectangle((0, 0), board.W, board.H, facecolor="none", edgecolor="black", lw=1.0, zorder=0))

    if show_trim and board.trim and board.trim > 0:
        ax.add_patch(
            Rectangle(
                (board.trim, board.trim),
                board.W - 2 * board.trim,
                board.H - 2 * board.trim,
                facecolor="none",
                edgecolor="#777777",
                lw=0.9,
                ls=(0, (2, 2)),
                zorder=0,
            )
        )

    # Cut segments
    # (We draw segments directly, and direction/air moves based on oriented strokes.)
    for (x1, y1, x2, y2, shared) in segs:
        if shared:
            ax.plot([x1, x2], [y1, y2], color="red", lw=2.0, solid_capstyle="butt", zorder=2)
        else:
            ax.plot([x1, x2], [y1, y2], color="#1f77b4", lw=1.1, solid_capstyle="butt", zorder=1)

    # Air moves & stroke direction
    cur = origin
    for s in oriented:
        # Air move
        if math.hypot(s.start[0] - cur[0], s.start[1] - cur[1]) > 1e-6:
            ax.add_patch(
                FancyArrowPatch(
                    cur, s.start,
                    arrowstyle="->",
                    mutation_scale=8,
                    lw=0.8,
                    linestyle=(0, (2, 2)),
                    color="#ff7f0e",
                    zorder=3,
                )
            )
        # Direction arrow for the stroke
        _stroke_direction_arrow(ax, s.trail, color="black", scale=1.0)
        cur = s.end

    # Origin marker
    ax.plot([origin[0]], [origin[1]], marker="s", markersize=6, color="black", zorder=5)

    if title:
        ax.set_title(title, fontsize=10)

    # Legend (proxy)
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="#1f77b4", lw=1.1, label="Cut (Blue)"),
        Line2D([0], [0], color="red", lw=2.0, label="Shared cut (Red)"),
        Line2D([0], [0], color="#ff7f0e", lw=0.8, ls=(0, (2, 2)), label="Air move"),
    ]
    if show_trim and board.trim and board.trim > 0:
        legend_items.append(Line2D([0], [0], color="#777777", lw=0.9, ls=(0, (2, 2)), label="Trim boundary"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, frameon=True)

    # Annotation (optional)
    # We keep it minimal for paper figures, but you can uncomment if needed:
    # ax.text(0.02*board.W, 0.98*board.H, f"L_shared={L_shared:.1f}  L_cut={L_cut:.1f}  strokes={n_strokes}",
    #         ha="left", va="top", fontsize=8)

    _ensure_dir(os.path.dirname(out_path) or ".")
    png, svg = _save_png_and_svg(fig, out_path)
    plt.close(fig)
    return png, svg


def plot_one_seed_outputs(
        boards: List[Board],
        boards_rows: List[dict],
        out_dir: str,
        *,
        title: str | None = None,
        max_boards: int = 6,
        plot_toolpath: bool = True,
        show_clearance: bool = True,
        vis_margin: float = 0.5,
        show_ids: bool = False,
        show_dims: bool = False,
        share_mode: str = "shared",
        cut_mode: str = "trail",
        origin: Point = (0.0, 0.0),
        trim: float = 0.0,
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
) -> None:
    """Export per-board figures and a compact metrics preview for one seed."""
    _ensure_dir(out_dir)

    rows = []
    for i, b in enumerate(boards[:max_boards]):
        stem = os.path.join(out_dir, f"board_{b.bid:03d}")
        plot_board_layout(
            b,
            stem + "_layout.png",
            show_trim=True,
            show_clearance=show_clearance,
            vis_margin=vis_margin,
            show_ids=show_ids,
            show_dims=show_dims,
            title=None,
        )

        if plot_toolpath:
            plot_board_toolpath(
                b,
                stem + "_toolpath.png",
                show_trim=True,
                share_mode=share_mode,
                cut_mode=cut_mode,
                origin=origin,
                title=None,
            )

        # Quick metrics (approx) for preview
        segs, L_shared, _ = build_segments_from_board(b, share_mode=share_mode)
        L_cut, _, n_strokes, _, _, stroke_trails = estimate_cut_and_strokes(segs, cut_mode=cut_mode, return_trails=True)
        order, _ = nn_tour([tr[0] for tr in stroke_trails] if stroke_trails else [], start=origin)
        oriented = _orient_strokes_by_order(stroke_trails, order, origin) if stroke_trails else []
        L_air = _compute_air_length(oriented, origin, close_to_origin=False)
        util = 0.0
        try:
            util = sum(pp.w0 * pp.h0 for pp in b.placed) / (b.W * b.H)
        except Exception:
            pass

        rows.append((b.bid, len(b.placed), util, L_cut, L_air))

    _plot_boards_metrics_preview(rows, os.path.join(out_dir, "boards_metrics_preview.png"))


def _plot_boards_metrics_preview(rows: Sequence[Tuple[int, int, float, float, float]], out_path: str) -> Tuple[str, str]:
    """Create a compact preview as a single figure (PNG+SVG)."""
    fig, ax = plt.subplots(figsize=(10, 0.45 * max(6, len(rows)) + 1.0))
    ax.axis("off")

    # Title
    ax.text(0.0, 1.0, "Boards metrics preview", fontsize=11, ha="left", va="top", transform=ax.transAxes)

    # Table-like text
    y = 0.92
    line_h = 0.06
    header = "{:<10} {:<10} {:<10} {:<12} {:<12}".format("Board", "Parts", "Util", "L_cut", "L_air")
    ax.text(0.0, y, header, family="monospace", fontsize=10, ha="left", va="top", transform=ax.transAxes)
    y -= line_h
    ax.text(0.0, y, "-" * len(header), family="monospace", fontsize=10, ha="left", va="top", transform=ax.transAxes)
    y -= line_h

    for bid, n_parts, util, L_cut, L_air in rows:
        line = "{:<10} {:<10} {:<10} {:<12} {:<12}".format(
            f"#{bid:03d}",
            str(n_parts),
            f"{util * 100:.2f}%",
            f"{L_cut / 1000.0:.2f}m",
            f"{L_air / 1000.0:.2f}m",
        )
        ax.text(0.0, y, line, family="monospace", fontsize=10, ha="left", va="top", transform=ax.transAxes)
        y -= line_h

    _ensure_dir(os.path.dirname(out_path) or ".")
    png, svg = _save_png_and_svg(fig, out_path)
    plt.close(fig)
    return png, svg
