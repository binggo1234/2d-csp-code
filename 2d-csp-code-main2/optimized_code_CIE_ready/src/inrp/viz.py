# viz.py
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import math

# CIE Color Palette (Paper-friendly)
COLOR_CUT = "#202020"       # Dark Grey/Black for normal cuts
COLOR_SHARED = "#D32F2F"    # Red for shared cuts
COLOR_AIR = "#1976D2"       # Blue for air moves
COLOR_PIERCE = "#388E3C"    # Green for pierce points (entry)
WIDTH_CUT = 1.0
WIDTH_SHARED = 2.5          # Thicker to emphasize optimization
WIDTH_AIR = 0.8


def _save_png_and_svg(fig, out_path: Path, *, dpi: int = 300) -> None:
    """Save the same figure into PNG + SVG.

    CIE-friendly: SVG is vector for the manuscript; PNG is quick preview.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    # SVG (vector)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)


def _compute_inner_cut_rect(pp) -> tuple[float, float, float, float]:
    """Infer the *nominal part contour* inside an inflated placement rect."""
    r = pp.rect
    w0 = pp.h0 if getattr(pp, "rotated", False) else pp.w0
    h0 = pp.w0 if getattr(pp, "rotated", False) else pp.h0
    dx = max((r.w - w0) * 0.5, 0.0)
    dy = max((r.h - h0) * 0.5, 0.0)
    return (r.x + dx, r.y + dy, w0, h0)


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.lines as mlines
        return plt, Rectangle, mlines
    except Exception as e:
        logger.warning("[PLOT] skip (matplotlib not available): %s", e)
        return None, None, None


def _simulate_air_path_points(comps, start=(0.0, 0.0)):
    """
    Replicate the Greedy Endpoint Strategy from routing.py strictly for VISUALIZATION.
    Returns a list of line segments [(x1,y1, x2,y2), ...] representing air moves.
    """
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    if not comps:
        return []

    # Deep copy to avoid mutating original data
    unvisited = [list(c) for c in comps]
    current_pos = start
    air_lines = []

    while unvisited:
        best_comp_idx = -1
        best_node_idx = -1
        min_dist = 1e100

        # Find closest entry point globally
        for i, nodes in enumerate(unvisited):
            for j, p in enumerate(nodes):
                d = _dist(current_pos, p)
                if d < min_dist:
                    min_dist = d
                    best_comp_idx = i
                    best_node_idx = j

        if best_comp_idx == -1:
            break

        # Record air move
        target_pos = unvisited[best_comp_idx][best_node_idx]
        if min_dist > 1e-6:
            air_lines.append((current_pos, target_pos))

        # Move "machine" to target
        current_pos = target_pos
        unvisited.pop(best_comp_idx)

    return air_lines


def plot_board_layout(
        board,
        out_path: Path,
        title: str,
        *,
        trim: float = 0.0,
        show_clearance: bool = True,
        show_ids: bool = False,
) -> None:
    """Save one board layout figure (Packing View)."""
    plt, Rectangle, _ = _safe_import_matplotlib()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 10 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    # Clean style for paper
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlim(-50, board.W + 50)
    ax.set_ylim(-50, board.H + 50)

    # board boundary
    ax.add_patch(Rectangle((0, 0), board.W, board.H, fill=False, linewidth=1.5, color="#555555", label="Board"))

    # usable boundary
    if trim and trim > 0:
        ax.add_patch(Rectangle(
            (trim, trim), board.W - 2 * trim, board.H - 2 * trim,
            fill=False, linestyle=":", linewidth=0.8, color="#888888"
        ))

    # Parts
    for pp in board.placed:
        r = pp.rect
        if show_clearance:
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=True, color="#F5F5F5", linewidth=0))
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, linewidth=0.5, linestyle="--", color="#CCCCCC"))

        x, y, w, h = _compute_inner_cut_rect(pp)
        ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor="#E0E0E0", edgecolor="black", linewidth=1.2, alpha=0.8))

        if show_ids:
            ax.text(x + w * 0.5, y + h * 0.5, str(pp.uid), ha="center", va="center", fontsize=7, fontweight='bold', color="#333")

    _save_png_and_svg(fig, out_path, dpi=300)
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
) -> None:
    """Save a detailed toolpath figure (Routing View).

    Now visualizes:
    1. Cut segments (Black)
    2. Shared segments (Red & Thick)
    3. Air moves (Blue Dashed) - using Greedy Endpoint simulation
    4. Pierce points (Green dots)
    """
    plt, Rectangle, mlines = _safe_import_matplotlib()
    if plt is None:
        return

    from .routing import build_segments_from_board, estimate_cut_and_strokes

    # 1. Build Geometry
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

    # 2. Get Components for Air Move Simulation
    # We use 'trail' mode just to get components quickly without heavy CPP calcs if not needed
    _, _, _, _, comps = estimate_cut_and_strokes(segs, cut_mode="trail")

    # 3. Simulate Air Path (Greedy Endpoint) for visualization
    air_lines = _simulate_air_path_points(comps, start=(0.0, 0.0))

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 12 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off") # Turn off axis for clean look

    # Draw Board
    ax.add_patch(Rectangle((0, 0), board.W, board.H, fill=False, linewidth=1.0, color="#999999"))

    # A. Draw Air Moves (Layer 1 - Bottom)
    for p1, p2 in air_lines:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                linestyle="--", linewidth=WIDTH_AIR, color=COLOR_AIR, alpha=0.6, zorder=1)

    # B. Draw Cuts (Layer 2)
    # We collect pierce points (starts of components/strokes)
    # For visualization, we just mark the start of every component as a pierce point candidate
    # (Approximate, but visually sufficient for high-level paper plots)
    pierce_points = []

    for s in segs:
        (x1, y1), (x2, y2) = s.a, s.b
        is_shared = s.shared

        lw = WIDTH_SHARED if is_shared else WIDTH_CUT
        col = COLOR_SHARED if is_shared else COLOR_CUT
        ord = 3 if is_shared else 2

        ax.plot([x1, x2], [y1, y2], linewidth=lw, color=col, zorder=ord, solid_capstyle='round')

    # C. Draw Pierce Points (Layer 4 - Top)
    # From air moves, we know exactly where the head lands.
    for _, p_dest in air_lines:
        ax.scatter(p_dest[0], p_dest[1], s=15, color=COLOR_PIERCE, zorder=4, marker='o', edgecolors='none')

    # Add Start Point
    ax.scatter(0, 0, s=40, color="black", marker="s", zorder=5, label="Home")

    # Legend (Paper style)
    handles = [
        mlines.Line2D([], [], color=COLOR_CUT, lw=WIDTH_CUT, label='Cut Profile'),
        mlines.Line2D([], [], color=COLOR_SHARED, lw=WIDTH_SHARED, label='Shared Cut'),
        mlines.Line2D([], [], color=COLOR_AIR, lw=WIDTH_AIR, linestyle="--", label='Air Move (Greedy)'),
        mlines.Line2D([], [], color=COLOR_PIERCE, marker='o', linestyle='None', markersize=5, label='Pierce Point'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=True, fontsize='small', framealpha=0.9)

    _save_png_and_svg(fig, out_path, dpi=300)
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
        # toolpath-related knobs (keep consistent with metrics)
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
) -> None:
    """Plot outputs for the first seed (paper-friendly)."""

    out_dir.mkdir(parents=True, exist_ok=True)

    for b in boards[:max_boards]:
        # 1. Layout Plot (Material Utilization View)
        plot_board_layout(
            b,
            out_dir / f"board_{b.bid:03d}_layout.png",
            title=f"{title} | Board {b.bid}",
            trim=trim,
            show_clearance=True,
            show_ids=show_ids,
            )

        # 2. Toolpath Plot (Routing/Path View) - The most important for CIE
        if plot_toolpath:
            plot_board_toolpath(
                b,
                out_dir / f"board_{b.bid:03d}_toolpath.png",
                title=f"{title} | Board {b.bid} Path",
                share_mode=share_mode,
                tab_enable=tab_enable,
                tab_per_part=tab_per_part,
                tab_len=tab_len,
                tab_corner_clear=tab_corner_clear,
                line_snap_eps=line_snap_eps,
                min_shared_len=min_shared_len,
                nd_coord=nd_coord,
                trim=trim,
                )

    # Metrics Summary Image
    plt, _Rectangle, _ = _safe_import_matplotlib()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Format nicely as a table-like text
    lines = ["Experiment Summary: " + title, "-"*60]
    header = f"{'Board':<6} {'Parts':<6} {'Util(%)':<10} {'Shared(mm)':<12} {'Air(mm)':<10} {'Lifts':<6}"
    lines.append(header)
    lines.append("-" * len(header))

    for row in boards_rows:
        lines.append(
            f"{int(row['board']):<6d} "
            f"{int(row['n_parts']):<6d} "
            f"{row['U']*100:<10.2f} "
            f"{row['L_shared']:<12.1f} "
            f"{row['L_air']:<10.1f} "
            f"{int(row['N_lift']):<6d}"
        )

    ax.text(0.01, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    _save_png_and_svg(fig, out_dir / "boards_metrics_preview.png", dpi=200)
    plt.close(fig)