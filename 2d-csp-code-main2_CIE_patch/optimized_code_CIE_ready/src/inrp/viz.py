# src/inrp/viz.py
from __future__ import annotations

import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)

# --- Drawing constants (paper-friendly, thin strokes) ---
COLOR_CUT = "#222222"        # normal cut
COLOR_SHARED = "#D32F2F"     # shared cut
COLOR_PART = "#B0B0B0"       # part outline (nominal)
COLOR_TRIM = "#888888"       # trim box
COLOR_AIR = "#1E88E5"        # air moves (for toolpath-only view)
COLOR_TEXT = "#00695C"       # stroke index

LW_PART = 0.45
LW_CUT = 0.85
LW_SHARED = 1.6
LW_AIR = 0.7


def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.lines as mlines
        return plt, patches, mlines
    except Exception as e:
        logger.warning("[PLOT] matplotlib missing: %s", e)
        return None, None, None


def _save_png_and_svg(fig, out_path: Path, *, dpi: int = 300) -> None:
    """Save PNG and SVG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_nominal_rect(pp):
    """Return nominal (inner) rectangle for visualization.

    pp.rect is the inflated rectangle (gap-inflated).
    pp.w0/pp.h0 store original dimensions.
    """
    r = pp.rect
    w0 = pp.h0 if getattr(pp, "rotated", False) else pp.w0
    h0 = pp.w0 if getattr(pp, "rotated", False) else pp.h0
    dx = max((r.w - w0) * 0.5, 0.0)
    dy = max((r.h - h0) * 0.5, 0.0)
    return (r.x + dx, r.y + dy, w0, h0)


# -----------------------------------------------------------------------------
# Path reconstruction for toolpath figure
# -----------------------------------------------------------------------------

def _reconstruct_ordered_path(segs, *, start=(0.0, 0.0), sa_iters: int = 2000):
    """Reconstruct an ordered sequence of strokes for visualization.

    We rebuild connected components from the segment graph and then order the
    resulting strokes by a TSP-like tour. We use simulated annealing (same
    family as routing.air_length_by_strokes) to keep consistency between
    visualization and metric mode.

    Returns:
      ordered_strokes: List[List[Point]]
      air_moves: List[Tuple[Point, Point]]
    """
    from .routing import build_graph_with_edges, simulated_annealing_tour

    adj = build_graph_with_edges(segs)
    if not adj:
        return [], []

    # connected components
    nodes = list(adj.keys())
    seen = set()
    comps = []
    for v in nodes:
        if v in seen:
            continue
        stack = [v]
        seen.add(v)
        comp = {v}
        while stack:
            x = stack.pop()
            for y, _, _ in adj.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
                    comp.add(y)
        comps.append(comp)

    # extract strokes as trails in each component
    all_strokes = []
    used_edges = set()

    for comp in comps:
        deg = {u: len(adj.get(u, [])) for u in comp}

        def trace_trail(u, v, eid0):
            path = [u, v]
            cur_eid = eid0
            cur_node = v
            while True:
                if deg.get(cur_node, 0) != 2:
                    break
                nxt = None
                for w, _, eid in adj.get(cur_node, []):
                    if eid != cur_eid and eid not in used_edges:
                        nxt = (w, eid)
                        break
                if not nxt:
                    break
                w, eid = nxt
                used_edges.add(eid)
                path.append(w)
                cur_eid = eid
                cur_node = w
                if cur_node == u:
                    break
            return path

        # start from junctions / endpoints
        for u in comp:
            if deg.get(u, 0) == 2:
                continue
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                all_strokes.append(trace_trail(u, v, eid))

        # cleanup cycles
        for u in comp:
            for v, _, eid in adj.get(u, []):
                if eid in used_edges:
                    continue
                used_edges.add(eid)
                all_strokes.append(trace_trail(u, v, eid))

    if not all_strokes:
        return [], []

    stroke_starts = [s[0] for s in all_strokes]
    order_indices, _ = simulated_annealing_tour(stroke_starts, start=start, iters=int(sa_iters))
    ordered_strokes = [all_strokes[i] for i in order_indices]

    air_moves = []
    current_pos = start
    for stroke in ordered_strokes:
        s0 = stroke[0]
        if _dist(current_pos, s0) > 0.1:
            air_moves.append((current_pos, s0))
        current_pos = stroke[-1]

    return ordered_strokes, air_moves


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def plot_layout_with_toolpath(
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
    """Single “paper-ready” overlay figure:

    - nominal part outlines (light gray)
    - cutting segments (black)
    - shared segments (red)
    - trim boundary (dashed)

    This is usually the key figure reviewers want.
    """
    plt, patches, mlines = _safe_import_matplotlib()
    if plt is None:
        return

    from .routing import build_segments_from_board

    segs, _L_shared, _n_tabs = build_segments_from_board(
        board,
        share_mode,
        tab_enable,
        tab_per_part,
        tab_len,
        tab_corner_clear,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        nd=nd_coord,
    )

    fig = plt.figure(figsize=(12, 12 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # board outer
    ax.add_patch(patches.Rectangle((0, 0), board.W, board.H, fill=False, linewidth=0.9, color="#A0A0A0"))

    # trim
    if trim and trim > 0:
        ax.add_patch(
            patches.Rectangle(
                (trim, trim),
                max(0.0, board.W - 2 * trim),
                max(0.0, board.H - 2 * trim),
                fill=False,
                linestyle=":",
                linewidth=0.8,
                color=COLOR_TRIM,
            )
        )

    # nominal outlines
    for pp in board.placed:
        x, y, w, h = _compute_nominal_rect(pp)
        ax.add_patch(
            patches.Rectangle((x, y), w, h, fill=False, edgecolor=COLOR_PART, linewidth=LW_PART)
        )

    # cut segments (shared highlighted)
    for s in segs:
        col = COLOR_SHARED if getattr(s, "shared", False) else COLOR_CUT
        lw = LW_SHARED if getattr(s, "shared", False) else LW_CUT
        z = 3 if getattr(s, "shared", False) else 2
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color=col, linewidth=lw, solid_capstyle="round", zorder=z)

    handles = [
        mlines.Line2D([], [], color=COLOR_PART, lw=LW_PART, label="Part outline"),
        mlines.Line2D([], [], color=COLOR_CUT, lw=LW_CUT, label="Cut"),
        mlines.Line2D([], [], color=COLOR_SHARED, lw=LW_SHARED, label="Shared cut"),
    ]
    if trim and trim > 0:
        handles.append(mlines.Line2D([], [], color=COLOR_TRIM, lw=0.8, ls=":", label="Trim boundary"))
    ax.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.9)

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
    sa_iters: int = 2000,
) -> None:
    """Visualizes the cutting path with stroke order.

    This figure is more “algorithmic”; for the key paper figure use
    plot_layout_with_toolpath().
    """
    plt, patches, mlines = _safe_import_matplotlib()
    if plt is None:
        return

    from .routing import build_segments_from_board

    segs, _, _ = build_segments_from_board(
        board,
        share_mode,
        tab_enable,
        tab_per_part,
        tab_len,
        tab_corner_clear,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        nd=nd_coord,
    )

    start = (float(trim), float(trim)) if trim and trim > 0 else (0.0, 0.0)
    ordered_strokes, air_moves = _reconstruct_ordered_path(segs, start=start, sa_iters=int(sa_iters))

    fig = plt.figure(figsize=(14, 14 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # board outer
    ax.add_patch(patches.Rectangle((0, 0), board.W, board.H, fill=False, linewidth=0.9, color="#A0A0A0"))

    # trim
    if trim and trim > 0:
        ax.add_patch(
            patches.Rectangle(
                (trim, trim),
                max(0.0, board.W - 2 * trim),
                max(0.0, board.H - 2 * trim),
                fill=False,
                linestyle=":",
                linewidth=0.8,
                color=COLOR_TRIM,
            )
        )

    # air moves
    for p1, p2 in air_moves:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle=":", linewidth=LW_AIR, color=COLOR_AIR, alpha=0.45, zorder=1)

    # geometry layer: shared vs normal
    for s in segs:
        col = COLOR_SHARED if getattr(s, "shared", False) else COLOR_CUT
        lw = LW_SHARED if getattr(s, "shared", False) else LW_CUT
        z = 3 if getattr(s, "shared", False) else 2
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color=col, linewidth=lw, solid_capstyle="round", zorder=z)

    # stroke indices (label at each stroke start)
    for idx, stroke in enumerate(ordered_strokes):
        p0 = stroke[0]
        fsize = 8 if len(ordered_strokes) < 50 else 5
        ax.text(p0[0], p0[1], str(idx + 1), color=COLOR_TEXT, fontsize=fsize, fontweight="bold", ha="right", va="bottom", zorder=10)

    # home
    ax.scatter(start[0], start[1], s=30, marker="s", color="black", zorder=20)

    handles = [
        mlines.Line2D([], [], color=COLOR_CUT, lw=LW_CUT, label="Cut"),
        mlines.Line2D([], [], color=COLOR_SHARED, lw=LW_SHARED, label="Shared cut"),
        mlines.Line2D([], [], color=COLOR_AIR, lw=LW_AIR, ls=":", label="Air move"),
    ]
    if trim and trim > 0:
        handles.append(mlines.Line2D([], [], color=COLOR_TRIM, lw=0.8, ls=":", label="Trim boundary"))
    ax.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.9)

    _save_png_and_svg(fig, out_path, dpi=300)
    plt.close(fig)


def plot_board_layout(
    board,
    out_path: Path,
    title: str,
    *,
    trim: float = 0.0,
    show_clearance: bool = True,
    show_ids: bool = False,
) -> None:
    """Save layout figure."""
    plt, patches, _ = _safe_import_matplotlib()
    if plt is None:
        return

    fig = plt.figure(figsize=(10, 10 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), board.W, board.H, fill=False, lw=1.2, color="#666"))
    if trim and trim > 0:
        ax.add_patch(
            patches.Rectangle(
                (trim, trim),
                max(0.0, board.W - 2 * trim),
                max(0.0, board.H - 2 * trim),
                fill=False,
                ls=":",
                lw=0.6,
                color=COLOR_TRIM,
            )
        )

    for pp in board.placed:
        # Clearance (inflated)
        if show_clearance:
            r = pp.rect
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=True, color="#F2F2F2"))
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=False, ls="--", lw=0.35, color="#D0D0D0"))

        # Nominal part (filled)
        x, y, w, h = _compute_nominal_rect(pp)
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=True, facecolor="#DDDDDD", edgecolor="#333", lw=0.9))
        if show_ids:
            ax.text(x + w / 2, y + h / 2, str(pp.uid), ha="center", va="center", fontsize=6, color="#444")

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
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for b in boards[:max_boards]:
        # 1) layout
        plot_board_layout(
            b,
            out_dir / f"board_{b.bid:03d}_layout.png",
            title=f"{title} | Layout {b.bid}",
            trim=trim,
            show_clearance=True,
            show_ids=show_ids,
        )

        # 2) overlay (key figure)
        plot_layout_with_toolpath(
            b,
            out_dir / f"board_{b.bid:03d}_overlay.png",
            title=f"{title} | Overlay {b.bid}",
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

        # 3) toolpath (optional)
        if plot_toolpath:
            plot_board_toolpath(
                b,
                out_dir / f"board_{b.bid:03d}_toolpath.png",
                title=f"{title} | Path {b.bid}",
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
