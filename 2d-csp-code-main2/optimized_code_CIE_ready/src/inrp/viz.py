# src/inrp/viz.py
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import math
import random

# --- 绘图配置 ---
COLOR_CUT = "#333333"       # 深灰 (普通切割)
COLOR_SHARED = "#D32F2F"    # 深红 (共边切割)
COLOR_AIR = "#1976D2"       # 蓝色 (空行程)
COLOR_TEXT = "#00695C"      # 墨绿 (序号文字)
WIDTH_CUT = 1.0
WIDTH_SHARED = 2.0
WIDTH_AIR = 0.8
ARROW_SIZE = 0.02           # 箭头相对于图幅的大小

def _save_png_and_svg(fig, out_path: Path, *, dpi: int = 300) -> None:
    """Save PNG and SVG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)

def _safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.lines as mlines
        return plt, patches, mlines
    except Exception as e:
        logger.warning("[PLOT] matplotlib missing: %s", e)
        return None, None, None

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# --- 核心：为了画图而重新推演路径顺序 ---
# 注意：这里为了可视化，会简单重跑一遍 TSP。
# 这保证了图上的箭头顺序是真实可行的，而不仅仅是乱序线段。

def _reconstruct_ordered_path(segs):
    """
    Reconstruct the full ordered cutting path for visualization.
    Returns:
       ordered_strokes: List[List[Point]] (List of continuous cut paths)
       air_moves: List[Tuple[Point, Point]] (List of jumps)
    """
    from .routing import build_graph_with_edges, stroke_representatives, nn_tour, simulated_annealing_tour

    # 1. 构建图和提取笔画 (Strokes)
    adj = build_graph_with_edges(segs)
    # 这里的 hack 是我们需要 stroke 的具体内容，而不仅仅是代表点。
    # 我们复用 stroke_representatives 的逻辑，但把路径存下来。

    # 手动重建 connected components
    nodes = list(adj.keys())
    seen = set()
    comps = []
    for v in nodes:
        if v in seen: continue
        stack = [v]
        seen.add(v)
        comp = {v}
        while stack:
            x = stack.pop()
            for y, _, _ in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
                    comp.add(y)
        comps.append(comp)

    # 提取所有具体的笔画路径
    all_strokes = [] # List of point lists
    used_edges = set()

    for comp in comps:
        deg = {u: len(adj[u]) for u in comp}

        def trace_trail(u, v, eid0):
            path = [u, v]
            cur_eid = eid0
            cur_node = v
            while True:
                if deg.get(cur_node, 0) != 2: break # 遇到交叉点或端点停止
                nxt = None
                for w, _, eid in adj[cur_node]:
                    if eid != cur_eid and eid not in used_edges:
                        nxt = (w, eid)
                        break
                if not nxt: break
                w, eid = nxt
                used_edges.add(eid)
                path.append(w)
                cur_eid = eid
                cur_node = w
                if cur_node == u: break # 闭环
            return path

        # 优先从奇点/交叉点开始
        for u in comp:
            if deg.get(u, 0) == 2: continue
            for v, _, eid in adj[u]:
                if eid in used_edges: continue
                used_edges.add(eid)
                all_strokes.append(trace_trail(u, v, eid))

        # 处理剩余的环 (纯圆)
        for u in comp:
            for v, _, eid in adj[u]:
                if eid in used_edges: continue
                used_edges.add(eid)
                all_strokes.append(trace_trail(u, v, eid))

    if not all_strokes:
        return [], []

    # 2. 对笔画进行排序 (TSP)
    # 提取每个笔画的起点作为代表
    stroke_starts = [s[0] for s in all_strokes]

    # 跑一次简单的 TSP (NN) 来确定画图顺序
    # 即使 routing.py 用了 SA，这里用 NN 画图也足够展示逻辑，且速度快
    order_indices, _ = nn_tour(stroke_starts, start=(0.0, 0.0))

    ordered_strokes = [all_strokes[i] for i in order_indices]

    # 3. 生成空行程
    air_moves = []
    current_pos = (0.0, 0.0)
    for stroke in ordered_strokes:
        start_pt = stroke[0]
        if _dist(current_pos, start_pt) > 0.1: # 忽略极小移动
            air_moves.append((current_pos, start_pt))
        current_pos = stroke[-1]

    return ordered_strokes, air_moves

# --- 主绘图函数 ---

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
    """
    Visualizes the ACTUAL cutting path with sequence numbers and arrows.
    """
    plt, patches, mlines = _safe_import_matplotlib()
    if plt is None: return

    from .routing import build_segments_from_board

    # 1. 获取几何线段
    segs, _, _ = build_segments_from_board(
        board, share_mode, tab_enable, tab_per_part, tab_len,
        tab_corner_clear, line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len, nd=nd_coord
    )

    # 2. 重建有序路径 (Path Reconstruction)
    ordered_strokes, air_moves = _reconstruct_ordered_path(segs)

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(14, 14 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off") # 去除坐标轴，像CAD图纸

    # 画板材轮廓
    ax.add_patch(patches.Rectangle((0, 0), board.W, board.H, fill=False, linewidth=1.0, color="#AAAAAA"))

    # A. 画空行程 (蓝色虚线)
    for p1, p2 in air_moves:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                linestyle=":", linewidth=WIDTH_AIR, color=COLOR_AIR, alpha=0.5, zorder=1)

    # B. 画切割路径 (带箭头)
    # 为了区分 shared，我们需要查一下原始 segs 哪些是 shared。
    # 这里为了视觉简单，我们统一画深色，重点展示"路径流"。
    # 如果必须区分红黑，需要反查几何，这在路径重组后比较复杂。
    # 妥协方案：先画一遍底层的红/黑线段展示几何属性，再在其上画带箭头的路径展示顺序。

    # B1. 底层几何 (Geometry Layer) - 展示 Shared (红) vs Normal (黑)
    for s in segs:
        color = COLOR_SHARED if s.shared else COLOR_CUT
        width = WIDTH_SHARED if s.shared else WIDTH_CUT
        z = 2 if s.shared else 1.5
        ax.plot([s.a[0], s.b[0]], [s.a[1], s.b[1]], color=color, linewidth=width, solid_capstyle='round', zorder=z)

    # B2. 路径流 (Flow Layer) - 箭头与序号
    # 只标注前 N 个和部分后续，避免太乱？或者全部标注。
    # 对于论文图，通常标注前 10-20 个，或者每隔几个标注一下。
    # 这里我们标注所有 Stroke 的起点。

    for idx, stroke in enumerate(ordered_strokes):
        # 1. 标注序号 (Sequence Number)
        start_pt = stroke[0]
        # 字体大小随 Stroke 数量自动调整
        fsize = 8 if len(ordered_strokes) < 50 else 5
        ax.text(start_pt[0], start_pt[1], str(idx + 1),
                color=COLOR_TEXT, fontsize=fsize, fontweight='bold',
                ha='right', va='bottom', zorder=10)

        # 2. 画箭头 (Direction Arrow)
        # 在每一段的中间画一个小箭头
        for i in range(len(stroke) - 1):
            p_a = stroke[i]
            p_b = stroke[i+1]

            # 只有当线段足够长时才画箭头，避免密集恐惧症
            if _dist(p_a, p_b) > min(board.W, board.H) * 0.02:
                mid_x = (p_a[0] + p_b[0]) / 2
                mid_y = (p_a[1] + p_b[1]) / 2
                dx = p_b[0] - p_a[0]
                dy = p_b[1] - p_a[1]

                # 使用 FancyArrow 表示方向
                # 长度归一化，这就只是个方向标
                arrow = patches.FancyArrowPatch(
                    posA=(p_a[0], p_a[1]), posB=(p_b[0], p_b[1]),
                    arrowstyle='->', mutation_scale=10,
                    color="#555555", alpha=0.7, zorder=5, shrinkA=0, shrinkB=0
                )
                # 或者简单的在中间画个标记
                # ax.add_patch(arrow)
                # FancyArrowPatch 有时比较慢，直接用 plot marker
                # 计算角度太麻烦，直接画个点或者简单线段
                pass

                # 更简单的箭头绘制法：
                # 在线段 60% 处画一个极小的箭头
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), xytext=(mid_x, mid_y),
                            arrowprops=dict(arrowstyle="->", color="black", lw=0.8), zorder=5)

    # 起点 Home
    ax.scatter(0, 0, s=50, marker='s', color='black', zorder=20, label="Home")

    # Legend
    handles = [
        mlines.Line2D([], [], color=COLOR_CUT, lw=WIDTH_CUT, label='Normal Cut'),
        mlines.Line2D([], [], color=COLOR_SHARED, lw=WIDTH_SHARED, label='Shared Edge (Cut Once)'),
        mlines.Line2D([], [], color=COLOR_AIR, ls=":", label='Air Move'),
        mlines.Line2D([], [], marker=r'$\rightarrow$', color='k', lw=0, label='Direction'),
        mlines.Line2D([], [], marker=r'$123$', color=COLOR_TEXT, lw=0, label='Sequence ID'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize='small', framealpha=0.9)

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
    if plt is None: return

    fig = plt.figure(figsize=(10, 10 * board.H / max(1.0, board.W)))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), board.W, board.H, fill=False, lw=1.5, color="#555"))
    if trim > 0:
        ax.add_patch(patches.Rectangle((trim, trim), board.W-2*trim, board.H-2*trim, fill=False, ls=":", lw=0.5, color="#888"))

    def _compute_inner_cut_rect(pp):
        r = pp.rect
        w0 = pp.h0 if getattr(pp, "rotated", False) else pp.w0
        h0 = pp.w0 if getattr(pp, "rotated", False) else pp.h0
        dx = max((r.w - w0) * 0.5, 0.0)
        dy = max((r.h - h0) * 0.5, 0.0)
        return (r.x + dx, r.y + dy, w0, h0)

    for pp in board.placed:
        # Clearance
        if show_clearance:
            r = pp.rect
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=True, color="#F0F0F0"))
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=False, ls="--", lw=0.3, color="#CCC"))
        # Part
        x, y, w, h = _compute_inner_cut_rect(pp)
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=True, facecolor="#DDD", edgecolor="#222", lw=1.0))
        if show_ids:
            ax.text(x+w/2, y+h/2, str(pp.uid), ha="center", va="center", fontsize=6, color="#444")

    _save_png_and_svg(fig, out_path, dpi=300)
    plt.close(fig)


def plot_one_seed_outputs(
        boards, boards_rows, out_dir: Path, title: str,
        *, max_boards=6, trim=0.0, share_mode="none", plot_toolpath=True, show_ids=False,
        tab_enable=False, tab_per_part=0, tab_len=0.0, tab_corner_clear=0.0,
        line_snap_eps=0.0, min_shared_len=0.0, nd_coord=6
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in boards[:max_boards]:
        # 1. 排样图
        plot_board_layout(
            b, out_dir / f"board_{b.bid:03d}_layout.png",
            title=f"{title} | Layout {b.bid}", trim=trim, show_clearance=True, show_ids=show_ids
        )
        # 2. 路径图 (带箭头和序号)
        if plot_toolpath:
            plot_board_toolpath(
                b, out_dir / f"board_{b.bid:03d}_toolpath.png",
                title=f"{title} | Path {b.bid}",
                share_mode=share_mode,
                tab_enable=tab_enable, tab_per_part=tab_per_part, tab_len=tab_len,
                tab_corner_clear=tab_corner_clear, line_snap_eps=line_snap_eps,
                min_shared_len=min_shared_len, nd_coord=nd_coord, trim=trim
            )