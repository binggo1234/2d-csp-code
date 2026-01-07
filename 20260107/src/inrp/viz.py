from __future__ import annotations

"""INRP 可视化：带序号的工程风刀路（论文最终版）。

特性：
1. 黑/红/灰极简工程风格。
2. 刀序序号清晰展示（1, 2, 3...）。
3. 空行程使用贝塞尔曲线箭头减少视觉干扰。
4. 支持 4 宫格对比图生成。
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # 服务器/批处理环境安全
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from .packer import Board
from .routing import (
    build_segments_from_board,
    estimate_cut_and_strokes,
)

Point = Tuple[float, float]

# --- 极简工程配色 ---
STYLE = {
    "cut": "black",            # 普通切割线（实线）
    "shared": "#D62728",       # 共边切割线（红，核心亮点）
    "air": "#B0B0B0",          # 空行程（灰虚线）
    "start_marker": "#2ca02c", # 起刀点（绿）
    "seq_bg": "black",         # 序号背景
    "seq_fg": "white",         # 序号文字
}


def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def _save_png_and_svg(fig, out_path: str, dpi: int = 300) -> Tuple[str, str]:
    """同时导出 PNG 与 SVG，方便论文/报告复用。"""
    root, ext = os.path.splitext(str(out_path))
    if ext.lower() not in {".png", ".svg"}:
        out_path = str(out_path) + ".png"
    png_path = os.path.splitext(str(out_path))[0] + ".png"
    svg_path = os.path.splitext(str(out_path))[0] + ".svg"
    out_dir = os.path.dirname(os.path.abspath(png_path))
    _ensure_dir(out_dir or ".")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    return png_path, svg_path


@dataclass
class _OrientedStroke:
    trail: List[Point]
    original_idx: int

    @property
    def start(self) -> Point:
        return self.trail[0]

    @property
    def end(self) -> Point:
        return self.trail[-1]

    def reversed(self) -> "_OrientedStroke":
        return _OrientedStroke(list(reversed(self.trail)), self.original_idx)


def _orient_strokes_by_order(
    stroke_trails: List[List[Point]],
    order: List[int],
    origin: Point,
) -> List[_OrientedStroke]:
    """贪心翻转路径方向，让下一刀起点更靠近当前位置，减少空行程。"""
    oriented: List[_OrientedStroke] = []
    cur = origin
    for idx in order:
        tr = stroke_trails[idx]
        s = _OrientedStroke(tr, idx)
        d_fwd = math.hypot(s.start[0] - cur[0], s.start[1] - cur[1])
        d_rev = math.hypot(s.end[0] - cur[0], s.end[1] - cur[1])
        if d_rev < d_fwd:
            s = s.reversed()
        oriented.append(s)
        cur = s.end
    return oriented


# ---------------- 核心绘图：单板刀路 ----------------

def plot_board_toolpath(
    board: Board,
    out_path: Optional[str] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    *,
    share_mode: str = "shared",  # 'none' 基线，'union' 共边合并
    cut_mode: str = "trail",
    origin: Point = (0.0, 0.0),
    show_sequence: bool = True,  # 是否显示序号球
    show_trim: bool = True,
    tab_enable: bool = False,
    tab_per_part: int = 0,
    tab_len: float = 0.0,
    tab_corner_clear: float = 0.0,
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
    trim: float = 0.0,
    use_curved_air: Optional[bool] = None,  # 兼容旧接口，默认使用曲线
) -> Optional[Tuple[str, str]]:
    """
    绘制刀路：
    - 黑线：普通切割
    - 红线：共边切割
    - 灰虚线贝塞尔箭头：空行程
    - 黑底白字序号：切割顺序
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        created_fig = True

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    ax.add_patch(Rectangle((0, 0), board.W, board.H, facecolor="none", edgecolor="black", lw=0.6))
    trim_val = trim if trim and trim > 0 else float(getattr(board, "trim", 0.0) or 0.0)
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

    _, _, _, _, oriented_trails, _ = estimate_cut_and_strokes(
        segs, cut_mode=cut_mode, return_trails=True, origin=origin, nd=nd_coord
    )
    oriented: List[_OrientedStroke] = []
    for idx, tr in enumerate(oriented_trails or []):
        if tr:
            oriented.append(_OrientedStroke(tr, idx))
    use_curved = True if use_curved_air is None else bool(use_curved_air)

    # 底层几何线：先普通再共边覆盖
    for seg in segs:
        if not seg.shared:
            ax.plot(
                [seg.a[0], seg.b[0]],
                [seg.a[1], seg.b[1]],
                color=STYLE["cut"],
                lw=0.8,
                solid_capstyle="round",
                zorder=1,
            )
    for seg in segs:
        if seg.shared:
            ax.plot(
                [seg.a[0], seg.b[0]],
                [seg.a[1], seg.b[1]],
                color=STYLE["shared"],
                lw=2.0,
                solid_capstyle="round",
                zorder=2,
            )

    # 空行程 + 序号
    cur = origin
    ax.scatter([origin[0]], [origin[1]], marker="s", s=40, c="black", zorder=10)
    if show_sequence:
        ax.text(origin[0], origin[1] - 15, "Origin", ha="center", va="top", fontsize=7, color="#333333")

    for i, s in enumerate(oriented):
        seq_num = i + 1
        dist = math.hypot(s.start[0] - cur[0], s.start[1] - cur[1])
        if dist > 1.0:
            if use_curved:
                arrow = patches.FancyArrowPatch(
                    cur,
                    s.start,
                    arrowstyle="->,head_length=3,head_width=2",
                    connectionstyle="arc3,rad=0.15",
                    color=STYLE["air"],
                    lw=0.6,
                    linestyle="--",
                    zorder=0,
                )
                ax.add_patch(arrow)
            else:
                ax.plot(
                    [cur[0], s.start[0]],
                    [cur[1], s.start[1]],
                    color=STYLE["air"],
                    lw=0.6,
                    linestyle="--",
                    zorder=0,
                )

        # 起刀点标记
        ax.scatter([s.start[0]], [s.start[1]], s=15, c=STYLE["start_marker"], zorder=5)

        # 序号标注
        if show_sequence:
            ax.text(
                s.start[0],
                s.start[1],
                str(seq_num),
                color=STYLE["seq_fg"],
                fontsize=5,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.1", fc=STYLE["seq_bg"], ec="none", alpha=0.8),
                zorder=20,
            )

        cur = s.end

    if created_fig and out_path:
        paths = _save_png_and_svg(fig, str(out_path))
        plt.close(fig)
        return paths
    if created_fig:
        plt.close(fig)
    return None


# ---------------- 4 宫格对比图（论文用） ----------------

def plot_paper_comparison_4panel(
    board_baseline: Board,
    board_proposed: Board,
    out_path: str,
) -> None:
    """
    生成 4 宫格对比图：
    (a) 基线布局几何          (b) 共边高亮布局
    (c) 基线刀序              (d) 共边优化刀序
    """
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_board_toolpath(board_baseline, ax=ax1, share_mode="none", show_sequence=False)
    ax1.set_title("(a) Baseline Layout", y=-0.05, fontsize=11)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_board_toolpath(board_proposed, ax=ax2, share_mode="union", show_sequence=False)
    ax2.set_title("(b) Proposed Layout (Shared)", y=-0.05, fontsize=11)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_board_toolpath(board_baseline, ax=ax3, share_mode="none", show_sequence=True)
    ax3.set_title("(c) Baseline Sequence", y=-0.05, fontsize=11)

    ax4 = fig.add_subplot(gs[1, 1])
    plot_board_toolpath(board_proposed, ax=ax4, share_mode="union", show_sequence=True)
    ax4.set_title("(d) Proposed Sequence (Optimized)", y=-0.05, fontsize=11)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=STYLE["cut"], lw=1, label="Cut"),
        Line2D([0], [0], color=STYLE["shared"], lw=2, label="Shared Cut"),
        Line2D([0], [0], color=STYLE["air"], lw=1, ls="--", label="Air Move"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=STYLE["start_marker"], label="Start", markersize=7),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", label="Seq. Number", markersize=7),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )

    _save_png_and_svg(fig, out_path)
    plt.close(fig)
    print(f"[Viz] Generated 4-panel comparison: {out_path}.png")


# ---------------- 兼容封装（runner 调用） ----------------

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
    """
    排样布局图：
    - 绘制板材、修边与零件矩形。
    - show_dims=True 时在每个零件中心标注尺寸（mm）。
    - highlight_shared=True 时叠加共边线。
    """
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
    trim_val = trim if trim and trim > 0 else float(getattr(board, "trim", 0.0) or 0.0)
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
            ax.add_patch(
                Rectangle(
                    (r.x, r.y),
                    r.w,
                    r.h,
                    facecolor="none",
                    edgecolor="#bbbbbb",
                    lw=0.6,
                    ls="--",
                    zorder=1,
                )
            )

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
            sz = min(w0, h0)
            fs = 8 if sz > 200 else (6 if sz > 100 else 5)
            ax.text(
                nominal_x + w0 / 2,
                nominal_y + h0 / 2,
                "\n".join(labels),
                ha="center",
                va="center",
                fontsize=fs,
                color="#111111",
                zorder=3,
            )

    overlay_mode = shared_overlay_mode if highlight_shared else (share_mode if share_mode is not None else None)
    if overlay_mode and overlay_mode != "none":
        try:
            segs_overlay, _, _ = build_segments_from_board(
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
    line_snap_eps: float = 0.0,
    min_shared_len: float = 0.0,
    nd_coord: int = 6,
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
                line_snap_eps=line_snap_eps,
                min_shared_len=min_shared_len,
                nd_coord=nd_coord,
                trim=trim,
                cut_mode=cut_mode,
            )

    # 生成简单的指标预览图，兼容原有输出
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [title, "", "Metrics Summary:"]
    for row in boards_rows:
        try:
            bid = int(row.get("board", row.get("bid", -1)))
        except Exception:
            bid = -1
        try:
            n_parts = int(row.get("n_parts", row.get("N_parts", 0)))
        except Exception:
            n_parts = 0
        util = float(row.get("U", 0.0) or 0.0) * 100.0
        L_cut = float(row.get("L_cut", 0.0) or 0.0)
        L_air = float(row.get("L_air", 0.0) or 0.0)
        lines.append(
            f"Board {bid:02d}: Parts={n_parts} Util={util:.1f}% L_cut={L_cut:.0f}mm L_air={L_air:.0f}mm"
        )

    ax.text(0.05, 0.9, "\n".join(lines), va="top", ha="left", fontsize=10, fontfamily="monospace")
    _save_png_and_svg(fig, out_dir / "boards_metrics_preview")
    plt.close(fig)
