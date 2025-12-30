# viz.py
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)
from pathlib import Path

def plot_one_seed_outputs(boards, boards_rows, out_dir: Path, title: str,
                          max_boards: int = 6) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        logger.warning("[PLOT] skip (matplotlib not available): %s", e)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 只画前 max_boards 张
    for b in boards[:max_boards]:
        # layout
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"{title} | board={b.bid} layout")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, b.W)
        ax.set_ylim(0, b.H)
        ax.add_patch(Rectangle((0, 0), b.W, b.H, fill=False))
        for pp in b.placed:
            r = pp.rect
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False))
        fig.savefig(out_dir / f"board_{b.bid:03d}_layout.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 画每块板的 metrics 文字图（快速检查）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [title, ""]
    for row in boards_rows:
        lines.append(
            f"board={int(row['board']):03d} n={int(row['n_parts'])} "
            f"U={row['U']:.4f} L_shared={row['L_shared']:.1f} "
            f"N_lift={int(row['N_lift'])} L_air={row['L_air']:.1f} "
            f"L_cut={row['L_cut']:.1f} T={row['T_est']:.2f}"
        )
    ax.text(0.01, 0.99, "\n".join(lines), va="top", ha="left", fontsize=9)
    fig.savefig(out_dir / "boards_metrics_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
