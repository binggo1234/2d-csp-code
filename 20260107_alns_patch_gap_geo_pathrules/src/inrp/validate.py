# validate.py
from __future__ import annotations

from typing import List

import logging

logger = logging.getLogger(__name__)

EPS = 1e-9


def _inside(r, W: float, H: float, trim: float = 0.0) -> bool:
    """Inside check with board-edge trimming."""
    t = float(trim)
    return (
        r.x >= t - EPS
        and r.y >= t - EPS
        and r.x + r.w <= W - t + EPS
        and r.y + r.h <= H - t + EPS
    )


def _overlap(a, b) -> bool:
    return (
        a.x < b.x + b.w - EPS
        and a.x + a.w > b.x + EPS
        and a.y < b.y + b.h - EPS
        and a.y + a.h > b.y + EPS
    )

def _distance_sq(a, b) -> float:
    """Squared Euclidean distance between two axis-aligned rectangles (non-overlap case)."""
    dx = 0.0
    ax2 = a.x + a.w
    bx2 = b.x + b.w
    if b.x > ax2:
        dx = b.x - ax2
    elif a.x > bx2:
        dx = a.x - bx2

    dy = 0.0
    ay2 = a.y + a.h
    by2 = b.y + b.h
    if b.y > ay2:
        dy = b.y - ay2
    elif a.y > by2:
        dy = a.y - by2

    return dx * dx + dy * dy



def validate_solution(parts: List, boards: List, W: float, H: float, *, trim: float = 0.0,
                      safe_gap: float = 0.0, touch_tol: float = 1e-6) -> None:
    """Validate:
    1) all input parts are placed exactly once
    2) each placed rect is inside the trimmed feasible domain
    3) no overlaps
    4) binary spacing constraint (optional): delta==0 or delta>=safe_gap

    Note: feasibility is checked on inflated rects (pp.rect).
    """

    in_uids = [p.uid for p in parts]
    placed_uids = []
    for b in boards:
        for pp in b.placed:
            placed_uids.append(pp.uid)

    if len(placed_uids) != len(in_uids) or len(set(placed_uids)) != len(set(in_uids)):
        miss = sorted(set(in_uids) - set(placed_uids))
        extra = sorted(set(placed_uids) - set(in_uids))
        dup = [u for u in set(placed_uids) if placed_uids.count(u) > 1]
        logger.warning("[CHECK] miss: %s%s", miss[:20], "..." if len(miss) > 20 else "")
        logger.warning("[CHECK] extra: %s%s", extra[:20], "..." if len(extra) > 20 else "")
        logger.warning("[CHECK] dup: %s%s", dup[:20], "..." if len(dup) > 20 else "")
        raise RuntimeError("放置数量校验失败：存在缺失/重复/多余。")

    for b in boards:
        ps = b.placed
        for pp in ps:
            if not _inside(pp.rect, W, H, trim=trim):
                raise RuntimeError(
                    f"越界（考虑TRIM={trim}）：board={b.bid} uid={pp.uid} raw={pp.pid_raw} "
                    f"rect=({pp.rect.x},{pp.rect.y},{pp.rect.w},{pp.rect.h}) W={W} H={H}"
                )
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                if _overlap(ps[i].rect, ps[j].rect):
                    raise RuntimeError(
                        f"重叠：board={b.bid}\n"
                        f"  A uid={ps[i].uid} raw={ps[i].pid_raw} rect=({ps[i].rect.x},{ps[i].rect.y},{ps[i].rect.w},{ps[i].rect.h})\n"
                        f"  B uid={ps[j].uid} raw={ps[j].pid_raw} rect=({ps[j].rect.x},{ps[j].rect.y},{ps[j].rect.w},{ps[j].rect.h})"
                    )

                # Binary spacing constraint (CNC feasibility)
                sg = float(safe_gap or 0.0)
                if sg > 0.0:
                    tol = float(touch_tol or 0.0)
                    d2 = _distance_sq(ps[i].rect, ps[j].rect)
                    if d2 > tol * tol and d2 < sg * sg:
                        raise RuntimeError(
                            f"不可加工微缝：board={b.bid} SAFE_GAP={sg} TOOL_D?\n"
                            f"  A uid={ps[i].uid} raw={ps[i].pid_raw} rect=({ps[i].rect.x},{ps[i].rect.y},{ps[i].rect.w},{ps[i].rect.h})\n"
                            f"  B uid={ps[j].uid} raw={ps[j].pid_raw} rect=({ps[j].rect.x},{ps[j].rect.y},{ps[j].rect.w},{ps[j].rect.h})\n"
                            f"  d2={d2}"
                        )
