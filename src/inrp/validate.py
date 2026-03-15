from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

EPS = 1e-9


def _inside(rect, W: float, H: float, trim: float = 0.0) -> bool:
    t = float(trim)
    return (
        rect.x >= t - EPS
        and rect.y >= t - EPS
        and rect.x + rect.w <= W - t + EPS
        and rect.y + rect.h <= H - t + EPS
    )


def _overlap(a, b) -> bool:
    return (
        a.x < b.x + b.w - EPS
        and a.x + a.w > b.x + EPS
        and a.y < b.y + b.h - EPS
        and a.y + a.h > b.y + EPS
    )


def _distance_sq(a, b) -> float:
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


def validate_solution(
    parts: List,
    boards: List,
    W: float,
    H: float,
    *,
    trim: float = 0.0,
    safe_gap: float = 0.0,
    touch_tol: float = 1e-6,
) -> None:
    input_uids = [p.uid for p in parts]
    placed_uids = [pp.uid for board in boards for pp in board.placed]

    if len(placed_uids) != len(input_uids) or len(set(placed_uids)) != len(set(input_uids)):
        missing = sorted(set(input_uids) - set(placed_uids))
        extra = sorted(set(placed_uids) - set(input_uids))
        dup = [uid for uid in set(placed_uids) if placed_uids.count(uid) > 1]
        logger.warning("[CHECK] missing=%s", missing[:10])
        logger.warning("[CHECK] extra=%s", extra[:10])
        logger.warning("[CHECK] duplicate=%s", dup[:10])
        raise RuntimeError("Placement validation failed: missing, extra, or duplicated parts detected.")

    for board in boards:
        placed = board.placed
        for pp in placed:
            if not _inside(pp.rect, W, H, trim=trim):
                raise RuntimeError(
                    f"Part out of board: board={board.bid} uid={pp.uid} rect="
                    f"({pp.rect.x},{pp.rect.y},{pp.rect.w},{pp.rect.h})"
                )

        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                if _overlap(placed[i].rect, placed[j].rect):
                    raise RuntimeError(
                        f"Overlap detected on board={board.bid}: uid={placed[i].uid} and uid={placed[j].uid}"
                    )

                gap = float(safe_gap or 0.0)
                if gap <= 0.0:
                    continue
                tol = float(touch_tol or 0.0)
                d2 = _distance_sq(placed[i].rect, placed[j].rect)
                if d2 > tol * tol and d2 < gap * gap:
                    raise RuntimeError(
                        f"Binary spacing violated on board={board.bid}: "
                        f"uid={placed[i].uid} and uid={placed[j].uid}"
                    )
