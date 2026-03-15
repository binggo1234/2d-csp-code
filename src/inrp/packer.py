"""Nesting-only packing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
from importlib import metadata as importlib_metadata
import importlib.util
import random

def _version_tuple(text: str) -> Tuple[int, ...]:
    out = []
    for token in str(text).replace("-", ".").split("."):
        if not token.isdigit():
            break
        out.append(int(token))
    return tuple(out)


try:
    if importlib.util.find_spec("numba") is None:
        raise ImportError("numba not installed")
    numpy_version = _version_tuple(importlib_metadata.version("numpy"))
    numba_version = _version_tuple(importlib_metadata.version("numba"))
    if numpy_version >= (2, 0) and numba_version < (0, 60):
        raise ImportError("numba build is too old for numpy 2")
    from numba import jit  # type: ignore
except Exception:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


EPS = 1e-9


@jit(nopython=True)
def _jit_overlap(ax, ay, aw, ah, bx, by, bw, bh, eps):
    return (ax < bx + bw - eps and ax + aw > bx + eps and ay < by + bh - eps and ay + ah > by + eps)


@jit(nopython=True)
def _jit_inside_trim(rx, ry, rw, rh, W, H, trim, eps):
    return (rx >= trim - eps and ry >= trim - eps and rx + rw <= W - trim + eps and ry + rh <= H - trim + eps)


@jit(nopython=True)
def _jit_distance_sq(ax, ay, aw, ah, bx, by, bw, bh):
    dx = 0.0
    ax2 = ax + aw
    bx2 = bx + bw
    if bx > ax2:
        dx = bx - ax2
    elif ax > bx2:
        dx = ax - bx2

    dy = 0.0
    ay2 = ay + ah
    by2 = by + bh
    if by > ay2:
        dy = by - ay2
    elif ay > by2:
        dy = ay - by2

    return dx * dx + dy * dy


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class PlacedPart:
    uid: int
    pid_raw: str
    rect: Rect
    w0: float
    h0: float
    rotated: bool = False


@dataclass
class Board:
    bid: int
    W: float
    H: float
    trim: float = 0.0
    safe_gap: float = 0.0
    touch_tol: float = 1e-6
    placed: List[PlacedPart] = field(default_factory=list)
    free_rects: List[Rect] = field(default_factory=list)
    _geom_sig_cache_nd: int = field(default=-1, init=False, repr=False)
    _geom_sig_cache: Optional[Tuple] = field(default=None, init=False, repr=False)
    _geom_hash_cache_nd: int = field(default=-1, init=False, repr=False)
    _geom_hash_cache: Optional[int] = field(default=None, init=False, repr=False)


@dataclass(frozen=True)
class LocalPlacementBlueprint:
    rect: Rect
    rotated: bool
    area_fit: float
    short_side: float
    long_side: float
    cavity_ratio: float
    fragmentation_penalty: float
    free_rects_after: Tuple[Rect, ...]


def board_used_area(board: Board) -> float:
    return sum(float(pp.w0) * float(pp.h0) for pp in board.placed)


def board_utilization(board: Board) -> float:
    area = float(board.W) * float(board.H)
    if area <= 0.0:
        return 0.0
    return board_used_area(board) / area


def board_area(board: Board) -> float:
    return float(board.W) * float(board.H)


def _invalidate_board_cache(board: Board) -> None:
    board._geom_sig_cache_nd = -1
    board._geom_sig_cache = None
    board._geom_hash_cache_nd = -1
    board._geom_hash_cache = None


def make_empty_board(
    bid: int,
    W: float,
    H: float,
    *,
    trim: float = 0.0,
    safe_gap: float = 0.0,
    touch_tol: float = 1e-6,
    place_mode: str = "maxrects_baf",
) -> Board:
    board = Board(
        bid=int(bid),
        W=float(W),
        H=float(H),
        trim=float(trim),
        safe_gap=float(safe_gap),
        touch_tol=float(touch_tol),
    )
    if str(place_mode).lower().startswith("maxrects"):
        _init_free_rects(board)
    return board


def clone_board(board: Board) -> Board:
    return Board(
        bid=int(board.bid),
        W=float(board.W),
        H=float(board.H),
        trim=float(board.trim),
        safe_gap=float(board.safe_gap),
        touch_tol=float(board.touch_tol),
        placed=list(board.placed),
        free_rects=list(board.free_rects),
    )


def _round_sig(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def board_signature(board: Board, nd: int = 3, *, include_bid: bool = True) -> Tuple:
    placed_sig = tuple(
        (
            int(pp.uid),
            _round_sig(pp.rect.x, nd),
            _round_sig(pp.rect.y, nd),
            _round_sig(pp.rect.w, nd),
            _round_sig(pp.rect.h, nd),
            int(bool(pp.rotated)),
        )
        for pp in board.placed
    )
    free_sig = tuple(
        (
            _round_sig(fr.x, nd),
            _round_sig(fr.y, nd),
            _round_sig(fr.w, nd),
            _round_sig(fr.h, nd),
        )
        for fr in board.free_rects
    )
    out = (
        _round_sig(board.W, nd),
        _round_sig(board.H, nd),
        _round_sig(board.trim, nd),
        _round_sig(board.safe_gap, nd),
        _round_sig(board.touch_tol, nd),
        placed_sig,
        free_sig,
    )
    if include_bid:
        return (int(board.bid),) + out
    return out


def board_geometry_signature(board: Board, nd: int = 3) -> Tuple:
    cached = board._geom_sig_cache
    if cached is not None and int(board._geom_sig_cache_nd) == int(nd):
        return cached
    out = board_signature(board, nd=nd, include_bid=False)
    board._geom_sig_cache_nd = int(nd)
    board._geom_sig_cache = out
    return out


def layout_signature(
    boards: Sequence[Board],
    nd: int = 3,
    *,
    include_bid: bool = True,
    canonical: bool = False,
) -> Tuple:
    sigs = tuple(board_signature(board, nd=nd, include_bid=include_bid) for board in boards)
    if canonical:
        return tuple(sorted(sigs))
    return sigs


def canonical_layout_signature(boards: Sequence[Board], nd: int = 3) -> Tuple:
    return layout_signature(boards, nd=nd, include_bid=False, canonical=True)


def _can_place(board: Board, rect: Rect) -> bool:
    if not _jit_inside_trim(rect.x, rect.y, rect.w, rect.h, board.W, board.H, board.trim, EPS):
        return False

    safe_gap = float(board.safe_gap or 0.0)
    touch_tol = float(board.touch_tol or 0.0)
    safe_sq = safe_gap * safe_gap
    touch_sq = touch_tol * touch_tol

    for pp in board.placed:
        pr = pp.rect
        if _jit_overlap(rect.x, rect.y, rect.w, rect.h, pr.x, pr.y, pr.w, pr.h, EPS):
            return False
        if safe_sq > 0.0:
            d2 = _jit_distance_sq(rect.x, rect.y, rect.w, rect.h, pr.x, pr.y, pr.w, pr.h)
            if d2 > touch_sq and d2 < safe_sq:
                return False
    return True


def _candidate_points(board: Board) -> List[Tuple[float, float]]:
    pts = {(float(board.trim), float(board.trim))}
    for pp in board.placed:
        x, y, w, h = pp.rect.x, pp.rect.y, pp.rect.w, pp.rect.h
        pts.add((x + w, y))
        pts.add((x, y + h))
    return sorted(pts, key=lambda t: (t[1], t[0]))


def _init_free_rects(board: Board) -> None:
    w = float(board.W) - 2.0 * float(board.trim)
    h = float(board.H) - 2.0 * float(board.trim)
    board.free_rects = [Rect(float(board.trim), float(board.trim), w, h)] if w > EPS and h > EPS else []
    _invalidate_board_cache(board)


def _rect_contains(a: Rect, b: Rect, eps: float = 1e-9) -> bool:
    return (
        b.x >= a.x - eps
        and b.y >= a.y - eps
        and b.x + b.w <= a.x + a.w + eps
        and b.y + b.h <= a.y + a.h + eps
    )


def _rect_intersects(a: Rect, b: Rect, eps: float = 1e-9) -> bool:
    return not (
        b.x >= a.x + a.w - eps
        or b.x + b.w <= a.x + eps
        or b.y >= a.y + a.h - eps
        or b.y + b.h <= a.y + eps
    )


def _split_free_rect(fr: Rect, used: Rect) -> List[Rect]:
    if not _rect_intersects(fr, used, eps=EPS):
        return [fr]

    out: List[Rect] = []
    fx2 = fr.x + fr.w
    fy2 = fr.y + fr.h
    ux2 = used.x + used.w
    uy2 = used.y + used.h

    if used.x > fr.x + EPS:
        out.append(Rect(fr.x, fr.y, used.x - fr.x, fr.h))
    if ux2 < fx2 - EPS:
        out.append(Rect(ux2, fr.y, fx2 - ux2, fr.h))
    if used.y > fr.y + EPS:
        out.append(Rect(fr.x, fr.y, fr.w, used.y - fr.y))
    if uy2 < fy2 - EPS:
        out.append(Rect(fr.x, uy2, fr.w, fy2 - uy2))
    return [r for r in out if r.w > EPS and r.h > EPS]


def _prune_free_rects(free_rects: List[Rect]) -> List[Rect]:
    out: List[Rect] = []
    for i, ri in enumerate(free_rects):
        if any(i != j and _rect_contains(rj, ri, eps=EPS) for j, rj in enumerate(free_rects)):
            continue
        out.append(ri)
    return out


def _update_free_rects_after_place(board: Board, used: Rect) -> None:
    if not board.free_rects:
        return
    pieces: List[Rect] = []
    for fr in board.free_rects:
        pieces.extend(_split_free_rect(fr, used))
    board.free_rects = _prune_free_rects(pieces)
    _invalidate_board_cache(board)


def _free_rects_after_place(board: Board, used: Rect) -> Tuple[Rect, ...]:
    tmp = Board(
        bid=board.bid,
        W=board.W,
        H=board.H,
        trim=board.trim,
        safe_gap=board.safe_gap,
        touch_tol=board.touch_tol,
        placed=board.placed,
        free_rects=list(board.free_rects),
    )
    _update_free_rects_after_place(tmp, used)
    return tuple(tmp.free_rects)


def _fragmentation_penalty(part, free_rects_after: Sequence[Rect]) -> float:
    if not free_rects_after:
        return 0.0
    pw = max(float(part.w), float(part.h))
    ph = min(float(part.w), float(part.h))
    slivers = 0.0
    for fr in free_rects_after:
        if fr.w < 0.35 * pw or fr.h < 0.35 * ph:
            slivers += 1.0
    return float(len(free_rects_after)) + 0.25 * slivers


def _local_blueprint_from_candidate(board: Board, part, rect: Rect, rotated: bool, area_fit: float, short_side: float, long_side: float) -> LocalPlacementBlueprint:
    free_rects_after = _free_rects_after_place(board, rect)
    rect_area = max(EPS, float(rect.w) * float(rect.h))
    cavity_ratio = (float(part.w0) * float(part.h0)) / rect_area
    frag_penalty = _fragmentation_penalty(part, free_rects_after)
    return LocalPlacementBlueprint(
        rect=rect,
        rotated=bool(rotated),
        area_fit=float(area_fit),
        short_side=float(short_side),
        long_side=float(long_side),
        cavity_ratio=float(cavity_ratio),
        fragmentation_penalty=float(frag_penalty),
        free_rects_after=free_rects_after,
    )


def _local_blueprint_key(bp: LocalPlacementBlueprint) -> Tuple[float, float, float, float, float]:
    return (
        float(bp.cavity_ratio),
        -float(bp.fragmentation_penalty),
        -float(bp.area_fit),
        -float(bp.long_side),
        -float(bp.short_side),
    )


def _part_dims(part, allow_rot: bool) -> List[Tuple[float, float, bool]]:
    dims = [(float(part.w), float(part.h), False)]
    if allow_rot and abs(float(part.w) - float(part.h)) > EPS:
        dims.append((float(part.h), float(part.w), True))
    return dims


def _append_part(board: Board, part, rect: Rect, rotated: bool) -> None:
    board.placed.append(
        PlacedPart(
            uid=int(part.uid),
            pid_raw=str(part.pid_raw),
            rect=rect,
            w0=float(part.w0),
            h0=float(part.h0),
            rotated=bool(rotated),
        )
    )
    _invalidate_board_cache(board)


def _place_one(
    board: Board,
    part,
    allow_rot: bool,
    *,
    place_mode: str = "maxrects_baf",
    rng: Optional[random.Random] = None,
    rand_topk: int = 3,
) -> bool:
    dims = _part_dims(part, allow_rot)

    if place_mode in {"maxrects_bssf", "maxrects_baf"}:
        if not board.free_rects:
            _init_free_rects(board)
        best = None
        for fr in board.free_rects:
            for cw, ch, rotated in dims:
                if cw > fr.w + EPS or ch > fr.h + EPS:
                    continue
                rect = Rect(fr.x, fr.y, cw, ch)
                if not _can_place(board, rect):
                    continue
                rem_w = max(fr.w - cw, 0.0)
                rem_h = max(fr.h - ch, 0.0)
                short_side = min(rem_w, rem_h)
                long_side = max(rem_w, rem_h)
                area_fit = max(fr.w * fr.h - cw * ch, 0.0)
                if place_mode == "maxrects_bssf":
                    key = (short_side, long_side, area_fit, fr.y, fr.x)
                else:
                    key = (area_fit, short_side, long_side, fr.y, fr.x)
                if best is None or key < best[0]:
                    best = (key, rect, rotated)
        if best is None:
            return False
        _, rect_best, rotated_best = best
        _append_part(board, part, rect_best, rotated_best)
        _update_free_rects_after_place(board, rect_best)
        return True

    pts = _candidate_points(board)

    if place_mode == "blf":
        for x, y in pts:
            for cw, ch, rotated in dims:
                rect = Rect(x, y, cw, ch)
                if _can_place(board, rect):
                    _append_part(board, part, rect, rotated)
                    return True
        return False

    if place_mode == "blf_rand":
        rng = rng or random.Random(0)
        dims2 = dims[:]
        rng.shuffle(dims2)
        topk = max(1, int(rand_topk))
        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            candidates = []
            while i < len(pts) and abs(pts[i][1] - y_level) <= 1e-12:
                x, y = pts[i]
                for cw, ch, rotated in dims2:
                    rect = Rect(x, y, cw, ch)
                    if _can_place(board, rect):
                        candidates.append((x, rect, rotated))
                i += 1
            if candidates:
                candidates.sort(key=lambda item: item[0])
                _, rect_best, rotated_best = rng.choice(candidates[: min(topk, len(candidates))])
                _append_part(board, part, rect_best, rotated_best)
                return True
        return False

    raise ValueError(f"unknown place_mode: {place_mode}")


def best_local_placement(
    board: Board,
    part,
    allow_rot: bool,
    *,
    place_mode: str = "maxrects_baf",
) -> Optional[LocalPlacementBlueprint]:
    dims = _part_dims(part, allow_rot)
    mode = str(place_mode).strip().lower()
    best: Optional[LocalPlacementBlueprint] = None

    if mode in {"maxrects_bssf", "maxrects_baf"}:
        if not board.free_rects:
            _init_free_rects(board)
        for fr in board.free_rects:
            for cw, ch, rotated in dims:
                if cw > fr.w + EPS or ch > fr.h + EPS:
                    continue
                rect = Rect(fr.x, fr.y, cw, ch)
                if not _can_place(board, rect):
                    continue
                rem_w = max(fr.w - cw, 0.0)
                rem_h = max(fr.h - ch, 0.0)
                short_side = min(rem_w, rem_h)
                long_side = max(rem_w, rem_h)
                area_fit = max(fr.w * fr.h - cw * ch, 0.0)
                cand = _local_blueprint_from_candidate(board, part, rect, rotated, area_fit, short_side, long_side)
                if best is None or _local_blueprint_key(cand) > _local_blueprint_key(best):
                    best = cand
        return best

    pts = _candidate_points(board)
    for x, y in pts:
        for cw, ch, rotated in dims:
            rect = Rect(x, y, cw, ch)
            if not _can_place(board, rect):
                continue
            cand = _local_blueprint_from_candidate(
                board,
                part,
                rect,
                rotated,
                area_fit=0.0,
                short_side=0.0,
                long_side=0.0,
            )
            if best is None or _local_blueprint_key(cand) > _local_blueprint_key(best):
                best = cand
    return best


def apply_local_blueprint(board: Board, part, blueprint: LocalPlacementBlueprint) -> Board:
    out = clone_board(board)
    out.placed.append(
        PlacedPart(
            uid=int(part.uid),
            pid_raw=str(part.pid_raw),
            rect=Rect(
                float(blueprint.rect.x),
                float(blueprint.rect.y),
                float(blueprint.rect.w),
                float(blueprint.rect.h),
            ),
            w0=float(part.w0),
            h0=float(part.h0),
            rotated=bool(blueprint.rotated),
        )
    )
    out.free_rects = list(blueprint.free_rects_after)
    return out


def _fits_on_board(part, W: float, H: float, allow_rot: bool, trim: float) -> bool:
    usable_w = W - 2.0 * float(trim)
    usable_h = H - 2.0 * float(trim)
    if usable_w <= 0 or usable_h <= 0:
        return False
    return bool(
        (part.w <= usable_w + EPS and part.h <= usable_h + EPS)
        or (allow_rot and part.h <= usable_w + EPS and part.w <= usable_h + EPS)
    )


def pack_one_board(
    parts: Sequence,
    W: float,
    H: float,
    allow_rot: bool,
    bid: int,
    *,
    trim: float = 0.0,
    safe_gap: float = 0.0,
    touch_tol: float = 1e-6,
    order: str = "size",
    place_mode: str = "maxrects_baf",
    rng: Optional[random.Random] = None,
    rand_topk: int = 3,
) -> Tuple[Board, List]:
    board = Board(bid=bid, W=W, H=H, trim=float(trim), safe_gap=float(safe_gap), touch_tol=float(touch_tol))
    if str(place_mode).lower().startswith("maxrects"):
        _init_free_rects(board)

    for part in parts:
        if not _fits_on_board(part, W, H, allow_rot, trim):
            raise RuntimeError(f"Part too large for board: uid={part.uid}")

    if order == "size":
        parts_sorted = sorted(parts, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    elif order == "area":
        parts_sorted = sorted(parts, key=lambda p: p.w * p.h, reverse=True)
    else:
        parts_sorted = list(parts)

    remaining = []
    for part in parts_sorted:
        if not _place_one(board, part, allow_rot, place_mode=place_mode, rng=rng, rand_topk=rand_topk):
            remaining.append(part)
    return board, remaining


def pack_multi_board(
    parts: Sequence,
    W: float,
    H: float,
    allow_rot: bool,
    *,
    trim: float = 0.0,
    safe_gap: float = 0.0,
    touch_tol: float = 1e-6,
    max_boards: int = 10000,
    order: str = "size",
    place_mode: str = "maxrects_baf",
    rng: Optional[random.Random] = None,
    rand_topk: int = 3,
) -> List[Board]:
    left = list(parts)
    boards: List[Board] = []
    bid = 1

    while left:
        if bid > max_boards:
            raise RuntimeError("Max boards limit reached.")
        board, left_next = pack_one_board(
            left,
            W,
            H,
            allow_rot,
            bid=bid,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            order=order,
            place_mode=place_mode,
            rng=rng,
            rand_topk=rand_topk,
        )
        if not board.placed:
            raise RuntimeError(f"Cannot place part: uid={left[0].uid}")
        boards.append(board)
        left = left_next
        bid += 1

    return boards
