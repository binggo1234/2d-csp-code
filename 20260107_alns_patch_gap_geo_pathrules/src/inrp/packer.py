# src/inrp/packer.py
"""Packing (nesting) utilities.

Key change for CIE paper consistency:
- TRIM is treated as board-edge trimming allowance (board feasible domain shrink),
  NOT part inflation.
- Part inflation is controlled by GAP only (see dataio.read_sample_parts).

Coordinate system:
- Board outer boundary: [0, W] x [0, H]
- Feasible placement domain with trim: [TRIM, W-TRIM] x [TRIM, H-TRIM]
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
import random
import math
import importlib.util
from collections import defaultdict

# --- Numba JIT 导入与回退机制 ---
# 注意：某些环境下 numba 可能“可见但不可用”（例如与 coverage/numba 版本冲突）。
# 因此这里必须用 try/except，而不能仅用 find_spec。
try:
    if importlib.util.find_spec("numba") is not None:
        from numba import jit  # type: ignore
        HAS_NUMBA = True
    else:
        raise ImportError("numba not installed")
except Exception:
    HAS_NUMBA = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
# --------------------------------------

EPS = 1e-9

# --- 静态 JIT 函数 (极速几何核心) ---
@jit(nopython=True)
def _jit_overlap(ax, ay, aw, ah, bx, by, bw, bh, eps):
    # strict overlap
    return (ax < bx + bw - eps and ax + aw > bx + eps and
            ay < by + bh - eps and ay + ah > by + eps)


@jit(nopython=True)
def _jit_inside_trim(rx, ry, rw, rh, W, H, trim, eps):
    """Inside check with board-edge trimming."""
    return (rx >= trim - eps and ry >= trim - eps and
            rx + rw <= W - trim + eps and ry + rh <= H - trim + eps)
@jit(nopython=True)
def _jit_distance_sq(ax, ay, aw, ah, bx, by, bw, bh):
    """Squared Euclidean distance between two axis-aligned rectangles (non-overlap case).

    Returns 0 when rectangles touch (edge/corner) or overlap in projection.
    """
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
# ------------------------------------------


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
    rect: Rect              # inflated rect (for packing/validation)
    w0: float               # original width
    h0: float               # original height
    rotated: bool = False
    eb_L: int = 0
    eb_R: int = 0
    eb_B: int = 0
    eb_T: int = 0
    edge_class: str = ""


@dataclass
class Board:
    bid: int
    W: float
    H: float
    trim: float = 0.0
    # Binary spacing: allow touch (delta≈0) OR keep delta>=safe_gap
    safe_gap: float = 0.0
    touch_tol: float = 1e-6
    placed: List[PlacedPart] = field(default_factory=list)
    free_rects: List[Rect] = field(default_factory=list)


# --- 使用 JIT 核心的包装函数 ---

def _overlap(a: Rect, b: Rect) -> bool:
    return _jit_overlap(a.x, a.y, a.w, a.h, b.x, b.y, b.w, b.h, EPS)


def _inside_board(r: Rect, board: Board) -> bool:
    return _jit_inside_trim(r.x, r.y, r.w, r.h, board.W, board.H, board.trim, EPS)


def _can_place(board: Board, r: Rect) -> bool:
    # Fast unpack to minimize attribute access
    rx, ry, rw, rh = r.x, r.y, r.w, r.h

    if not _jit_inside_trim(rx, ry, rw, rh, board.W, board.H, board.trim, EPS):
        return False

    safe_gap = float(getattr(board, "safe_gap", 0.0) or 0.0)
    touch_tol = float(getattr(board, "touch_tol", 1e-6) or 0.0)
    safe_sq = safe_gap * safe_gap
    touch_sq = touch_tol * touch_tol

    for pp in board.placed:
        pr = pp.rect
        # 1) Overlap is strictly forbidden
        if _jit_overlap(rx, ry, rw, rh, pr.x, pr.y, pr.w, pr.h, EPS):
            return False
        # 2) Binary spacing constraint (CNC milling feasibility)
        #    allow: d≈0 (touch/share) OR d>=safe_gap
        if safe_sq > 0.0:
            d2 = _jit_distance_sq(rx, ry, rw, rh, pr.x, pr.y, pr.w, pr.h)
            if d2 > touch_sq and d2 < safe_sq:
                return False
    return True
# --------------------------------------


def _candidate_points(board: Board) -> List[Tuple[float, float]]:
    """Bottom-left candidate points.

    Start from (trim, trim) so all placements satisfy board-edge trimming.
    """
    pts = {(float(board.trim), float(board.trim))}
    for pp in board.placed:
        x, y, w, h = pp.rect.x, pp.rect.y, pp.rect.w, pp.rect.h
        pts.add((x + w, y))
        pts.add((x, y + h))
    return sorted(pts, key=lambda t: (t[1], t[0]))


def _touch_shared_len(a: Rect, b: Rect) -> float:
    """Shared edge length between two rectangles if they touch."""
    shared = 0.0
    if abs((a.x + a.w) - b.x) <= 1e-6 or abs((b.x + b.w) - a.x) <= 1e-6:
        y0 = max(a.y, b.y)
        y1 = min(a.y + a.h, b.y + b.h)
        if y1 - y0 > 1e-6:
            shared += (y1 - y0)
    if abs((a.y + a.h) - b.y) <= 1e-6 or abs((b.y + b.h) - a.y) <= 1e-6:
        x0 = max(a.x, b.x)
        x1 = min(a.x + a.w, b.x + b.w)
        if x1 - x0 > 1e-6:
            shared += (x1 - x0)
    return shared


def _border_contact(r: Rect, board: Board) -> float:
    """Contact length between rectangle and *trimmed* border."""
    c = 0.0
    t = float(board.trim)
    W = float(board.W)
    H = float(board.H)

    # left / right borders at x=t and x=W-t
    if abs(r.x - t) <= 1e-6:
        c += r.h
    if abs((r.x + r.w) - (W - t)) <= 1e-6:
        c += r.h

    # bottom / top borders at y=t and y=H-t
    if abs(r.y - t) <= 1e-6:
        c += r.w
    if abs((r.y + r.h) - (H - t)) <= 1e-6:
        c += r.w

    return c


def _edge_flags_after_rotation(eb_L: int, eb_R: int, eb_B: int, eb_T: int, rotated: bool) -> Tuple[int, int, int, int]:
    if not rotated:
        return int(eb_L), int(eb_R), int(eb_B), int(eb_T)
    # 90deg rotation mapping (deterministic convention):
    # old B->new L, old T->new R, old R->new B, old L->new T
    return int(eb_B), int(eb_T), int(eb_R), int(eb_L)


def _init_free_rects(board: Board) -> None:
    w = float(board.W) - 2.0 * float(board.trim)
    h = float(board.H) - 2.0 * float(board.trim)
    if w > EPS and h > EPS:
        board.free_rects = [Rect(float(board.trim), float(board.trim), w, h)]
    else:
        board.free_rects = []


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
        contained = False
        for j, rj in enumerate(free_rects):
            if i == j:
                continue
            if _rect_contains(rj, ri, eps=EPS):
                contained = True
                break
        if not contained:
            out.append(ri)
    return out


def _update_free_rects_after_place(board: Board, used: Rect) -> None:
    if not board.free_rects:
        return
    pieces: List[Rect] = []
    for fr in board.free_rects:
        pieces.extend(_split_free_rect(fr, used))
    board.free_rects = _prune_free_rects(pieces)


def _place_one(
        board: Board,
        uid: int,
        pid_raw: str,
        w: float,
        h: float,
        w0: float,
        h0: float,
        eb_L: int,
        eb_R: int,
        eb_B: int,
        eb_T: int,
        edge_class: str,
        allow_rot: bool,
        place_mode: str = "blf",
        rng: Optional[random.Random] = None,
        rand_topk: int = 3,
) -> bool:
    dims = [(w, h, False)]
    if allow_rot and abs(w - h) > EPS:
        dims.append((h, w, True))

    def _append(r: Rect, rot: bool) -> None:
        eL, eR, eB, eT = _edge_flags_after_rotation(eb_L, eb_R, eb_B, eb_T, rot)
        board.placed.append(
            PlacedPart(
                uid=uid,
                pid_raw=pid_raw,
                rect=r,
                w0=w0,
                h0=h0,
                rotated=rot,
                eb_L=eL,
                eb_R=eR,
                eb_B=eB,
                eb_T=eT,
                edge_class=edge_class,
            )
        )

    if place_mode in {"maxrects_bssf", "maxrects_baf"}:
        if not board.free_rects:
            _init_free_rects(board)
        best = None
        for fr in board.free_rects:
            for (cw, ch, rot) in dims:
                if cw > fr.w + EPS or ch > fr.h + EPS:
                    continue
                r = Rect(fr.x, fr.y, cw, ch)
                if not _can_place(board, r):
                    continue
                rem_w = max(fr.w - cw, 0.0)
                rem_h = max(fr.h - ch, 0.0)
                short_side = min(rem_w, rem_h)
                long_side = max(rem_w, rem_h)
                area_fit = max(fr.w * fr.h - cw * ch, 0.0)
                if place_mode == "maxrects_bssf":
                    key = (short_side, long_side, area_fit, fr.y, fr.x)
                else:  # maxrects_baf
                    key = (area_fit, short_side, long_side, fr.y, fr.x)
                if best is None or key < best[0]:
                    best = (key, r, rot)
        if best is None:
            return False
        _, r_best, rot_best = best
        _append(r_best, rot_best)
        _update_free_rects_after_place(board, r_best)
        return True

    pts = _candidate_points(board)

    if place_mode == "blf":
        for (x, y) in pts:
            for (cw, ch, rot) in dims:
                r = Rect(x, y, cw, ch)
                if _can_place(board, r):
                    _append(r, rot)
                    return True
        return False

    if place_mode == "blf_rand":
        rng = rng or random.Random(0)
        topk = max(1, int(rand_topk))
        dims2 = dims[:]
        rng.shuffle(dims2)
        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            cand = []
            while i < len(pts) and abs(pts[i][1] - y_level) <= 1e-12:
                x, y = pts[i]
                for (cw, ch, rot) in dims2:
                    r = Rect(x, y, cw, ch)
                    if _can_place(board, r):
                        cand.append((x, r, rot))
                i += 1
            if cand:
                cand.sort(key=lambda t: t[0])
                pick_pool = cand[:min(topk, len(cand))]
                _x_pick, r_best, rot_best = rng.choice(pick_pool)
                _append(r_best, rot_best)
                return True
        return False

    if place_mode == "blf_ccf":
        i = 0
        while i < len(pts):
            y_level = pts[i][1]
            cand = []
            while i < len(pts) and abs(pts[i][1] - y_level) <= 1e-12:
                x, y = pts[i]
                for (cw, ch, rot) in dims:
                    r = Rect(x, y, cw, ch)
                    if _can_place(board, r):
                        shared = 0.0
                        for pp in board.placed:
                            shared += _touch_shared_len(r, pp.rect)
                        border = _border_contact(r, board)
                        key = (-shared, -border, x)
                        cand.append((key, r, rot))
                i += 1
            if cand:
                cand.sort(key=lambda t: t[0])
                _, r_best, rot_best = cand[0]
                _append(r_best, rot_best)
                return True
        return False

    raise ValueError(f"unknown place_mode: {place_mode}")


def _fits_on_board(p, W: float, H: float, allow_rot: bool, trim: float) -> bool:
    """Quick feasibility check using usable board size (W-2*trim, H-2*trim)."""
    usable_W = W - 2.0 * float(trim)
    usable_H = H - 2.0 * float(trim)
    if usable_W <= 0 or usable_H <= 0:
        return False
    ok = (p.w <= usable_W + EPS and p.h <= usable_H + EPS) or (allow_rot and p.h <= usable_W + EPS and p.w <= usable_H + EPS)
    return bool(ok)


def pack_one_board(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        bid: int,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        order: str = "size",
        place_mode: str = "blf",
        rng: Optional[random.Random] = None,
        rand_topk: int = 3,
) -> Tuple[Board, List]:
    board = Board(bid=bid, W=W, H=H, trim=float(trim), safe_gap=float(safe_gap), touch_tol=float(touch_tol))
    if str(place_mode).lower().startswith("maxrects"):
        _init_free_rects(board)
    for p in parts:
        if not _fits_on_board(p, W, H, allow_rot, trim):
            raise RuntimeError(f"Part too big for trimmed board: uid={p.uid} trim={trim}")

    if order == "size":
        parts_sorted = sorted(parts, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    elif order == "area":
        parts_sorted = sorted(parts, key=lambda p: p.w * p.h, reverse=True)
    elif order == "none":
        parts_sorted = parts[:]
    else:
        parts_sorted = parts[:]

    remaining = []
    for p in parts_sorted:
        ok = _place_one(
            board,
            p.uid,
            p.pid_raw,
            p.w,
            p.h,
            p.w0,
            p.h0,
            getattr(p, "eb_L", 0),
            getattr(p, "eb_R", 0),
            getattr(p, "eb_B", 0),
            getattr(p, "eb_T", 0),
            getattr(p, "edge_class", ""),
            allow_rot,
            place_mode=place_mode,
            rng=rng,
            rand_topk=rand_topk,
        )
        if not ok:
            remaining.append(p)
    return board, remaining


def pack_multi_board(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        max_boards: int = 10_000,
        order: str = "size",
        place_mode: str = "blf",
        rng: Optional[random.Random] = None,
        rand_topk: int = 3,
) -> List[Board]:
    left = parts[:]
    boards: List[Board] = []
    bid = 1
    while left:
        if bid > max_boards:
            raise RuntimeError("Max boards limit reached")
        board, left2 = pack_one_board(
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
        if len(board.placed) == 0:
            raise RuntimeError(f"Cannot place part: uid={left[0].uid}")
        boards.append(board)
        left = left2
        bid += 1
    return boards


def pack_baselineA(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        place_mode: str = "blf",
) -> List[Board]:
    return pack_multi_board(
        parts,
        W,
        H,
        allow_rot,
        trim=trim,
        safe_gap=safe_gap,
        touch_tol=touch_tol,
        order="size",
        place_mode=place_mode,
    )


def _avg_util(boards: List[Board]) -> float:
    if not boards:
        return 0.0
    s = 0.0
    for b in boards:
        used = sum(pp.rect.w * pp.rect.h for pp in b.placed)
        s += used / (b.W * b.H)
    return s / len(boards)


def _lex_key_boards(boards: List[Board]) -> Tuple[int, float]:
    # Lexicographic objective: minimize #boards, then maximize utilization.
    return (len(boards), -_avg_util(boards))


def _lns_destroy_repair_sequence(seq: List, rng: random.Random, destroy_frac: float = 0.08) -> List:
    n = len(seq)
    if n <= 4:
        return seq[:]
    frac = max(0.02, min(0.30, float(destroy_frac)))
    k = max(2, min(n - 1, int(round(n * frac))))

    cand = seq[:]
    idxs = sorted(rng.sample(range(n), k), reverse=True)
    removed = [cand.pop(i) for i in idxs]
    removed.sort(key=lambda p: p.w * p.h, reverse=True)

    for p in removed:
        side = max(float(p.w), float(p.h))
        area = float(p.w) * float(p.h)
        pos_set = {0, len(cand)}
        for _ in range(min(4, len(cand) + 1)):
            pos_set.add(rng.randrange(len(cand) + 1))

        best_pos = None
        best_score = None
        for pos in pos_set:
            score = 0.0
            if pos > 0:
                left = cand[pos - 1]
                score += abs(max(left.w, left.h) - side) + 1e-3 * abs(left.w * left.h - area)
            if pos < len(cand):
                right = cand[pos]
                score += abs(max(right.w, right.h) - side) + 1e-3 * abs(right.w * right.h - area)
            score += 1e-6 * rng.random()
            if best_score is None or score < best_score:
                best_score = score
                best_pos = pos
        cand.insert(int(best_pos if best_pos is not None else len(cand)), p)
    return cand


def pack_baselineB(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        restarts: int = 50,
        seed: int = 0,
        rand_place: bool = True,
        rand_topk: int = 3,
        place_mode_override: str = "",
) -> List[Board]:
    rng = random.Random(seed)
    base = parts[:]

    def key_size(p):
        return (max(p.w, p.h), p.w * p.h)

    def key_area(p):
        return (p.w * p.h, max(p.w, p.h))

    def key_maxside(p):
        return (max(p.w, p.h), p.w * p.h)

    def key_minside(p):
        return (min(p.w, p.h), p.w * p.h)

    def key_ratio(p):
        return (max(p.w, p.h) / max(EPS, min(p.w, p.h)), p.w * p.h)

    recipes: List[Callable[[List], List]] = []
    recipes.append(lambda arr: sorted(arr, key=key_size, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_area, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_maxside, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_ratio, reverse=True))
    recipes.append(lambda arr: sorted(arr, key=key_minside, reverse=True))

    def _shuffle(arr):
        arr2 = arr[:]
        rng.shuffle(arr2)
        return arr2

    recipes.append(_shuffle)

    def _noisy_size(arr):
        return sorted(arr, key=lambda p: (key_size(p), rng.random()), reverse=True)

    recipes.append(_noisy_size)

    if place_mode_override:
        place_mode = str(place_mode_override)
    else:
        place_mode = "blf_rand" if rand_place else "blf"
    arr0 = recipes[0](base)
    boards0 = pack_multi_board(arr0, W, H, allow_rot, trim=trim, safe_gap=safe_gap, touch_tol=touch_tol, order="none", place_mode=place_mode, rng=rng, rand_topk=rand_topk)
    best: Optional[List[Board]] = boards0
    best_key = _lex_key_boards(boards0)

    R = max(1, int(restarts))
    for t in range(1, R):
        rec = recipes[t % len(recipes)]
        arr = rec(base)
        boards = pack_multi_board(arr, W, H, allow_rot, trim=trim, safe_gap=safe_gap, touch_tol=touch_tol, order="none", place_mode=place_mode, rng=rng, rand_topk=rand_topk)
        key = _lex_key_boards(boards)
        if key < best_key:
            best = boards
            best_key = key
    return best


# --- 新增：Stage 1 SA 序列优化器 ---

def pack_sequence_sa(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        seed: int,
        iters: int = 2000,
        T0: float = 0.02,
        alpha: float = 0.995,
        place_mode: str = "blf",
        lns_enable: bool = False,
        lns_prob: float = 0.15,
        lns_destroy_frac: float = 0.08,
) -> List[Board]:
    """Simulated Annealing to optimize input sequence of parts.

    Objective: Minimize N_boards, then Maximize Utilization.
    """
    rng = random.Random(seed)

    # 1) Initial solution (sort by area desc)
    current_seq = sorted(parts, key=lambda p: p.w * p.h, reverse=True)

    current_boards = pack_multi_board(
        current_seq,
        W,
        H,
        allow_rot,
        trim=trim,
        safe_gap=safe_gap,
        touch_tol=touch_tol,
        order="none",
        place_mode=place_mode,
    )
    current_u = _avg_util(current_boards)
    current_n = len(current_boards)

    best_boards = current_boards
    best_key = _lex_key_boards(current_boards)

    T = float(T0)
    n_parts = len(parts)

    for _ in range(int(iters)):
        if n_parts < 2:
            break
        if bool(lns_enable) and n_parts >= 6 and rng.random() < max(0.0, min(1.0, float(lns_prob))):
            new_seq = _lns_destroy_repair_sequence(current_seq, rng, destroy_frac=lns_destroy_frac)
        else:
            idx1, idx2 = rng.sample(range(n_parts), 2)
            new_seq = current_seq[:]
            new_seq[idx1], new_seq[idx2] = new_seq[idx2], new_seq[idx1]

        try:
            new_boards = pack_multi_board(
                new_seq,
                W,
                H,
                allow_rot,
                trim=trim,
                safe_gap=safe_gap,
                touch_tol=touch_tol,
                order="none",
                place_mode=place_mode,
            )
        except RuntimeError:
            continue

        new_u = _avg_util(new_boards)
        new_n = len(new_boards)
        new_key = _lex_key_boards(new_boards)
        cur_key = (current_n, -current_u)

        if new_key[0] < cur_key[0]:
            accept = True
        elif new_key[0] > cur_key[0]:
            accept = False
        elif new_key[1] < cur_key[1]:
            accept = True
        else:
            delta_u = new_u - current_u
            accept = (rng.random() < math.exp(delta_u / max(1e-9, T)))

        if accept:
            current_seq = new_seq
            current_boards = new_boards
            current_n = new_n
            current_u = new_u

            if _lex_key_boards(current_boards) < best_key:
                best_boards = current_boards
                best_key = _lex_key_boards(current_boards)

        T *= float(alpha)

    return best_boards


# --- 新增：Stage 1 ALNS（自适应大邻域搜索）序列优化器 ---

def _solution_fitness(
        boards: List[Board],
        *,
        board_W: float,
        board_H: float,
        feed_cut: float,
        feed_air: float,
        t_lift: float,
        share_mode: str,
        cut_mode: str,
        tool_d: float = 0.0,
        kerf_mode: str = "none",
        lead_in: float = 0.0,
        lead_out: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        # routing process rules
        route_hierarchical: bool = False,
        route_large_frac: float = 0.20,
        route_ramp_enable: bool = False,
        route_ramp_len: float = 0.0,
        anti_shift_enable: bool = False,
        anti_shift_area_m2: float = 0.05,
        anti_shift_ar: float = 5.0,
        anti_shift_two_pass: bool = False,
        # weights
        w1: float = 1.0,
        w2: float = 0.2,
        w3: float = 0.1,
) -> Tuple[float, Dict[str, float]]:
    """Return (fitness, components).

    Fitness is minimized:
        w1*(1-U) + w2*(1 - L_shared/L_total) + w3*(T_move/T_total)
    """
    # Local import to avoid any risk of circular imports during package init.
    from .metrics import compute_board_metrics

    if not boards:
        return 1e9, {"U": 0.0, "share": 0.0, "move_ratio": 1.0}

    U_list = []
    L_shared_sum = 0.0
    L_total_sum = 0.0
    T_move_sum = 0.0
    T_total_sum = 0.0

    for b in boards:
        row = compute_board_metrics(
            b,
            board_W=board_W,
            board_H=board_H,
            feed_cut=feed_cut,
            feed_air=feed_air,
            t_lift=t_lift,
            share_mode=share_mode,
            cut_mode=cut_mode,
            tool_d=tool_d,
            kerf_mode=kerf_mode,
            lead_in=lead_in,
            lead_out=lead_out,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            route_hierarchical=route_hierarchical,
            route_large_frac=route_large_frac,
            route_ramp_enable=route_ramp_enable,
            route_ramp_len=route_ramp_len,
            anti_shift_enable=anti_shift_enable,
            anti_shift_area_m2=anti_shift_area_m2,
            anti_shift_ar=anti_shift_ar,
            anti_shift_two_pass=anti_shift_two_pass,
            nd_coord=nd_coord,
        )
        U_list.append(float(row.get("U", 0.0)))
        L_shared_sum += float(row.get("L_shared", 0.0))
        L_total_sum += float(row.get("L_cut", 0.0))
        T_move_sum += float(row.get("T_air", 0.0))
        T_total_sum += float(row.get("T_est", 0.0))

    U = sum(U_list) / max(1, len(U_list))
    share = (L_shared_sum / max(EPS, L_total_sum)) if L_total_sum > 0 else 0.0
    move_ratio = (T_move_sum / max(EPS, T_total_sum)) if T_total_sum > 0 else 1.0

    fitness = float(w1) * (1.0 - U) + float(w2) * (1.0 - share) + float(w3) * (move_ratio)
    return float(fitness), {"U": float(U), "share": float(share), "move_ratio": float(move_ratio)}


def _shared_per_part(boards: List[Board]) -> Dict[int, float]:
    """Compute per-part shared-edge contribution (proxy) from current placement."""
    contrib = defaultdict(float)
    for b in boards:
        placed = b.placed
        n = len(placed)
        for i in range(n):
            ri = placed[i].rect
            ui = placed[i].uid
            for j in range(i + 1, n):
                rj = placed[j].rect
                uj = placed[j].uid
                s = _touch_shared_len(ri, rj)
                if s > 0.0:
                    contrib[ui] += s
                    contrib[uj] += s
    return dict(contrib)


def _destroy_stability(boards: List[Board], rng: random.Random, k: int) -> List[int]:
    """Remove unstable parts: small area + large aspect ratio."""
    scored = []
    for b in boards:
        for pp in b.placed:
            w, h = float(pp.w0), float(pp.h0)
            area = w * h
            ar = max(w, h) / max(EPS, min(w, h))
            score = (ar / max(EPS, area))
            scored.append((score, pp.uid))
    scored.sort(reverse=True)
    pick = [uid for _, uid in scored[:k]]
    rng.shuffle(pick)
    return pick


def _destroy_non_cec(boards: List[Board], rng: random.Random, k: int) -> List[int]:
    """Remove parts that contribute little/no shared edge."""
    contrib = _shared_per_part(boards)
    scored = []
    for b in boards:
        for pp in b.placed:
            s = float(contrib.get(pp.uid, 0.0))
            scored.append((s, pp.uid))
    scored.sort(key=lambda t: (t[0], rng.random()))
    return [uid for _, uid in scored[:k]]


def _destroy_cluster(boards: List[Board], rng: random.Random, k: int) -> List[int]:
    """Remove parts in a random spatial neighborhood."""
    all_pp = []
    for b in boards:
        for pp in b.placed:
            r = pp.rect
            cx = float(r.x + 0.5 * r.w)
            cy = float(r.y + 0.5 * r.h)
            all_pp.append((cx, cy, pp.uid))
    if not all_pp:
        return []
    cx0, cy0, _ = rng.choice(all_pp)
    all_pp.sort(key=lambda t: (t[0] - cx0) ** 2 + (t[1] - cy0) ** 2)
    pick = [uid for *_xy, uid in all_pp[:k]]
    rng.shuffle(pick)
    return pick



def _destroy_gap_target(boards: List[Board], rng: random.Random, k: int) -> List[int]:
    """Gap-driven destroy (MaxRects free-rect targeting).

    Idea:
      - find a large free-rectangle (gap) left by MaxRects
      - remove parts close to this gap so that repair can re-pack and 'heal' fragmentation.
    """
    gaps = []
    for b in boards:
        for fr in (b.free_rects or []):
            area = float(fr.w) * float(fr.h)
            if area > 1e-6:
                gaps.append((area, b, fr))
    if not gaps:
        # Fallback for non-MaxRects placement modes
        return _destroy_cluster(boards, rng, k)

    gaps.sort(key=lambda t: t[0], reverse=True)
    top = gaps[: min(10, len(gaps))]

    # Weighted random choose a gap by area
    total = sum(a for a, *_ in top)
    r = rng.random() * max(EPS, total)
    acc = 0.0
    chosen = top[0]
    for a, b, fr in top:
        acc += a
        if acc >= r:
            chosen = (a, b, fr)
            break

    gap_area, _b0, g = chosen

    scored = []
    for b in boards:
        for pp in b.placed:
            pr = pp.rect
            d2 = _jit_distance_sq(pr.x, pr.y, pr.w, pr.h, g.x, g.y, g.w, g.h)
            d = math.sqrt(max(0.0, float(d2)))
            # closer parts are more responsible for the gap fragmentation
            score = float(gap_area) / (1e-6 + d)
            scored.append((score, pp.uid))
    scored.sort(key=lambda t: (t[0], rng.random()), reverse=True)
    pick = [uid for _, uid in scored[:k]]
    rng.shuffle(pick)
    return pick

def _repair_cec_greedy(seq_keep: List, removed: List, rng: random.Random) -> List:
    """CEC-greedy repair on the sequence (insertion bias)."""
    cand = seq_keep[:]
    removed2 = removed[:]
    removed2.sort(key=lambda p: p.w * p.h, reverse=True)

    for p in removed2:
        side = max(float(p.w), float(p.h))
        area = float(p.w) * float(p.h)
        ar = side / max(EPS, min(float(p.w), float(p.h)))

        pos_set = {0, len(cand)}
        for _ in range(min(8, len(cand) + 1)):
            pos_set.add(rng.randrange(len(cand) + 1))

        best_pos = None
        best_score = None
        for pos in pos_set:
            score = 0.0
            if pos > 0:
                left = cand[pos - 1]
                score += abs(max(left.w, left.h) - side) + 1e-3 * abs(left.w * left.h - area)
            if pos < len(cand):
                right = cand[pos]
                score += abs(max(right.w, right.h) - side) + 1e-3 * abs(right.w * right.h - area)
            score += 0.05 * ar
            score += 1e-6 * rng.random()
            if best_score is None or score < best_score:
                best_score = score
                best_pos = pos

        cand.insert(int(best_pos if best_pos is not None else len(cand)), p)

    return cand


def _repair_maxrects_fill(seq_keep: List, removed: List, rng: random.Random) -> List:
    """MaxRects-fill repair: place big removed parts early, randomize the tail."""
    big = sorted(removed, key=lambda p: p.w * p.h, reverse=True)
    tail = seq_keep[:]
    rng.shuffle(tail)
    return big + tail



def _repair_geo_cec(seq_keep: List, removed: List, rng: random.Random) -> Dict[str, object]:
    """Geometric CEC repair (placement-mode override).

    We re-pack using `place_mode='blf_ccf'`, which evaluates shared-edge + border contact
    at candidate points (true geometric signal), instead of only biasing insertion order.
    """
    big = sorted(removed, key=lambda p: p.w * p.h, reverse=True)
    tail = seq_keep[:]
    rng.shuffle(tail)
    seq = big + tail
    return {"seq": seq, "place_mode": "blf_ccf"}

def pack_sequence_alns(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        seed: int,
        iters: int = 600,
        start_place_mode: str = "maxrects_baf",
        # Fitness weights
        w1: float = 1.0,
        w2: float = 0.25,
        w3: float = 0.10,
        # Metrics config
        feed_cut: float = 12000.0,
        feed_air: float = 30000.0,
        t_lift: float = 0.8,
        share_mode: str = "union",
        cut_mode: str = "trail",
        tool_d: float = 0.0,
        kerf_mode: str = "none",
        lead_in: float = 0.0,
        lead_out: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        # Routing process rules (affects move_ratio term)
        route_hierarchical: bool = False,
        route_large_frac: float = 0.20,
        route_ramp_enable: bool = False,
        route_ramp_len: float = 0.0,
        anti_shift_enable: bool = False,
        anti_shift_area_m2: float = 0.05,
        anti_shift_ar: float = 5.0,
        anti_shift_two_pass: bool = False,
        # ALNS control
        destroy_frac: float = 0.10,
        destroy_min: int = 6,
        destroy_max: int = 120,
        segment_len: int = 50,
        reaction: float = 0.2,
        T0: float = 0.05,
        alpha: float = 0.995,
) -> List[Board]:
    """ALNS for global sequencing to improve utilization + shared edge + air-move ratio."""

    rng = random.Random(seed)
    uid2part = {p.uid: p for p in parts}
    seq0 = sorted(parts, key=lambda p: p.w * p.h, reverse=True)

    def _pack(seq: List, place_mode_override: Optional[str] = None) -> List[Board]:
        pm = str(place_mode_override or start_place_mode)
        return pack_multi_board(
            seq,
            W,
            H,
            allow_rot,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            order="none",
            place_mode=pm,
            rng=rng,
            rand_topk=3,
        )

    cur_seq = seq0
    cur_boards = _pack(cur_seq)
    cur_fit, _ = _solution_fitness(
        cur_boards,
        board_W=W,
        board_H=H,
        feed_cut=feed_cut,
        feed_air=feed_air,
        t_lift=t_lift,
        share_mode=share_mode,
        cut_mode=cut_mode,
        tool_d=tool_d,
        kerf_mode=kerf_mode,
        lead_in=lead_in,
        lead_out=lead_out,
        line_snap_eps=line_snap_eps,
        min_shared_len=min_shared_len,
        nd_coord=nd_coord,
        route_hierarchical=route_hierarchical,
        route_large_frac=route_large_frac,
        route_ramp_enable=route_ramp_enable,
        route_ramp_len=route_ramp_len,
        anti_shift_enable=anti_shift_enable,
        anti_shift_area_m2=anti_shift_area_m2,
        anti_shift_ar=anti_shift_ar,
        anti_shift_two_pass=anti_shift_two_pass,
        w1=w1,
        w2=w2,
        w3=w3,
    )

    best_seq = cur_seq
    best_boards = cur_boards
    best_fit = cur_fit

    destroy_ops = [
        ("stability", _destroy_stability),
        ("non_cec", _destroy_non_cec),
        ("cluster", _destroy_cluster),
        ("gap_target", _destroy_gap_target),
    ]
    repair_ops = [
        ("cec_greedy", _repair_cec_greedy),
        ("maxrects_fill", _repair_maxrects_fill),
        ("geo_cec", _repair_geo_cec),
    ]

    wD = {name: 1.0 for name, _ in destroy_ops}
    wR = {name: 1.0 for name, _ in repair_ops}
    sD = {name: 0.0 for name, _ in destroy_ops}
    sR = {name: 0.0 for name, _ in repair_ops}
    cD = {name: 0 for name, _ in destroy_ops}
    cR = {name: 0 for name, _ in repair_ops}

    def _pick(weights: Dict[str, float]) -> str:
        total = sum(max(0.0, v) for v in weights.values())
        if total <= 0:
            return rng.choice(list(weights.keys()))
        r = rng.random() * total
        acc = 0.0
        for k2, v2 in weights.items():
            acc += max(0.0, v2)
            if acc >= r:
                return k2
        return rng.choice(list(weights.keys()))

    T = float(T0)
    n = len(parts)
    seg = max(10, int(segment_len))
    react = max(0.0, min(1.0, float(reaction)))

    for it in range(int(iters)):
        frac = max(0.02, min(0.30, float(destroy_frac)))
        k = int(round(frac * n))
        k = max(int(destroy_min), k)
        k = min(int(destroy_max), max(2, k))
        k = min(k, max(1, n - 1))

        dname = _pick(wD)
        rname = _pick(wR)
        dop = dict(destroy_ops)[dname]
        rop = dict(repair_ops)[rname]

        rm_uids = dop(cur_boards, rng, k)
        rm_set = set(rm_uids)
        if not rm_set:
            continue
        keep = [p for p in cur_seq if p.uid not in rm_set]
        removed = [uid2part[uid] for uid in rm_uids if uid in uid2part]
        if not removed:
            continue

        cand_pack = rop(keep, removed, rng)
        cand_place_mode = None
        if isinstance(cand_pack, dict):
            cand_seq = list(cand_pack.get("seq", []))
            cand_place_mode = cand_pack.get("place_mode", None)
        else:
            cand_seq = cand_pack

        if len(cand_seq) >= 2 and rng.random() < 0.10:
            i, j = rng.sample(range(len(cand_seq)), 2)
            cand_seq[i], cand_seq[j] = cand_seq[j], cand_seq[i]

        try:
            cand_boards = _pack(cand_seq, place_mode_override=cand_place_mode)
        except RuntimeError:
            continue

        cand_fit, _ = _solution_fitness(
            cand_boards,
            board_W=W,
            board_H=H,
            feed_cut=feed_cut,
            feed_air=feed_air,
            t_lift=t_lift,
            share_mode=share_mode,
            cut_mode=cut_mode,
            tool_d=tool_d,
            kerf_mode=kerf_mode,
            lead_in=lead_in,
            lead_out=lead_out,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
            route_hierarchical=route_hierarchical,
            route_large_frac=route_large_frac,
            route_ramp_enable=route_ramp_enable,
            route_ramp_len=route_ramp_len,
            anti_shift_enable=anti_shift_enable,
            anti_shift_area_m2=anti_shift_area_m2,
            anti_shift_ar=anti_shift_ar,
            anti_shift_two_pass=anti_shift_two_pass,
            w1=w1,
            w2=w2,
            w3=w3,
        )

        improved = cand_fit + 1e-12 < best_fit
        if cand_fit <= cur_fit:
            accept = True
        else:
            accept = (rng.random() < math.exp(-(cand_fit - cur_fit) / max(1e-9, T)))

        # reward uses *candidate* comparison to current/best
        reward = 0.0
        if improved:
            best_fit = cand_fit
            best_seq = cand_seq
            best_boards = cand_boards
            reward = 5.0
        elif accept:
            reward = 1.0

        if accept:
            cur_seq = cand_seq
            cur_boards = cand_boards
            cur_fit = cand_fit

        sD[dname] += reward
        sR[rname] += reward
        cD[dname] += 1
        cR[rname] += 1

        if (it + 1) % seg == 0:
            for name in wD:
                if cD[name] > 0:
                    wD[name] = (1.0 - react) * wD[name] + react * (sD[name] / max(1.0, cD[name]))
                sD[name] = 0.0
                cD[name] = 0
            for name in wR:
                if cR[name] > 0:
                    wR[name] = (1.0 - react) * wR[name] + react * (sR[name] / max(1.0, cR[name]))
                sR[name] = 0.0
                cR[name] = 0

        T *= float(alpha)

    return best_boards


def _board_score_shared_border_trail(
        board: Board,
        *,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
) -> Tuple[float, float, int]:
    # Fast proxy during SA: approximate shared/border without full routing.
    # L_shared: sum of touching lengths between parts on this board.
    placed = board.placed
    n = len(placed)
    L_shared = 0.0
    for i in range(n):
        ri = placed[i].rect
        for j in range(i + 1, n):
            rj = placed[j].rect
            L_shared += _touch_shared_len(ri, rj)

    # Border contact
    border = sum(_border_contact(pp.rect, board) for pp in board.placed)
    # Stroke count proxy: use number of placed parts (rough upper bound)
    n_stroke = n
    return float(L_shared), float(border), int(n_stroke)


def _repack_fixed_parts_ccf(
        parts_on_board: List,
        W: float,
        H: float,
        allow_rot: bool,
        bid: int,
        restarts: int,
        seed: int,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        border_w: float = 0.05,
        trail_penalty: float = 0.02,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        method: str = "restarts",
        sa_iters: int = 600,
        sa_T0: float = 5000.0,
        sa_alpha: float = 0.995,
        sa_max_starts: int = 5,
        sa_patience: int = 200,
) -> Optional[Board]:
    rng = random.Random(seed)
    base = sorted(parts_on_board, key=lambda p: (max(p.w, p.h), p.w * p.h), reverse=True)
    best_board: Optional[Board] = None
    best_key = None

    def _eval_order(arr) -> Tuple[Optional[Board], Optional[Tuple[float, float, float, int]]]:
        b, remaining = pack_one_board(arr, W, H, allow_rot, bid=bid, trim=trim, safe_gap=safe_gap, touch_tol=touch_tol, order="none", place_mode="blf_ccf")
        if remaining:
            return None, None

        shared, border, n_stroke = _board_score_shared_border_trail(
            b,
            cut_mode=cut_mode,
            tab_enable=tab_enable,
            tab_per_part=tab_per_part,
            tab_len=tab_len,
            tab_corner_clear=tab_corner_clear,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
        )
        score = float(shared) + float(border_w) * float(border) - float(trail_penalty) * float(n_stroke)
        key = (-score, -shared, -border, n_stroke)
        return b, (score, shared, border, n_stroke, key)

    def _update_best(b: Board, info) -> bool:
        """Update global best board/key. Return True if improved."""
        nonlocal best_board, best_key
        _score, _shared, _border, _n_stroke, key = info
        improved = (best_board is None or key < best_key)
        if improved:
            best_board, best_key = b, key
        return improved

    method = (method or "restarts").lower().strip()
    starts = max(1, int(restarts))
    if method == "sa":
        starts = min(starts, max(1, int(sa_max_starts)))

    for t in range(starts):
        cur_order = base[:]
        if t > 0:
            rng.shuffle(cur_order)
        b0, info0 = _eval_order(cur_order)
        if b0 is None:
            continue
        _update_best(b0, info0)
        no_improve = 0

        if method != "sa" or sa_iters <= 0 or len(cur_order) < 2:
            continue

        cur_info = info0
        cur_score = float(cur_info[0])
        T = float(sa_T0)
        n = len(cur_order)
        iters2 = int(sa_iters)

        for _ in range(iters2):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand_order = cur_order[:]
            cand_order[i], cand_order[j] = cand_order[j], cand_order[i]

            b1, info1 = _eval_order(cand_order)
            if b1 is None:
                T *= float(sa_alpha)
                no_improve += 1
                if no_improve >= int(sa_patience):
                    break
                continue

            cand_score = float(info1[0])
            delta = cand_score - cur_score

            if delta >= 0.0:
                accept = True
            else:
                accept = (rng.random() < math.exp(delta / max(1e-9, T)))

            if accept:
                cur_order = cand_order
                cur_info = info1
                cur_score = cand_score
                if _update_best(b1, cur_info):
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            if no_improve >= int(sa_patience):
                break

            T *= float(sa_alpha)

    return best_board


def pack_proposed_shared(
        parts: List,
        W: float,
        H: float,
        allow_rot: bool,
        *,
        trim: float = 0.0,
        safe_gap: float = 0.0,
        touch_tol: float = 1e-6,
        restarts: int = 10,
        seed: int = 0,
        inner_restarts: int = 30,
        border_w: float = 0.05,
        trail_penalty: float = 0.02,
        cut_mode: str = "trail",
        tab_enable: bool = False,
        tab_per_part: int = 0,
        tab_len: float = 0.0,
        tab_corner_clear: float = 0.0,
        line_snap_eps: float = 0.0,
        min_shared_len: float = 0.0,
        nd_coord: int = 6,
        stage2_method: str = "restarts",
        stage2_sa_iters: int = 600,
        stage2_sa_T0: float = 5000.0,
        stage2_sa_alpha: float = 0.995,
        stage2_sa_max_starts: int = 5,
        stage2_sa_patience: int = 200,
        stage2_sa_auto: bool = False,
        stage2_sa_base_n: int = 200,
        stage2_sa_min_iters: int = 10,
        stage1_method: str = "sa",
        stage1_sa_iters: int = 2000,
        stage1_place_mode: str = "blf",
        stage1_lns_enable: bool = False,
        stage1_lns_prob: float = 0.15,
        stage1_lns_destroy_frac: float = 0.08,
        # Stage1 ALNS (optional)
        stage1_alns_iters: int = 600,
        alns_w1: float = 1.0,
        alns_w2: float = 0.25,
        alns_w3: float = 0.10,
        alns_destroy_frac: float = 0.10,
        alns_destroy_min: int = 6,
        alns_destroy_max: int = 120,
        alns_segment_len: int = 50,
        alns_reaction: float = 0.2,
        alns_T0: float = 0.05,
        alns_alpha: float = 0.995,
        # Fitness evaluation params (align with cfg/metrics)
        feed_cut: float = 12000.0,
        feed_air: float = 30000.0,
        t_lift: float = 0.8,
        tool_d: float = 0.0,
        kerf_mode: str = "none",
        lead_in: float = 0.0,
        lead_out: float = 0.0,
        # Routing process rules (for fitness move_ratio term)
        route_hierarchical: bool = False,
        route_large_frac: float = 0.20,
        route_ramp_enable: bool = False,
        route_ramp_len: float = 0.0,
        anti_shift_enable: bool = False,
        anti_shift_area_m2: float = 0.05,
        anti_shift_ar: float = 5.0,
        anti_shift_two_pass: bool = False,
) -> List[Board]:
    """Proposed(shared) Route-2 Two-stage (Paper-friendly).

    Stage1: sequence optimization (SA) or BaselineB to get minimal N_board.
    Stage2: fixed-membership internal repacking (SA) for shared edges.

    NOTE: TRIM is applied as board feasible domain shrink (see Board.trim).
    """

    # Stage 1
    m1 = (stage1_method or "").lower().strip()
    if m1 == "alns":
        boards_stage1 = pack_sequence_alns(
            parts,
            W,
            H,
            allow_rot,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            seed=seed,
            iters=int(stage1_alns_iters),
            start_place_mode=stage1_place_mode,
            w1=float(alns_w1),
            w2=float(alns_w2),
            w3=float(alns_w3),
            feed_cut=float(feed_cut),
            feed_air=float(feed_air),
            t_lift=float(t_lift),
            share_mode="union",
            cut_mode=cut_mode,
            tool_d=float(tool_d),
            kerf_mode=str(kerf_mode),
            lead_in=float(lead_in),
            lead_out=float(lead_out),
            line_snap_eps=float(line_snap_eps),
            min_shared_len=float(min_shared_len),
            nd_coord=int(nd_coord),
            route_hierarchical=bool(route_hierarchical),
            route_large_frac=float(route_large_frac),
            route_ramp_enable=bool(route_ramp_enable),
            route_ramp_len=float(route_ramp_len),
            anti_shift_enable=bool(anti_shift_enable),
            anti_shift_area_m2=float(anti_shift_area_m2),
            anti_shift_ar=float(anti_shift_ar),
            anti_shift_two_pass=bool(anti_shift_two_pass),
            destroy_frac=float(alns_destroy_frac),
            destroy_min=int(alns_destroy_min),
            destroy_max=int(alns_destroy_max),
            segment_len=int(alns_segment_len),
            reaction=float(alns_reaction),
            T0=float(alns_T0),
            alpha=float(alns_alpha),
        )
    elif m1 == "sa":
        boards_stage1 = pack_sequence_sa(
            parts,
            W,
            H,
            allow_rot,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            seed=seed,
            iters=stage1_sa_iters,
            place_mode=stage1_place_mode,
            lns_enable=stage1_lns_enable,
            lns_prob=stage1_lns_prob,
            lns_destroy_frac=stage1_lns_destroy_frac,
        )
    else:
        boards_stage1 = pack_baselineB(
            parts,
            W,
            H,
            allow_rot,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            restarts=restarts,
            seed=seed,
            place_mode_override=stage1_place_mode,
        )

    # Stage 2
    uid2part = {p.uid: p for p in parts}
    improved: List[Board] = []

    for b in boards_stage1:
        subset = [uid2part[pp.uid] for pp in b.placed]
        # Optional: scale the heavy stage-2 SA budget for large industrial boards.
        sa_iters_eff = int(stage2_sa_iters)
        if (stage2_method or "").lower().strip() == "sa" and bool(stage2_sa_auto):
            m = max(1, len(subset))
            base_n = max(1, int(stage2_sa_base_n))
            scaled = int(round(sa_iters_eff * float(base_n) / float(max(base_n, m))))
            sa_iters_eff = max(int(stage2_sa_min_iters), scaled)
        sa_pat_eff = max(10, int(stage2_sa_patience))
        b2 = _repack_fixed_parts_ccf(
            subset,
            W,
            H,
            allow_rot,
            bid=b.bid,
            restarts=inner_restarts,
            seed=seed * 100000 + b.bid,
            trim=trim,
            safe_gap=safe_gap,
            touch_tol=touch_tol,
            border_w=border_w,
            trail_penalty=trail_penalty,
            cut_mode=cut_mode,
            tab_enable=tab_enable,
            tab_per_part=tab_per_part,
            tab_len=tab_len,
            tab_corner_clear=tab_corner_clear,
            line_snap_eps=line_snap_eps,
            min_shared_len=min_shared_len,
            nd_coord=nd_coord,
            method=stage2_method,
            sa_iters=sa_iters_eff,
            sa_T0=stage2_sa_T0,
            sa_alpha=stage2_sa_alpha,
            sa_max_starts=stage2_sa_max_starts,
            sa_patience=sa_pat_eff,
        )
        improved.append(b2 if b2 is not None else b)

    return improved